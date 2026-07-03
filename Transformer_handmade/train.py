"""Training loop for Seq2Seq Transformer reproducing "Attention Is All You Need".

Key ingredients:
- NoamOpt learning-rate schedule (warmup + inverse-sqrt decay)
- Gradient accumulation to simulate large batches
- BPE tokenization
- Full WMT14 DE-EN dataset (~4.5M pairs)
"""

import warnings
import random
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to sys.path

import torch

# Reduce CUDA memory fragmentation (critical for large attention matrices)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from torch import nn

from Transformer_handmade.config import get_config
from Transformer_handmade.data import build_dataloaders
from Transformer_handmade.model import Seq2SeqTransformer, NoamOpt

# TF32 gives free ~2× matmul throughput on Ampere/Ada/Blackwell vs FP32
torch.set_float32_matmul_precision("high")

# Suppress the PyTorch nested-tensor prototype warning triggered internally by
# nn.Transformer when padding masks are provided.
warnings.filterwarnings("ignore", message=".*nested tensor.*prototype.*")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _serialize_config(config) -> dict:
    payload = dict(config.__dict__)
    payload["data_dir"] = str(config.data_dir)
    payload["artifact_dir"] = str(config.artifact_dir)
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def save_artifacts(
    model: Seq2SeqTransformer,
    noam: NoamOpt,
    step: int,
    config,
    src_vocab_size: int,
    tgt_vocab_size: int,
) -> None:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": noam.state_dict(),
            "step": step,
            "config": _serialize_config(config),
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
        },
        config.checkpoint_path,
    )


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cuda", enabled=False)
    )
    for batch in dataloader:
        src = batch.src.to(device, non_blocking=True)
        tgt = batch.tgt.to(device, non_blocking=True)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        with amp_ctx:
            logits = model(
                src,
                tgt_input,
                src_padding_mask=batch.src_padding_mask.to(device, non_blocking=True),
                tgt_padding_mask=batch.tgt_padding_mask[:, :-1].to(device, non_blocking=True),
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def train() -> Path:
    config = get_config()
    set_seed(config.seed)
    device = resolve_device(config.device)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"d_model: {config.d_model}  N: {config.N}  h: {config.h}  d_ff: {config.d_ff}")
    print(f"batch_size: {config.batch_size}  grad_accum: {config.grad_accum_steps}"
          f"  effective: {config.effective_batch_size}")
    print(f"warmup_steps: {config.warmup_steps}  total_steps: {config.steps}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"AMP: {config.use_amp} (bfloat16)")

    loaders, src_tokenizer, tgt_tokenizer = build_dataloaders(config)
    model = Seq2SeqTransformer(config, src_tokenizer.vocab_size, tgt_tokenizer.vocab_size).to(device)
    model.set_pad_ids(src_tokenizer.pad_id, tgt_tokenizer.pad_id)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"src vocab: {src_tokenizer.vocab_size}  tgt vocab: {tgt_tokenizer.vocab_size}")

    base_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,                                   # NoamOpt overwrites this
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    noam = NoamOpt(base_optimizer, config.d_model, config.warmup_steps)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_tokenizer.pad_id,
        label_smoothing=config.label_smoothing,
    )

    src_tokenizer.save(config.tokenizer_path)   # shared vocab → one file

    optimizer_step = 0
    batch_count = 0
    accum_loss = 0.0

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if config.use_amp and device.type == "cuda"
        else torch.amp.autocast("cuda", enabled=False)
    )

    while optimizer_step < config.steps:
        for batch in loaders["train"]:
            src = batch.src.to(device, non_blocking=True)
            tgt = batch.tgt.to(device, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with amp_ctx:
                logits = model(
                    src,
                    tgt_input,
                    src_padding_mask=batch.src_padding_mask.to(device, non_blocking=True),
                    tgt_padding_mask=batch.tgt_padding_mask[:, :-1].to(device, non_blocking=True),
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss = loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            batch_count += 1

            # Step only after accumulating enough gradients
            if batch_count % config.grad_accum_steps == 0:
                noam.step()
                noam.zero_grad()

                optimizer_step += 1
                if optimizer_step % config.log_every == 0 or optimizer_step == 1:
                    avg_loss = accum_loss / config.log_every if optimizer_step >= config.log_every else accum_loss
                    print(f"step={optimizer_step} lr={noam.current_lr:.6f} train_loss={avg_loss:.4f}")
                    accum_loss = 0.0

                if optimizer_step % config.eval_every == 0:
                    val_loss = evaluate(model, loaders["validation"], criterion, device, config.use_amp)
                    print(f"step={optimizer_step} val_loss={val_loss:.4f}")

                if optimizer_step % config.save_every == 0:
                    save_artifacts(
                        model, noam, optimizer_step, config,
                        src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
                    )

                if optimizer_step >= config.steps:
                    break

    save_artifacts(
        model, noam, optimizer_step, config,
        src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
    )
    print(f"checkpoint saved to {config.checkpoint_path}")
    return config.checkpoint_path


if __name__ == "__main__":
    train()

"""Training loop for Seq2Seq Transformer reproducing "Attention Is All You Need".

Key ingredients:
- NoamOpt learning-rate schedule (warmup + inverse-sqrt decay)
- Gradient accumulation to simulate large batches
- BPE tokenization
- Full WMT14 DE-EN dataset (~4.5M pairs)
"""

import os
import random
import warnings
from pathlib import Path

import argparse
import torch

# Reduce CUDA memory fragmentation (critical for large attention matrices)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from torch import nn

from Transformer_handmade.config import get_config
from Transformer_handmade.data import BPETokenizer, build_dataloaders
from Transformer_handmade.model import NoamOpt, Seq2SeqTransformer

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


def save_avg_snapshot(
    model: Seq2SeqTransformer,
    step: int,
    config,
    src_vocab_size: int,
    tgt_vocab_size: int,
    keep: int,
) -> None:
    """Write a lightweight snapshot for later checkpoint averaging."""
    snap_dir = config.artifact_dir / "avg_ckpts"
    snap_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": step,
            "config": _serialize_config(config),
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
        },
        snap_dir / f"step_{step}.pt",
    )
    snapshots = sorted(snap_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for stale in snapshots[:-keep]:
        stale.unlink()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-rolled Seq2Seq Transformer.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from config.checkpoint_path (model + optimizer + step), reusing its saved tokenizer instead of re-fitting BPE.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override config.steps.",
    )
    parser.add_argument(
        "--checkpoint-avg-every", type=int, default=1000,
        help="Steps between rotating snapshots kept for checkpoint averaging (0 disables).",
    )
    parser.add_argument(
        "--checkpoint-avg-keep", type=int, default=5,
        help="Number of most-recent snapshots to retain for averaging.",
    )
    parser.add_argument(
        "--max-tokens-per-batch", type=int, default=None,
        help="Override config.max_tokens_per_batch.",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Override config.grad_accum_steps.",
    )
    return parser.parse_args()

def train() -> Path:
    args = parse_args()
    config = get_config()
    if args.steps is not None:
        config.steps = args.steps
    if args.max_tokens_per_batch is not None:
        config.max_tokens_per_batch = args.max_tokens_per_batch
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
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

    resume_ckpt = None
    reuse_tokenizer = None
    if args.resume:
        if not config.checkpoint_path.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint at {config.checkpoint_path}")
        resume_ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
        reuse_tokenizer = BPETokenizer.load(config.tokenizer_path)
        print(f"Resuming from step {resume_ckpt['step']} (checkpoint: {config.checkpoint_path}, tokenizer: {config.tokenizer_path})")

    loaders, src_tokenizer, tgt_tokenizer = build_dataloaders(config, tokenizer=reuse_tokenizer)
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

    optimizer_step = 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        noam.load_state_dict(resume_ckpt["optimizer_state_dict"])
        optimizer_step = resume_ckpt["step"]
        if optimizer_step >= config.steps:
            print(f"Checkpoint already at step {optimizer_step} >= target {config.steps}; nothing to do.")
            return config.checkpoint_path
    else:
        src_tokenizer.save(config.tokenizer_path)   # shared vocab → one file

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

                if args.checkpoint_avg_every and optimizer_step % args.checkpoint_avg_every == 0:
                    save_avg_snapshot(
                        model, optimizer_step, config,
                        src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
                        keep=args.checkpoint_avg_keep,
                    )

                if optimizer_step >= config.steps:
                    break

    save_artifacts(
        model, noam, optimizer_step, config,
        src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
    )
    if args.checkpoint_avg_every:
        save_avg_snapshot(
            model, optimizer_step, config,
            src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
            keep=args.checkpoint_avg_keep,
        )
    print(f"checkpoint saved to {config.checkpoint_path}")
    return config.checkpoint_path


if __name__ == "__main__":
    train()

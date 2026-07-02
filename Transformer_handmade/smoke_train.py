"""Short real-data training smoke test.

Trains on a small WMT14 subset for a few hundred steps to confirm the
hand-written attention/FFN model learns (loss goes down). Writes all
artifacts to a temp dir so the real tokenizer/checkpoints are untouched.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch import nn

from Transformer_handmade.config import get_config
from Transformer_handmade.data import build_dataloaders
from Transformer_handmade.model import Seq2SeqTransformer, NoamOpt
from Transformer_handmade.train import evaluate, resolve_device, set_seed


def main() -> None:
    cfg = get_config()
    # --- shrink everything for a fast smoke run on REAL data ---
    cfg.artifact_dir = Path("Transformer_handmade/artifacts_smoke")
    cfg.train_samples = 50_000
    cfg.valid_samples = 1_000
    cfg.test_samples = 1_000
    cfg.vocab_size = 8_000
    cfg.max_seq_len = 64
    cfg.batch_size = 64
    cfg.grad_accum_steps = 1
    cfg.steps = 400
    cfg.warmup_steps = 100
    cfg.eval_every = 200
    cfg.log_every = 25
    cfg.num_workers = 4

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  | subset {cfg.train_samples} pairs, vocab {cfg.vocab_size}, {cfg.steps} steps")

    loaders, src_tok, tgt_tok = build_dataloaders(cfg)
    model = Seq2SeqTransformer(cfg, src_tok.vocab_size, tgt_tok.vocab_size).to(device)
    model.set_pad_ids(src_tok.pad_id, tgt_tok.pad_id)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  | vocab {src_tok.vocab_size}")

    base_opt = torch.optim.Adam(
        model.parameters(), lr=0,
        betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps,
    )
    noam = NoamOpt(base_opt, cfg.d_model, cfg.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id, label_smoothing=cfg.label_smoothing)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if cfg.use_amp and device.type == "cuda"
        else torch.amp.autocast("cuda", enabled=False)
    )

    first_loss = None
    step = 0
    running = 0.0
    while step < cfg.steps:
        for batch in loaders["train"]:
            src = batch.src.to(device, non_blocking=True)
            tgt = batch.tgt.to(device, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with amp_ctx:
                logits = model(
                    src, tgt_in,
                    src_padding_mask=batch.src_padding_mask.to(device, non_blocking=True),
                    tgt_padding_mask=batch.tgt_padding_mask[:, :-1].to(device, non_blocking=True),
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            noam.step()
            noam.zero_grad()

            step += 1
            running += loss.item()
            if first_loss is None:
                first_loss = loss.item()
            if step % cfg.log_every == 0:
                print(f"step={step:4d}  lr={noam.current_lr:.6f}  train_loss={running/cfg.log_every:.4f}")
                running = 0.0
            if step % cfg.eval_every == 0:
                vl = evaluate(model, loaders["validation"], criterion, device, cfg.use_amp)
                print(f"           >>> val_loss={vl:.4f}")
            if step >= cfg.steps:
                break

    final_val = evaluate(model, loaders["validation"], criterion, device, cfg.use_amp)
    print("=" * 50)
    print(f"first train_loss ≈ {first_loss:.4f}   final val_loss = {final_val:.4f}")
    print("loss decreased" if final_val < first_loss else "WARNING: loss did not decrease")


if __name__ == "__main__":
    main()

"""Test model by test-dataset and saved checkpoint.

Use BLEU score to evaluate the translation quality by sacrebleu.
Also includes unit tests for model components and tokenizer.
"""

import argparse
import warnings
from pathlib import Path

import sacrebleu
import torch

from Transformer_handmade.average_checkpoints import average_checkpoints
from Transformer_handmade.config import TransformerConfig, get_config
from Transformer_handmade.data import (
    BPETokenizer,
    TranslationDataset,
    build_dataloaders,
    load_parallel_records,
)
from Transformer_handmade.model import Seq2SeqTransformer

# Suppress the PyTorch nested-tensor prototype warning triggered internally by
# nn.Transformer when padding masks are provided.
warnings.filterwarnings("ignore", message=".*nested tensor.*prototype.*")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Seq2SeqTransformer, BPETokenizer, BPETokenizer, TransformerConfig, torch.device]:
    """Load model, tokenizers, and config from a saved checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    config_payload = ckpt.get("config", {})
    config = TransformerConfig(**config_payload) if config_payload else get_config()
    device = resolve_device(config.device)

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    src_tokenizer = tokenizer
    tgt_tokenizer = tokenizer

    model = Seq2SeqTransformer(config, ckpt["src_vocab_size"], ckpt["tgt_vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_pad_ids(src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    model.eval()
    return model, src_tokenizer, tgt_tokenizer, config, device


def is_avg_checkpoint_stale(avg_path: Path, snapshot_dir: Path, n: int) -> bool:
    if not avg_path.exists():
        return True

    snapshots = sorted(snapshot_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not snapshots:
        return True

    newest_snapshot_mtime = max(path.stat().st_mtime for path in snapshots[-n:])
    return avg_path.stat().st_mtime < newest_snapshot_mtime


def load_avg_checkpoint(
    avg_path: Path,
) -> tuple[Seq2SeqTransformer, BPETokenizer, BPETokenizer, TransformerConfig, torch.device]:
    return load_checkpoint(avg_path)


def load_checkpoints(
    checkpoint_paths: list[Path],
) -> tuple[list[Seq2SeqTransformer], BPETokenizer, BPETokenizer, TransformerConfig, torch.device]:
    """Load multiple checkpoints for logit-level ensemble decoding."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")

    models: list[Seq2SeqTransformer] = []
    src_tokenizer: BPETokenizer | None = None
    tgt_tokenizer: BPETokenizer | None = None
    base_config: TransformerConfig | None = None
    device: torch.device | None = None

    for path in checkpoint_paths:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        config_payload = ckpt.get("config", {})
        config = TransformerConfig(**config_payload) if config_payload else get_config()
        if base_config is None:
            base_config = config
            device = resolve_device(config.device)
            tokenizer = BPETokenizer.load(config.tokenizer_path)
            src_tokenizer = tokenizer
            tgt_tokenizer = tokenizer
        else:
            if config.tokenizer_path != base_config.tokenizer_path:
                print(f"[warn] tokenizer path differs for {path}; using {base_config.tokenizer_path}")

        model = Seq2SeqTransformer(config, ckpt["src_vocab_size"], ckpt["tgt_vocab_size"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.set_pad_ids(src_tokenizer.pad_id, tgt_tokenizer.pad_id)
        model.eval()
        models.append(model)

    assert src_tokenizer is not None and tgt_tokenizer is not None and base_config is not None and device is not None
    return models, src_tokenizer, tgt_tokenizer, base_config, device


# ---------------------------------------------------------------------------
# unit / smoke tests
# ---------------------------------------------------------------------------

def test_positional_encoding() -> None:
    """PositionalEncoding: shape, device persistence, value sanity."""
    from Transformer_handmade.model.transformer import PositionalEncoding

    d_model, max_len, dropout = 64, 32, 0.0
    pe = PositionalEncoding(d_model, dropout, max_len)
    x = torch.randn(2, 16, d_model)
    out = pe(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("[PASS] test_positional_encoding")


def test_generate_square_subsequent_mask() -> None:
    """Autoregressive mask: upper-triangle is True (blocked)."""
    config = get_config()
    model = Seq2SeqTransformer(config, src_vocab_size=100, tgt_vocab_size=100)
    mask = model.generate_square_subsequent_mask(4, torch.device("cpu"))
    assert not mask[0, 0].item()
    assert not mask[1, 0].item()
    assert not mask[1, 1].item()
    assert mask[0, 1].item()
    assert mask[0, 2].item()
    assert mask[0, 3].item()
    print("[PASS] test_generate_square_subsequent_mask")


def test_tokenizer_roundtrip() -> None:
    """tokenize → encode → decode survives identity on simple text."""
    tokenizer = BPETokenizer(lowercase=False)
    texts = ["hello world", "foo bar baz", "the quick brown fox jumps over the lazy dog"]
    tokenizer.fit(texts, vocab_size=200, min_freq=1)

    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"roundtrip failed: {text!r} → {decoded!r}"
    print("[PASS] test_tokenizer_roundtrip")


def test_tokenizer_specials() -> None:
    """Special tokens are consistent: pad/unk/bos/eos ids exist and differ."""
    tokenizer = BPETokenizer()
    tokenizer.fit(["sample"], vocab_size=200, min_freq=1)

    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.vocab_size >= 4
    print("[PASS] test_tokenizer_specials")


def test_dataloader_batch_shape() -> None:
    """Smoke-test: build_dataloaders produces correctly shaped batches."""
    config = get_config()
    config.train_samples = 32
    config.valid_samples = 16
    config.test_samples = 16
    config.batch_size = 4
    config.vocab_size = 200
    config.max_seq_len = 64

    loaders, _, _ = build_dataloaders(config)
    batch = next(iter(loaders["train"]))

    assert batch.src.ndim == 2
    assert batch.tgt.ndim == 2
    assert batch.src.shape[0] == batch.tgt.shape[0]
    assert batch.src_padding_mask.shape == batch.src.shape
    assert batch.tgt_padding_mask.shape == batch.tgt.shape
    print("[PASS] test_dataloader_batch_shape")


def test_model_forward() -> None:
    """Forward pass produces logits with shape (B, T_tgt, vocab)."""
    config = get_config()
    config.vocab_size = 200
    config.max_seq_len = 64
    src_vocab, tgt_vocab = 200, 200
    model = Seq2SeqTransformer(config, src_vocab, tgt_vocab)
    model.eval()

    src = torch.randint(4, src_vocab, (2, 16))
    tgt = torch.randint(4, tgt_vocab, (2, 12))
    src_mask = src.eq(0)
    tgt_mask = tgt.eq(0)

    with torch.no_grad():
        logits = model(src, tgt, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)

    assert logits.shape == (2, 12, tgt_vocab), f"unexpected logits shape {logits.shape}"
    print("[PASS] test_model_forward")


def test_greedy_decode_shape() -> None:
    """Greedy decode returns token ids in correct shape."""
    config = get_config()
    config.max_seq_len = 32
    src_vocab, tgt_vocab = 100, 100
    model = Seq2SeqTransformer(config, src_vocab, tgt_vocab)
    model.set_pad_ids(0, 0)
    model.eval()

    src = torch.randint(4, src_vocab, (2, 8))
    src_mask = src.eq(0)

    with torch.no_grad():
        out = model.greedy_decode(src, src_mask, bos_id=2, eos_id=3, max_len=16)

    assert out.ndim == 2
    assert out.shape[0] == src.shape[0]
    assert out.shape[1] <= 16
    assert (out[:, 0] == 2).all(), "first token must be BOS"
    print("[PASS] test_greedy_decode_shape")


def test_beam_search_shape() -> None:
    """Beam search returns token ids with BOS prefix."""
    config = get_config()
    config.max_seq_len = 32
    config.beam_size = 3
    src_vocab, tgt_vocab = 100, 100
    model = Seq2SeqTransformer(config, src_vocab, tgt_vocab)
    model.set_pad_ids(0, 0)
    model.eval()

    src = torch.randint(4, src_vocab, (1, 8))
    src_mask = src.eq(0)

    with torch.no_grad():
        out = model.beam_search_decode(
            src, src_mask, bos_id=2, eos_id=3, max_len=16,
            beam_size=3, length_penalty=0.6,
        )

    assert out.ndim == 2
    assert out.shape[0] == 1
    assert (out[:, 0] == 2).all(), "first token must be BOS"
    print("[PASS] test_beam_search_shape")


def test_shared_embeddings() -> None:
    """When share_embeddings=True, src/tgt/generator share weights."""
    config = get_config()
    vocab_size = 100
    model = Seq2SeqTransformer(config, vocab_size, vocab_size, share_embeddings=True)

    assert model.src_embedding.weight is model.tgt_embedding.weight
    assert model.src_embedding.weight is model.generator.weight
    print("[PASS] test_shared_embeddings")


def _ensemble_step_logits(
    models: list[Seq2SeqTransformer],
    generated: torch.Tensor,
    memories: list[torch.Tensor],
    src_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Average next-token logits across checkpoints for the current prefix."""
    logits_sum = None
    tgt_mask = models[0].generate_square_subsequent_mask(generated.size(1), generated.device)
    tgt_padding_mask = generated.eq(models[0].tgt_pad_id)

    for model, memory in zip(models, memories):
        decoder_output = model.transformer.decoder(
            model.embed_tgt(generated),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        step_logits = model.generator(decoder_output[:, -1])
        logits_sum = step_logits if logits_sum is None else logits_sum + step_logits

    return logits_sum / len(models)


@torch.no_grad()
def ensemble_greedy_decode(
    models: list[Seq2SeqTransformer],
    src: torch.Tensor,
    src_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> torch.Tensor:
    memories = [m.transformer.encoder(m.embed_src(src), src_key_padding_mask=src_padding_mask) for m in models]
    generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)

    for _ in range(max_len - 1):
        next_token_logits = _ensemble_step_logits(models, generated, memories, src_padding_mask)
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if torch.all(next_token.squeeze(1) == eos_id):
            break

    return generated


@torch.no_grad()
def ensemble_beam_search_decode(
    models: list[Seq2SeqTransformer],
    src: torch.Tensor,
    src_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    assert src.size(0) == 1, "beam_search_decode expects batch_size=1"
    device = src.device
    memories = [m.transformer.encoder(m.embed_src(src), src_key_padding_mask=src_padding_mask) for m in models]

    seqs = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, dtype=torch.float, device=device)
    completed: list[tuple[torch.Tensor, float]] = []

    for _ in range(max_len - 1):
        B, T = seqs.shape
        expanded_src_mask = src_padding_mask.expand(B, -1)
        expanded_memories = [memory.expand(B, -1, -1) for memory in memories]

        logits = _ensemble_step_logits(models, seqs, expanded_memories, expanded_src_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        topk_lp, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

        cand_scores = scores.unsqueeze(1) + topk_lp
        lp = ((5.0 + T + 1) / 6.0) ** length_penalty
        flat_normed = (cand_scores / lp).view(-1)
        flat_scores = cand_scores.view(-1)
        flat_ids = topk_ids.view(-1)
        parents = torch.arange(B, device=device).unsqueeze(1).expand(-1, beam_size).reshape(-1)

        top_k_idx = torch.topk(flat_normed, min(beam_size, flat_normed.numel())).indices

        next_seqs: list[torch.Tensor] = []
        next_scores: list[float] = []
        for idx in top_k_idx.tolist():
            p = parents[idx].item()
            tok = flat_ids[idx].item()
            sc = flat_scores[idx].item()
            new_seq = torch.cat([seqs[p], seqs.new_tensor([tok])])
            if tok == eos_id:
                completed.append((new_seq, sc))
            else:
                next_seqs.append(new_seq)
                next_scores.append(sc)

        if not next_seqs:
            break

        seqs = torch.stack(next_seqs)
        scores = seqs.new_tensor(next_scores, dtype=torch.float)
    else:
        for j in range(seqs.size(0)):
            completed.append((seqs[j], scores[j].item()))

    if not completed:
        return seqs[0:1] if seqs.size(0) > 0 else seqs.new_full((1, 1), eos_id)

    def _normed_score(c: tuple[torch.Tensor, float]) -> float:
        seq, sc = c
        return sc / ((5.0 + seq.size(0)) / 6.0) ** length_penalty

    best_seq, _ = max(completed, key=_normed_score)
    return best_seq.unsqueeze(0)


# ---------------------------------------------------------------------------
# BLEU evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bleu(
    model: Seq2SeqTransformer | list[Seq2SeqTransformer],
    dataset: TranslationDataset,
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    device: torch.device,
    max_len: int,
    beam_size: int,
    length_penalty: float,
    max_samples: int | None = None,
    output_path: str | None = None,
) -> dict:
    """Run beam-search decoding on `dataset` and compute corpus BLEU via sacrebleu."""
    import time
    ensemble = isinstance(model, list)
    if ensemble:
        for item in model:
            item.eval()
    else:
        model.eval()

    hypotheses: list[str] = []
    references: list[str] = []
    sources: list[str] = []

    total = min(len(dataset), max_samples) if max_samples else len(dataset)
    log_every = max(1, total // 20)
    t0 = time.time()
    for i in range(total):
        sample = dataset[i]
        src = sample["src_ids"].unsqueeze(0).to(device)
        src_ref = model[0] if ensemble else model
        src_mask = src.eq(src_ref.src_pad_id)

        if beam_size > 1:
            if ensemble:
                generated = ensemble_beam_search_decode(
                    model,
                    src=src,
                    src_padding_mask=src_mask,
                    bos_id=tgt_tokenizer.bos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    max_len=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                generated = model.beam_search_decode(
                    src=src,
                    src_padding_mask=src_mask,
                    bos_id=tgt_tokenizer.bos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    max_len=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
        else:
            if ensemble:
                generated = ensemble_greedy_decode(
                    model,
                    src=src,
                    src_padding_mask=src_mask,
                    bos_id=tgt_tokenizer.bos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    max_len=max_len,
                )
            else:
                generated = model.greedy_decode(
                    src=src,
                    src_padding_mask=src_mask,
                    bos_id=tgt_tokenizer.bos_id,
                    eos_id=tgt_tokenizer.eos_id,
                    max_len=max_len,
                )
        hyp = tgt_tokenizer.decode(generated[0].tolist())
        ref = sample["tgt_text"]
        src_text = src_tokenizer.decode(sample["src_ids"].tolist())
        hypotheses.append(hyp)
        references.append(ref)
        sources.append(src_text)

        if (i + 1) % log_every == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{i+1}/{total}] {rate:.1f} sent/s  ETA {eta:.0f}s", flush=True)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("SRC\tHYP\tREF\n")
            for src_text, hyp, ref in zip(sources, hypotheses, references):
                f.write(f"{src_text}\t{hyp}\t{ref}\n")
        print(f"Predictions saved to: {output_path}")

    return {
        "bleu": bleu.score,
        "bleu_str": str(bleu),
        "num_samples": len(hypotheses),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the PyTorch Seq2Seq Transformer — unit tests + BLEU evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Path to checkpoint .pt file. Repeat the flag to run logit-level ensemble (default: config.checkpoint_path).",
    )
    parser.add_argument(
        "--max-bleu-samples",
        type=int,
        default=256,
        help="Max test samples for BLEU (default: 256; set 0 for all).",
    )
    parser.add_argument(
        "--skip-unit",
        action="store_true",
        help="Skip unit tests, only run BLEU evaluation.",
    )
    parser.add_argument(
        "--skip-bleu",
        action="store_true",
        help="Skip BLEU evaluation, only run unit tests.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=4,
        help="Beam size for BLEU evaluation (default: 4, set 1 for greedy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as TSV (columns: SRC, HYP, REF).",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="Use the averaged checkpoint from the last N snapshots.",
    )
    parser.add_argument(
        "--avg-n",
        type=int,
        default=5,
        help="Number of latest snapshots to average when --avg is set.",
    )
    args = parser.parse_args()

    # ---- unit tests ----
    if not args.skip_unit:
        print("=" * 50)
        print("Running unit tests ...")
        print("=" * 50)
        test_positional_encoding()
        test_generate_square_subsequent_mask()
        test_tokenizer_roundtrip()
        test_tokenizer_specials()
        test_dataloader_batch_shape()
        test_model_forward()
        test_greedy_decode_shape()
        test_beam_search_shape()
        test_shared_embeddings()
        print()
        print("All unit tests passed!")
        print()

    # ---- BLEU evaluation ----
    if not args.skip_bleu:
        print("=" * 50)
        print("BLEU evaluation")
        print("=" * 50)

        base_config = get_config()
        checkpoint_paths = args.checkpoint or [base_config.checkpoint_path]
        if len(checkpoint_paths) == 1:
            checkpoint_path = checkpoint_paths[0]
        else:
            checkpoint_path = None

        for checkpoint_path_item in checkpoint_paths:
            if not checkpoint_path_item.exists():
                print(f"[FAIL] checkpoint not found: {checkpoint_path_item}")
                return

        if len(checkpoint_paths) == 1:
            print(f"Loading checkpoint: {checkpoint_paths[0]}")
            if args.avg:
                avg_path = base_config.artifact_dir / "pytorch_transformer_avg.pt"
                if is_avg_checkpoint_stale(avg_path, base_config.artifact_dir / "avg_ckpts", args.avg_n):
                    print(f"Rebuilding averaged checkpoint: {avg_path}")
                    average_checkpoints(base_config.artifact_dir / "avg_ckpts", args.avg_n, avg_path)
                else:
                    print(f"Using cached averaged checkpoint: {avg_path}")
                model, src_tok, tgt_tok, config, device = load_avg_checkpoint(avg_path)
            else:
                model, src_tok, tgt_tok, config, device = load_checkpoint(checkpoint_paths[0])
        else:
            print("Loading ensemble checkpoints:")
            for checkpoint_path_item in checkpoint_paths:
                print(f"  {checkpoint_path_item}")
            model, src_tok, tgt_tok, config, device = load_checkpoints(checkpoint_paths)
            print(f"  ensemble size: {len(model)}")
        print(f"  device:     {device}")
        print(f"  d_model:    {config.d_model}")
        print(f"  N layers:   {config.N}")
        print(f"  h heads:    {config.h}")

        records = load_parallel_records(config, config.test_split, config.test_samples)
        dataset = TranslationDataset(records, src_tok, tgt_tok, config.max_seq_len)
        print(f"  test samples: {len(dataset)}")

        max_samples = args.max_bleu_samples if args.max_bleu_samples > 0 else None
        result = evaluate_bleu(
            model, dataset, src_tok, tgt_tok, device,
            max_len=config.max_seq_len,
            beam_size=args.beam,
            length_penalty=config.length_penalty,
            max_samples=max_samples,
            output_path=args.output,
        )

        print()
        decode_mode = f"beam-{args.beam}" if args.beam > 1 else "greedy"
        print(f"Corpus BLEU ({result['num_samples']} samples, {decode_mode}):")
        print(result["bleu_str"])
        print()


if __name__ == "__main__":
    main()

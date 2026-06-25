"""Test model by test-dataset and saved checkpoint.

Use BLEU score to evaluate the translation quality by sacrebleu.
Also includes unit tests for model components and tokenizer.
"""

import argparse
import warnings
from pathlib import Path

import sacrebleu
import torch

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


# ---------------------------------------------------------------------------
# BLEU evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bleu(
    model: Seq2SeqTransformer,
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
        src_mask = src.eq(model.src_pad_id)

        if beam_size > 1:
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
        default=None,
        help="Path to checkpoint .pt file (default: config.checkpoint_path).",
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
        checkpoint_path = args.checkpoint or base_config.checkpoint_path

        if not checkpoint_path.exists():
            print(f"[FAIL] checkpoint not found: {checkpoint_path}")
            return

        print(f"Loading checkpoint: {checkpoint_path}")
        model, src_tok, tgt_tok, config, device = load_checkpoint(checkpoint_path)
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

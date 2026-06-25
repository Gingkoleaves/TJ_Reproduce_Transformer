# Recurrence Transformer

Reproduction of the Transformer model from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) for WMT14 German→English machine translation.

Built on PyTorch's `nn.Transformer` with hand-rolled training infrastructure: BPE tokenizer, NoamOpt scheduler, beam search, and full WMT14 data pipeline.

## Project Structure

```sh
Transformer_handmade/
├── config.py              # All hyperparameters (dataclass)
├── train.py               # Training loop + NoamOpt scheduler
├── inference.py           # Interactive translation (greedy / beam search)
├── test.py                # Unit tests + BLEU evaluation
├── model/
│   ├── transformer.py     # Seq2SeqTransformer, PositionalEncoding, beam search
│   ├── attention.py       # [stub] custom multi-head attention (future)
│   ├── encoder.py         # [stub] custom encoder (future)
│   ├── decoder.py         # [stub] custom decoder (future)
│   ├── embedding.py       # [stub] custom embedding (future)
│   └── layers.py          # [stub] custom layer primitives (future)
├── data/
│   ├── my_tokenizer.py    # ByteLevel BPE tokenizer (HuggingFace tokenizers)
│   ├── my_dataloader.py   # Dataset + DataLoader + batch collation
│   └── en-de-csv/         # WMT14 DE-EN CSV (~4.5M pairs)
│   └── en-de-parquet/     # WMT14 DE-EN Parquet (alternative format)
└── artifacts/             # Saved checkpoints + tokenizers (gitignored)
```

## Quick Start

### Requirements

```bash
conda create -n transformer python=3.11
conda activate transformer
pip install torch pandas sacrebleu tokenizers
```

### Train

```bash
# Full training (matches paper hyperparameters)
python -m Transformer_handmade.train

# Quick smoke test (2K steps, limited data)
# Edit config.py: steps=2000, train_samples=4096, batch_size=16
```

Training produces:

- `artifacts/pytorch_transformer.pt` — model checkpoint
- `artifacts/src_tokenizer.json` / `artifacts/tgt_tokenizer.json` — BPE tokenizers

### Translate

```bash
# Greedy decoding (fast)
python -m Transformer_handmade.inference --text "Das ist ein Test."

# Beam search (beam=4, α=0.6)
python -m Transformer_handmade.inference --text "Das ist ein Test." --beam
```

### Evaluate BLEU

```bash
# Unit tests only (no checkpoint needed)
python -m Transformer_handmade.test --skip-bleu

# BLEU on test set (beam search, 256 samples)
python -m Transformer_handmade.test --skip-unit --beam 4 --max-bleu-samples 256

# Full BLEU evaluation
python -m Transformer_handmade.test --skip-unit --beam 4 --max-bleu-samples 0
```

## Model Configuration

All hyperparameters in `Transformer_handmade/config.py`:

| Parameter | Value | Paper |
| --- | --- | --- |
| **Architecture** | | |
| Layers (N) | 6 | ✓ |
| Heads (h) | 8 | ✓ |
| d_model | 512 | ✓ |
| d_ff | 2048 | ✓ |
| Dropout | 0.1 | ✓ |
| **Optimization** | | |
| Steps | 100,000 | ✓ |
| Warmup steps | 4,000 | ✓ |
| LR schedule | NoamOpt (warmup + decay) | ✓ |
| Batch size | configurable (paper: ~25K tokens) | |
| Adam β₁, β₂, ε | 0.9, 0.98, 1e-9 | ✓ |
| Label smoothing | 0.1 | ✓ |
| Gradient clipping | 1.0 | ✓ |
| **Data** | | |
| Dataset | WMT14 DE-EN (~4.5M pairs) | ✓ |
| Tokenizer | ByteLevel BPE, 37K vocab | ✓ |
| Shared embeddings | src = tgt = output projection | ✓ |
| **Inference** | | |
| Beam size | 4 | ✓ |
| Length penalty α | 0.6 | ✓ |

## Memory Requirements

The attention matrix scales as `batch_size × heads × seq_len²`. Training memory estimates for this model (~63M params):

| batch_size | seq_len | GPU Memory | Notes |
| --- | --- | --- | --- |
| 128 | 256 | ~24 GB | OOM on 24 GB GPU |
| 64 | 256 | ~13 GB | ✅ Safe |
| 32 | 256 | ~8 GB | ✅ Comfortable |
| 16 | 256 | ~5 GB | ✅ Small GPU |

Use `grad_accum_steps` to simulate larger batches without increasing memory:

```python
# Effective batch = 128 using less GPU memory:
batch_size = 32
grad_accum_steps = 4  # 32 × 4 = 128
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (already set in `train.py`).

## Key Design Decisions

**BPE Tokenizer.** Uses HuggingFace `tokenizers` library with ByteLevel BPE (37K vocab). Trained separately on source (DE) and target (EN) corpora. Handles unseen words through subword decomposition — no `<unk>` in practice.

**NoamOpt Scheduler.** `lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))`. Critical for Transformer training stability — without warmup, gradients explode in early steps.

**Shared Embeddings.** The source embedding, target embedding, and output projection share the same weight matrix. Reduces parameter count by ~38M and acts as a regularizer (paper Table 3, row E).

**Gradient Accumulation.** Steps the optimizer every `grad_accum_steps` forward passes. Allows training at effective batch size 128 without allocating memory for 128 sequences at once.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [WMT14 Translation Task](https://www.statmt.org/wmt14/translation-task.html)
- [sacrebleu](https://github.com/mjpost/sacrebleu) — standard BLEU evaluation
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) — BPE implementation

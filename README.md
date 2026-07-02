# Recurrence Transformer

Reproduction of the Transformer model from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) for WMT14 German→English machine translation.

Built on PyTorch's `nn.Transformer` with hand-rolled training infrastructure: BPE tokenizer, NoamOpt scheduler, beam search, and full WMT14 data pipeline.

## Results

Trained on a single RTX 5090 Laptop GPU (~30 hours, 80K/100K steps):(Mine are de-en, but paper is en-de)

| Decode | Samples | BLEU | Paper (base) |
|--------|---------|------|-------------|
| Greedy | 3003 (full test) | 26.93 | — |
| Beam-4 (α=0.6) | 3003 (full test) | **28.19** | **27.3** |

BLEU breakdown (beam-4, full): `63.1/36.4/23.0/14.8  BP=0.948`

Current pkt cost me 11 hours to train, and 1 houe to test in single NVIDIA GeForce RTX 5090.

## Project Structure

```shell
Transformer_handmade/
├── config.py              # All hyperparameters (dataclass)
├── train.py               # Training loop + NoamOpt scheduler
├── inference.py           # Interactive translation (greedy / beam search)
├── test.py                # Unit tests + BLEU evaluation
├── scripts/
│   ├── train.sh           # Launch training with logging
│   ├── eval.sh            # Full test-set BLEU evaluation
│   └── translate.sh       # Translate a single sentence
├── model/
│   ├── transformer.py     # Seq2SeqTransformer, PositionalEncoding, beam search
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── layers.py
├── data/
│   ├── my_tokenizer.py    # ByteLevel BPE tokenizer (HuggingFace tokenizers)
│   ├── my_dataloader.py   # Dataset + DataLoader + batch collation
│   └── en-de-csv/         # WMT14 DE-EN CSV (~4.5M pairs)
└── artifacts/             # Checkpoints + tokenizers (gitignored)
    ├── pytorch_transformer.pt
    ├── tokenizer.json
    └── logs/
```

## Quick Start

### Requirements

```bash
conda create -n transformer python=3.11
conda activate transformer
pip install torch pandas sacrebleu tokenizers
```

### Training

```bash
bash Transformer_handmade/scripts/train.sh
```

Logs are saved to `Transformer_handmade/artifacts/logs/train_YYYYMMDD_HHMMSS.log`.

To monitor live:
```bash
tail -f Transformer_handmade/artifacts/logs/train_*.log
```

Training produces:
- `artifacts/pytorch_transformer.pt` — model checkpoint
- `artifacts/tokenizer.json` — shared BPE tokenizer (37K vocab)

### BLEU Evaluation

```bash
# Full test set, beam-4 (paper setting) — ~30 min on RTX 5090
bash Transformer_handmade/scripts/eval.sh

# Save predictions as TSV (SRC / HYP / REF columns)
bash Transformer_handmade/scripts/eval.sh --output predictions.tsv

# Quick sanity check (256 samples)
bash Transformer_handmade/scripts/eval.sh --max-bleu-samples 256

# Greedy decoding (fast)
bash Transformer_handmade/scripts/eval.sh --beam 1
```

Or run directly via Python for more options:
```bash
python -m Transformer_handmade.test --skip-unit --beam 4 --max-bleu-samples 0 --output predictions.tsv
```

### Inference

```bash
# Greedy decoding (~instant)
bash Transformer_handmade/scripts/translate.sh "Die Katze sitzt auf der Matte."

# Beam search (beam=4, α=0.6)
bash Transformer_handmade/scripts/translate.sh --beam "Die Katze sitzt auf der Matte."
```

Or from Python:
```python
from Transformer_handmade.inference import translate

print(translate("Guten Morgen, wie geht es Ihnen?"))           # greedy
print(translate("Guten Morgen, wie geht es Ihnen?", use_beam=True))  # beam-4
```

Input language is **German (DE)**, output is **English (EN)**.

### Unit Tests

```bash
python -m Transformer_handmade.test --skip-bleu
```

## Model Configuration

All hyperparameters in `Transformer_handmade/config.py`:

| Parameter | Value | Paper |
|-----------|-------|-------|
| **Architecture** | | |
| Layers (N) | 6 | ✓ |
| Heads (h) | 8 | ✓ |
| d_model | 512 | ✓ |
| d_ff | 2048 | ✓ |
| Dropout | 0.1 | ✓ |
| **Optimization** | | |
| Steps | 100,000 | ✓ |
| Warmup steps | 4,000 | ✓ |
| LR schedule | NoamOpt | ✓ |
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

### Single-GPU Adaptations

The paper used 8×P100 GPUs with an effective batch of ~25K tokens. Adaptations for a single GPU:

| Setting | This repo | Paper |
|---------|-----------|-------|
| Effective batch | 512 (64×8 accum) | ~833 |
| Precision | bfloat16 AMP | fp32 |
| Hardware | 1×RTX 5090 (24 GB) | 8×P100 (16 GB each) |
| Steps trained | 80K | 100K |

## Memory Requirements

| batch_size | GPU Memory | Notes |
|------------|-----------|-------|
| 128 | ~24 GB | OOM on 24 GB GPU |
| 64 | ~13 GB | Default (safe) |
| 32 | ~8 GB | Comfortable |
| 16 | ~5 GB | Small GPU |

Use `grad_accum_steps` to maintain effective batch size with less memory:
```python
batch_size = 32
grad_accum_steps = 4   # effective batch = 128
```

## Decode Methods

| Method | Speed | BLEU | Notes |
|--------|-------|------|-------|
| Greedy | ~7 sent/s | 26.93 | Best token at each step |
| Beam-4 | ~1.6 sent/s | 28.19 | Paper setting; keeps 4 candidates |

Beam search applies a **length penalty** to prevent bias toward short sequences:

$$\text{score}(Y) = \frac{\log P(Y|X)}{((5 + |Y|) / 6)^{\alpha}}, \quad \alpha = 0.6$$

## Key Design Decisions

**NoamOpt Scheduler.** `lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))`. Warmup is critical for stability — without it, gradients explode in early steps.

**Shared Embeddings.** Source embedding, target embedding, and output projection share the same weight matrix. Reduces parameters by ~38M and acts as a regularizer (paper Table 3, row E).

**Batched Beam Search.** All active beams are processed in a single batched decoder call per step, giving a `beam_size×` speedup over the naive sequential implementation. No KV caching — decoding is still O(T²) per sentence.

**bfloat16 AMP.** RTX 5090 supports bfloat16 natively; no `GradScaler` needed. Combined with TF32 matmul (`torch.set_float32_matmul_precision("high")`), training throughput improves ~1.5× over fp32.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [WMT14 Translation Task](https://www.statmt.org/wmt14/translation-task.html)
- [sacrebleu](https://github.com/mjpost/sacrebleu) — standard BLEU evaluation
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) — BPE implementation

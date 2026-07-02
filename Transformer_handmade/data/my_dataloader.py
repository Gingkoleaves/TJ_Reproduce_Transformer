"""Data loading for WMT14 DE-EN translation.

Includes length-grouped batching (§5.1 of the paper):
sentences are sorted by source length so each batch contains sequences
of similar length, minimizing wasted padding computation.
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from Transformer_handmade.config import TransformerConfig
from Transformer_handmade.data.my_tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def resolve_path(config: TransformerConfig, split: str) -> list[Path]:
    """Resolve the path to the data file for a given split."""
    if config.data_format == "csv":
        return [config.data_dir / config.csv_dirname / f"wmt14_translate_de-en_{split}.csv"]
    elif config.data_format == "parquet":
        search_dir = config.data_dir / config.parquet_dirname
        pattern = f"{split}-*-of-*.parquet"
        files = list(search_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found for pattern: {pattern} in {search_dir}")
        return files
    else:
        raise ValueError(f"Unsupported data format: {config.data_format}")
   

def load_parallel_records(
    config: TransformerConfig,
    split: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    paths = resolve_path(config, split)
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    src_text = (row.get(config.src_lang) or "").strip()
                    tgt_text = (row.get(config.tgt_lang) or "").strip()
                    if not src_text or not tgt_text:
                        continue
                    records.append({"src": src_text, "tgt": tgt_text})
                    if limit is not None and len(records) >= limit:
                        return records
            continue

        frame = pd.read_parquet(path)
        for row in frame.itertuples(index=False):
            src_text = getattr(row, config.src_lang, "").strip()
            tgt_text = getattr(row, config.tgt_lang, "").strip()
            if not src_text or not tgt_text:
                continue
            records.append({"src": src_text, "tgt": tgt_text})
            if limit is not None and len(records) >= limit:
                return records
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        max_seq_len: int,
        length_cache_path: str | Path | None = None,
    ) -> None:
        self.records = records
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        self.src_lengths = self._load_or_build_lengths(
            records, src_tokenizer, max_seq_len, length_cache_path,
        )

    @staticmethod
    def _load_or_build_lengths(
        records: list[dict[str, str]],
        tokenizer: BPETokenizer,
        max_seq_len: int,
        cache_path: str | Path | None,
    ) -> list[int]:
        total = len(records)

        # Try cache first — keyed by record count + tokenizer vocab size
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_key = {
                "n_records": total,
                "max_seq_len": max_seq_len,
                "vocab_size": tokenizer.vocab_size,
            }
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if cached.get("key") == cache_key:
                    print(f"  lengths cache hit ({cache_path})")
                    return cached["lengths"]
                else:
                    print(f"  lengths cache stale, recomputing …")

        # Compute lengths
        print(f"  pre-computing lengths for {total} records …")
        lengths: list[int] = []
        for i, rec in enumerate(records):
            tok_len = min(len(tokenizer.encode(rec["src"])), max_seq_len)
            lengths.append(tok_len)
            if (i + 1) % 500_000 == 0 or i == total - 1:
                print(f"  … {i+1}/{total}")

        # Persist
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"key": cache_key, "lengths": lengths}, cache_path)

        return lengths

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.records[idx]
        src_ids = self.src_tokenizer.encode(sample["src"])[: self.max_seq_len]
        tgt_ids = self.tgt_tokenizer.encode(sample["tgt"])[: self.max_seq_len]
        return {
            "src_text": sample["src"],
            "tgt_text": sample["tgt"],
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Length-grouped batch sampler (§5.1 of the paper)
# ---------------------------------------------------------------------------

class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Sort indices by length, chunk into batches, then shuffle the batches.

    Within each batch all sequences have similar source length → minimal
    padding.  Batch order is randomised across epochs while within-batch
    length similarity is preserved.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        # Sort indices ascending by length
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        # Chunk into batches of consecutive sorted indices
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        # Optionally shuffle the last (partial) batch into a random full batch
        # to avoid always training on the same tail.
        if len(batches) >= 2 and len(batches[-1]) < self.batch_size:
            last = batches.pop()
            # Insert each element of the partial batch into random positions
            # among the preceding batches (simple heuristic)
            if batches:
                for idx in last:
                    random.choice(batches).append(idx)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

@dataclass
class TranslationBatch:
    src: torch.Tensor
    tgt: torch.Tensor
    src_padding_mask: torch.Tensor
    tgt_padding_mask: torch.Tensor
    src_texts: list[str]
    tgt_texts: list[str]

# 整理sample序列为同长度，用pad补齐
def build_collate_fn(pad_id: int):
    def collate_fn(samples: list[dict[str, Any]]) -> TranslationBatch:
        src_ids = [sample["src_ids"] for sample in samples]
        tgt_ids = [sample["tgt_ids"] for sample in samples]
        src = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
        tgt = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
        src_padding_mask = src.eq(pad_id)
        tgt_padding_mask = tgt.eq(pad_id)
        return TranslationBatch(
            src=src,
            tgt=tgt,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            src_texts=[sample["src_text"] for sample in samples],
            tgt_texts=[sample["tgt_text"] for sample in samples],
        )

    return collate_fn


# ---------------------------------------------------------------------------
# Tokenizer builder (shared vocab — paper §5.1)
# ---------------------------------------------------------------------------

def _build_tokenizers(
    config: TransformerConfig,
    train_records: list[dict[str, str]],
) -> tuple[BPETokenizer, BPETokenizer]:
    """Train a single shared BPE tokenizer on combined source + target text."""
    combined_texts: list[str] = []
    for record in train_records:
        combined_texts.append(record["src"])
        combined_texts.append(record["tgt"])

    shared = BPETokenizer(lowercase=config.lowercase)
    shared.fit(combined_texts, vocab_size=config.vocab_size, min_freq=config.min_token_freq)
    return shared, shared


# ---------------------------------------------------------------------------
# Top-level dataloader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: TransformerConfig,
) -> tuple[dict[str, DataLoader], BPETokenizer, BPETokenizer]:
    train_records = load_parallel_records(config, config.train_split, config.train_samples)
    valid_records = load_parallel_records(config, config.valid_split, config.valid_samples)
    test_records = load_parallel_records(config, config.test_split, config.test_samples)

    src_tokenizer, tgt_tokenizer = _build_tokenizers(config, train_records)
    collate_fn = build_collate_fn(src_tokenizer.pad_id)

    datasets = {
        "train": TranslationDataset(
            train_records, src_tokenizer, tgt_tokenizer, config.max_seq_len,
            length_cache_path=config.artifact_dir / "train_lengths.pt",
        ),
        "validation": TranslationDataset(
            valid_records, src_tokenizer, tgt_tokenizer, config.max_seq_len,
            length_cache_path=config.artifact_dir / "valid_lengths.pt",
        ),
        "test": TranslationDataset(
            test_records, src_tokenizer, tgt_tokenizer, config.max_seq_len,
            length_cache_path=config.artifact_dir / "test_lengths.pt",
        ),
    }

    # Training: length-grouped batching (paper §5.1)
    train_sampler = LengthGroupedBatchSampler(
        datasets["train"].src_lengths,
        batch_size=config.batch_size,
        shuffle=True,
    )

    pin = config.num_workers > 0
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            persistent_workers=pin,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            persistent_workers=pin,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin,
            persistent_workers=pin,
        ),
    }
    return loaders, src_tokenizer, tgt_tokenizer

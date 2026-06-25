"""Character-level BPE tokenizer built on HuggingFace `tokenizers` library.

Matches the "Attention Is All You Need" recipe (§5.1):
- Splits on whitespace, then applies BPE on characters within each word.
- Shared vocabulary for source and target languages.
- 37K tokens, trained on combined DE+EN corpora.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC, Lowercase, Sequence


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"


class BPETokenizer:
    """Character-level BPE tokenizer (Metaspace → SentencePiece-compatible).

    Preserves the same public API as the old SimpleTokenizer so training,
    inference, and test code work without changes.
    """

    def __init__(self, lowercase: bool = False, special_tokens: SpecialTokens | None = None):
        self.lowercase = lowercase
        self.special_tokens = special_tokens or SpecialTokens()
        self._tokenizer: Tokenizer | None = None

    # ---- properties ----

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(self.special_tokens.pad)

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id(self.special_tokens.unk)

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(self.special_tokens.bos)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(self.special_tokens.eos)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def _tok(self) -> Tokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized — call fit() or load() first")
        return self._tokenizer

    # ---- build ----

    def fit(self, texts: list[str], vocab_size: int = 37000, min_freq: int = 2) -> None:
        """Train a character-level BPE tokenizer on `texts`.

        ``Metaspace`` replaces spaces with '▁' (U+2581) so the BPE model sees
        spaces as regular characters.  On decode, '▁' is converted back to
        spaces — the same approach used by SentencePiece and many NMT
        implementations.
        """
        specials = [
            self.special_tokens.pad,
            self.special_tokens.unk,
            self.special_tokens.bos,
            self.special_tokens.eos,
        ]
        tok = Tokenizer(models.BPE(unk_token=self.special_tokens.unk))
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
        tok.decoder = decoders.Metaspace(replacement="▁")

        # Normalizer: NFKC + optional lowercase
        normalizers = [NFKC()]
        if self.lowercase:
            normalizers.append(Lowercase())
        tok.normalizer = Sequence(normalizers)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=specials,
            min_frequency=min_freq,
            show_progress=False,
        )

        def _iter():
            for t in texts:
                yield t

        tok.train_from_iterator(_iter(), trainer)
        self._tokenizer = tok

    # ---- encode / decode ----

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        encoding = self._tok.encode(text)
        ids = encoding.ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        # Save BPE model as JSON
        tok_path = path.with_suffix(".bpe.json")
        self._tok.save(str(tok_path), pretty=True)
        # Save lightweight metadata alongside
        meta = {
            "lowercase": self.lowercase,
            "special_tokens": self.special_tokens.__dict__,
            "bpe_path": str(tok_path.name),
        }
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        inst = cls(
            lowercase=meta["lowercase"],
            special_tokens=SpecialTokens(**meta["special_tokens"]),
        )
        # Resolve BPE model path (relative to the meta file)
        bpe_path = Path(meta.get("bpe_path", path.with_suffix(".bpe.json").name))
        if not bpe_path.is_absolute():
            bpe_path = path.parent / bpe_path
        inst._tokenizer = Tokenizer.from_file(str(bpe_path))
        return inst

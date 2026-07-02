from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransformerConfig:
    # ====== Transformer architecture (matches "Attention Is All You Need" base) ======
    N: int = 6
    h: int = 8
    d_model: int = 512
    d_key: int = 64
    d_value: int = 64
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 256

    # ====== Optimization (matches paper §5.3) ======
    steps: int = 100_000        # paper: 100K steps on 8×P100
    batch_size: int = 64        # per-GPU batch; effective = batch_size * grad_accum_steps
    warmup_steps: int = 4_000   # paper: 4000 warmup steps
    label_smoothing: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    clip_grad_norm: float = 1.0
    grad_accum_steps: int = 8   # effective batch = 512 (paper ~833, relaxed for single GPU)

    # ====== AMP (bfloat16 — no scaler needed, stable on RTX 5090) ======
    use_amp: bool = True
    
    # ====== Data ======
    data_format: str = "csv"
    src_lang: str = "en"
    tgt_lang: str = "de"
    data_dir: Path = Path("Transformer_handmade/data")
    csv_dirname: str = "en-de-csv"
    parquet_dirname: str = "en-de-parquet"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"
    # None = use all available data
    train_samples: int | None = None
    valid_samples: int | None = 3_000
    test_samples: int | None = None
    # BPE tokenizer settings (matches paper: 37K shared vocab)
    vocab_size: int = 37_000
    min_token_freq: int = 2
    lowercase: bool = False
    num_workers: int = 4

    # ====== Inference ======
    beam_size: int = 4
    length_penalty: float = 0.6

    # ====== Runtime ======
    seed: int = 42
    device: str = "cuda"
    log_every: int = 100
    eval_every: int = 2_000
    save_every: int = 10_000

    # ====== Output ======
    artifact_dir: Path = Path("Transformer_handmade/artifacts")
    checkpoint_name: str = "pytorch_transformer.pt"
    tokenizer_name: str = "tokenizer.json"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifact_dir = Path(self.artifact_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact_dir / self.checkpoint_name

    @property
    def tokenizer_path(self) -> Path:
        return self.artifact_dir / self.tokenizer_name

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


def get_config() -> TransformerConfig:
    return TransformerConfig()

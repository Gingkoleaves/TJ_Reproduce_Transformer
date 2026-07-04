"""Average the last N training snapshots into a single checkpoint.

Paper §5.3: "For the base models, we used a single model obtained by
averaging the last 5 checkpoints." Snapshots are written during training by
`train.py --checkpoint-avg-every` into `artifacts/avg_ckpts/step_*.pt`.

Usage:
    python -m Transformer_handmade.average_checkpoints
    python -m Transformer_handmade.average_checkpoints --n 5 --output artifacts/pytorch_transformer_avg.pt
"""

import argparse
from pathlib import Path

import torch

from Transformer_handmade.config import get_config


def average_checkpoints(snapshot_dir: Path, n: int, output: Path) -> Path:
    snapshots = sorted(snapshot_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")

    snapshots = snapshots[-n:]
    print(f"Averaging {len(snapshots)} snapshots:")
    for path in snapshots:
        print(f"  {path}")

    avg_state = None
    ref_ckpt = None
    for path in snapshots:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ref_ckpt = ref_ckpt or ckpt
        sd = ckpt["model_state_dict"]
        if avg_state is None:
            avg_state = {key: value.clone().float() for key, value in sd.items()}
        else:
            for key, value in sd.items():
                avg_state[key] += value.float()

    assert avg_state is not None and ref_ckpt is not None
    for key in avg_state:
        avg_state[key] /= len(snapshots)
        avg_state[key] = avg_state[key].to(ref_ckpt["model_state_dict"][key].dtype)

    torch.save(
        {
            "model_state_dict": avg_state,
            "step": ref_ckpt["step"],
            "config": ref_ckpt["config"],
            "src_vocab_size": ref_ckpt["src_vocab_size"],
            "tgt_vocab_size": ref_ckpt["tgt_vocab_size"],
            "averaged_from_steps": [int(path.stem.split("_")[1]) for path in snapshots],
        },
        output,
    )
    print(f"Averaged checkpoint saved to {output}")
    return output


def main() -> None:
    config = get_config()
    parser = argparse.ArgumentParser(description="Average the last N checkpoint snapshots.")
    parser.add_argument("--snapshot-dir", type=Path, default=config.artifact_dir / "avg_ckpts")
    parser.add_argument("--n", type=int, default=5, help="Number of most-recent snapshots to average.")
    parser.add_argument("--output", type=Path, default=config.artifact_dir / "pytorch_transformer_avg.pt")
    args = parser.parse_args()
    average_checkpoints(args.snapshot_dir, args.n, args.output)


if __name__ == "__main__":
    main()

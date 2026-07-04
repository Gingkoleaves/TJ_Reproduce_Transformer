"""Average the last N training snapshots into a single checkpoint.

Paper §5.3: "For the base models, we used a single model obtained by
averaging the last 5 checkpoints." Snapshots are written during training by
`train.py --checkpoint-avg-every` into `artifacts/avg_ckpts/step_*.pt`.

Usage:
    python -m Transformer_handmade.average_checkpoints
    python -m Transformer_handmade.average_checkpoints --n 5 --output artifacts/pytorch_transformer_avg.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from Transformer_handmade.config import get_config


def main() -> None:
    config = get_config()
    parser = argparse.ArgumentParser(description="Average the last N checkpoint snapshots.")
    parser.add_argument("--snapshot-dir", type=Path, default=config.artifact_dir / "avg_ckpts")
    parser.add_argument("--n", type=int, default=5, help="Number of most-recent snapshots to average.")
    parser.add_argument("--output", type=Path, default=config.artifact_dir / "pytorch_transformer_avg.pt")
    args = parser.parse_args()

    snapshots = sorted(args.snapshot_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {args.snapshot_dir}")
    snapshots = snapshots[-args.n:]
    print(f"Averaging {len(snapshots)} snapshots:")
    for p in snapshots:
        print(f"  {p}")

    avg_state = None
    ref_ckpt = None
    for path in snapshots:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ref_ckpt = ref_ckpt or ckpt
        sd = ckpt["model_state_dict"]
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg_state[k] += v.float()

    for k in avg_state:
        avg_state[k] /= len(snapshots)
        avg_state[k] = avg_state[k].to(ref_ckpt["model_state_dict"][k].dtype)

    torch.save(
        {
            "model_state_dict": avg_state,
            "step": ref_ckpt["step"],
            "config": ref_ckpt["config"],
            "src_vocab_size": ref_ckpt["src_vocab_size"],
            "tgt_vocab_size": ref_ckpt["tgt_vocab_size"],
            "averaged_from_steps": [int(p.stem.split("_")[1]) for p in snapshots],
        },
        args.output,
    )
    print(f"Averaged checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()

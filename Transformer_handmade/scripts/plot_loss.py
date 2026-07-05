"""Parse training logs and plot the README loss curves.

Regenerates the four per-run charts plus the combined val-loss chart in
Notes/assets/ from the logs listed in RUNS. To add a run, append a
(key, log file, title, subtitle) entry and a COMBINED_LABELS entry.

Usage:
    conda run -n transformer python Transformer_handmade/scripts/plot_loss.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "Transformer_handmade/artifacts/logs"
OUT = REPO / "Notes/assets"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]  # blue, aqua, yellow, green

TRAIN_RE = re.compile(r"step=(\d+) lr=\S+ train_loss=([\d.]+)")
VAL_RE = re.compile(r"step=(\d+) val_loss=([\d.]+)")

RUNS = [
    # (key, log file, title, subtitle)
    ("main_run1", "train_20260624_233450.log",
     "main Run 1 — DE→EN, nn.Transformer",
     "batch 64 × grad-accum 8 (512 sentences/step) · stopped at 80K/100K steps · RTX 5090"),
    ("main_run2", "train_20260704_111436.log",
     "main Run 2 — EN→DE, nn.Transformer",
     "true 25K-token batches, no grad accumulation · 100K steps · RTX 4090 48GB"),
    ("dev1_run1", "train_20260703_012236.log",
     "dev1 Run 1 — EN→DE, hand-rolled model",
     "batch 64 × grad-accum 8 (512 sentences/step) · stopped at 92K/100K steps · RTX 5090"),
    ("dev1_run2", "train_20260703_182328.log",
     "dev1 Run 2 — EN→DE, hand-rolled model, token-budget batching",
     "12K tokens × grad-accum 2 (≈24K tokens/step) · 100K steps · RTX 5090"),
]

COMBINED_LABELS = {
    "main_run1": "main Run 1 (DE→EN, 512 sent)",
    "main_run2": "main Run 2 (EN→DE, 25K tok)",
    "dev1_run1": "dev1 Run 1 (EN→DE, 512 sent)",
    "dev1_run2": "dev1 Run 2 (EN→DE, 24K tok)",
}


def parse(path: Path):
    train, val = [], []
    for line in path.read_text().splitlines():
        m = TRAIN_RE.search(line)
        if m:
            step, loss = int(m.group(1)), float(m.group(2))
            if loss < 20:  # drop the known step=100 accumulated-sum logging bug
                train.append((step, loss))
            continue
        m = VAL_RE.search(line)
        if m:
            val.append((int(m.group(1)), float(m.group(2))))
    return train, val


def kfmt(x, _):
    return f"{x/1000:.0f}K" if x else "0"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlabel("optimizer step", color=MUTED, fontsize=9.5)
    ax.set_ylabel("cross-entropy loss (label-smoothed)", color=MUTED, fontsize=9.5)


def plot_single(key, title, subtitle, train, val):
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=160)
    fig.set_facecolor(SURFACE)
    style_axes(ax)

    ts, tl = zip(*train)
    ax.plot(ts, tl, color=SLOTS[0], linewidth=1.4, label="train loss")
    if val:
        vs, vl = zip(*val)
        ax.plot(vs, vl, color=SLOTS[1], linewidth=1.8, marker="o",
                markersize=3, markevery=max(1, len(vs) // 25), label="val loss")
        # direct labels at line ends (relief rule for aqua)
        ax.annotate(f"val {vl[-1]:.3f}", (vs[-1], vl[-1]),
                    xytext=(6, 4), textcoords="offset points",
                    color=INK2, fontsize=9, fontweight="bold")
    ax.annotate(f"train {tl[-1]:.2f}", (ts[-1], tl[-1]),
                xytext=(6, -10), textcoords="offset points",
                color=INK2, fontsize=9)
    if key == "dev1_run1":
        ax.text(37000, 6.35, "val loss diverged 16K–58K\nwhile train loss kept falling,\nthen recovered from 60K on",
                color=INK2, fontsize=8.5, ha="center", va="top")

    ax.set_xlim(0, max(ts[-1], val[-1][0] if val else 0) * 1.09)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5,
                    labelcolor=INK2)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                 loc="left", pad=22)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)

    fig.tight_layout()
    out = OUT / f"loss_{key}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  train pts={len(train)} val pts={len(val)} "
          f"last train step={ts[-1]} last val={val[-1] if val else None}")


def plot_combined(all_val):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    fig.set_facecolor(SURFACE)
    style_axes(ax)

    # dodge the two labels that end at ~3.0 on the same step
    label_dy = {"main_run2": -9, "dev1_run2": 7}
    for i, (key, val) in enumerate(all_val.items()):
        vs, vl = zip(*val)
        ax.plot(vs, vl, color=SLOTS[i], linewidth=1.8,
                label=COMBINED_LABELS[key])
        ax.annotate(f"{vl[-1]:.3f}", (vs[-1], vl[-1]),
                    xytext=(5, label_dy.get(key, 0)),
                    textcoords="offset points",
                    color=SLOTS[i], fontsize=8.5, fontweight="bold",
                    va="center")

    ax.set_xlim(0, 109000)
    ax.set_ylabel("validation loss", color=MUTED, fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK2)
    ax.set_title("Validation loss — all four runs", color=INK, fontsize=12,
                 fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.03,
            "main Run 1 is DE→EN (easier direction, not directly comparable); "
            "the three EN→DE runs converge to val loss ≈ 3.0–3.26",
            transform=ax.transAxes, color=INK2, fontsize=9)

    fig.tight_layout()
    out = OUT / "loss_all_runs.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    all_val = {}
    for key, logname, title, subtitle in RUNS:
        train, val = parse(LOGS / logname)
        plot_single(key, title, subtitle, train, val)
        all_val[key] = val
    plot_combined(all_val)


if __name__ == "__main__":
    main()

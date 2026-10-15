"""Aggregate and visualize loss trajectories across all trained neural operator models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate training and validation loss curves from all available model runs."
    )
    parser.add_argument("--root", default="results", help="Root results directory to traverse.")
    parser.add_argument(
        "--output",
        default="results/all_model_losses.png",
        help="Path for aggregate loss figure.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    plt.figure(figsize=(11, 6))
    plotted: Set[Path] = set()

    for history in sorted(root.glob("**/training_history.npz")):
        data = np.load(history)
        for key in data.files:
            if "loss" in key.lower() or "train" in key.lower() or "val" in key.lower():
                label = f"{history.parent.name}: {key}"
                plt.semilogy(np.maximum(data[key], 1e-12), label=label)
                plotted.add(history.parent)

    for metrics in sorted(root.glob("**/metrics.json")):
        if metrics.parent in plotted:
            continue
        try:
            values = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        loss = values.get("test_loss", values.get("best_validation_data_loss", values.get("best_validation_loss")))
        if isinstance(loss, (int, float)) and np.isfinite(loss):
            plt.axhline(
                max(float(loss), 1e-12),
                linestyle="--",
                label=f"{metrics.parent.name}: reported loss ({loss:.4e})",
            )

    has_metrics = list(root.glob("**/metrics.json"))
    if not plotted and not has_metrics:
        print(f"Notice: No existing loss histories found below {root}. Skipping plot generation.")
        return

    plt.xlabel("Epoch (Continuous Curves) / Checkpoint Benchmark (Dashed Lines)")
    plt.ylabel("Loss Metric (Log Scale)")
    plt.title("Comparative Training and Validation Loss Trajectories")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Aggregate loss figure saved to: {out}")


if __name__ == "__main__":
    main()

"""Visualize loss convergence curves from training histories for a single model family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for loss curve plotting."""
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss trajectories for a specific model family."
    )
    parser.add_argument(
        "--root",
        default="runs",
        help="Root directory or specific run directory containing training_history.npz.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["train_loss", "val_loss"],
        help="Explicit metric keys to plot (e.g., train_loss, val_loss, train_data_loss).",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Optional model family filter (e.g., 'fno' or 'pino').",
    )
    parser.add_argument(
        "--output",
        default="runs/loss_trajectories.png",
        help="Output path for the generated loss figure.",
    )
    return parser.parse_args(args)


def plot_loss_histories(
    root_dir: Path,
    keys: List[str],
    output_path: Path,
    family_filter: Optional[str] = None,
) -> bool:
    """Plot training history curves matching explicit metric keys."""
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        print(f"Notice: Directory not found: {root_dir}")
        return False

    history_files = sorted(root_dir.glob("**/training_history.npz"))
    if root_dir.name == "training_history.npz":
        history_files = [root_dir]

    if family_filter:
        history_files = [h for h in history_files if family_filter.lower() in str(h).lower()]

    if not history_files:
        print(f"Notice: No training history files found under {root_dir}.")
        return False

    plt.figure(figsize=(9, 5))
    has_curves = False

    for h_file in history_files:
        run_name = h_file.parent.name
        try:
            data = np.load(h_file)
        except Exception:
            continue

        for k in keys:
            if k in data.files:
                series = data[k]
                if len(series) == 0:
                    continue
                epochs = np.arange(1, len(series) + 1)
                label = f"{run_name}: {k.replace('_', ' ')}"
                plt.semilogy(epochs, np.maximum(series, 1.0e-12), label=label, linewidth=1.5)
                has_curves = True

    if not has_curves:
        print(f"Notice: None of the requested keys {keys} found in training histories.")
        plt.close()
        return False

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE (Log Scale)")
    plt.title(f"Loss Convergence Trajectories ({', '.join(keys)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Loss trajectory plot written to: {output_path}")
    return True


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for plotting loss trajectories."""
    parsed = parse_args(args)
    root = Path(parsed.root)
    out = Path(parsed.output)
    plot_loss_histories(root, parsed.keys, out, family_filter=parsed.family)


if __name__ == "__main__":
    main()

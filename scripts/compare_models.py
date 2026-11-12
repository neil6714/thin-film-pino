"""Comparative benchmark evaluating trained neural operator models on held-out trajectories."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from thin_film_pino.evaluation.evaluate import (
    evaluate_baselines,
    evaluate_single_model,
    format_evaluation_summary,
    load_model_from_checkpoint,
)
from thin_film_pino.training.train import load_dataset_in_memory


def evaluate_checkpoint_model(
    data: Mapping[str, Any],
    checkpoint_path: Union[str, Path],
    split_indices: np.ndarray,
    batch_size: int = 4,
    n_bootstraps: int = 500,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint against held-out trajectory data.

    Parameters
    ----------
    data : Mapping[str, Any]
        In-memory dataset dictionary.
    checkpoint_path : Union[str, Path]
        Path to PyTorch checkpoint.
    split_indices : np.ndarray
        Indices of test trajectories.
    batch_size : int, default=4
        Batch size for model inference.
    n_bootstraps : int, default=500
        Number of bootstrap iterations.
    device : Optional[torch.device], default=None
        Device for evaluation.

    Returns
    -------
    Dict[str, Any]
        Evaluation metrics dictionary.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(ckpt_path, dict(data), device)
    metrics = evaluate_single_model(
        model=model,
        data=dict(data),
        split_indices=split_indices,
        batch_size=batch_size,
        n_bootstraps=n_bootstraps,
        device=device,
    )
    return metrics


def plot_comparison_panels(
    comparison_data: Mapping[str, Mapping[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Generate 3-panel comparison figure with independent scales per channel.

    Parameters
    ----------
    comparison_data : Mapping[str, Mapping[str, Any]]
        Dictionary mapping model names to channel metrics.
    output_path : Union[str, Path]
        Output file path for saving PNG.
    """
    models = list(comparison_data.keys())
    if not models:
        return

    channels = [
        ("concentration", "Concentration MAE (mol/m^3)"),
        ("surface_coverage", "Surface Coverage MAE (-)"),
        ("film_thickness", "Film Thickness MAE (m)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(models))

    for ax, (ch_key, ch_label) in zip(axes, channels):
        maes = []
        errors_lower = []
        errors_upper = []

        for m in models:
            m_metrics = comparison_data[m].get(ch_key, {})
            val = m_metrics.get("MAE", 0.0)
            maes.append(val)

            ci_info = m_metrics.get("bootstrap_mae_ci", {})
            if ci_info and "mae_ci_lower" in ci_info and "mae_ci_upper" in ci_info:
                err_low = max(0.0, val - ci_info["mae_ci_lower"])
                err_high = max(0.0, ci_info["mae_ci_upper"] - val)
                errors_lower.append(err_low)
                errors_upper.append(err_high)
            else:
                errors_lower.append(0.0)
                errors_upper.append(0.0)

        yerr = [errors_lower, errors_upper]
        bars = ax.bar(
            x,
            maes,
            yerr=yerr,
            capsize=5,
            color=["#888888", "#aaaaaa", "#1f77b4", "#ff7f0e"][: len(models)],
            edgecolor="black",
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=25, ha="right")
        ax.set_ylabel(ch_label)
        ax.set_title(ch_label.split(" ")[0] + " Channel")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2e}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Model Benchmark vs Baselines on Held-Out Test Trajectories", y=1.02)
    plt.tight_layout()

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def main(args: Optional[Sequence[str]] = None) -> None:
    """Run model comparison suite."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare neural operator models against held-out ALD test trajectories."
    )
    parser.add_argument(
        "--input", required=True, help="Path to prepared transient ALD dataset (.npz)."
    )
    parser.add_argument(
        "--output", default="runs/comparison", help="Output directory for comparison metrics."
    )
    parser.add_argument(
        "--operator",
        default="runs/fno/best_model.pt",
        help="Path to baseline temporal operator checkpoint.",
    )
    parser.add_argument(
        "--pino",
        default="runs/pino/best_model.pt",
        help="Path to physics-informed temporal PINO checkpoint.",
    )
    parser.add_argument(
        "--operator-seeds",
        nargs="*",
        default=None,
        help="Optional glob pattern or list of multiple FNO seed checkpoints.",
    )
    parser.add_argument(
        "--pino-seeds",
        nargs="*",
        default=None,
        help="Optional glob pattern or list of multiple PINO seed checkpoints.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "ood", "validation", "train"],
        help="Dataset split partition to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=500,
        help="Number of bootstrap resamples across trajectories.",
    )
    parsed = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_path = Path(parsed.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    data = load_dataset_in_memory(input_path)
    split_key = f"{parsed.split}_indices"
    if split_key not in data:
        raise KeyError(f"Requested split '{parsed.split}' not present in dataset.")
    split_idx = data[split_key]
    test_size = int(len(split_idx))

    out_dir = Path(parsed.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute Baselines (Train-Mean and Persistence)
    baselines = evaluate_baselines(data, split_idx, n_bootstraps=parsed.n_bootstraps)
    comparison_results: Dict[str, Dict[str, Any]] = {
        "train_mean": baselines["train_mean"],
        "persistence": baselines["persistence"],
    }

    # 2. Evaluate Checkpoint Models
    model_targets: List[Tuple[str, str, Optional[List[str]]]] = [
        ("temporal_operator", parsed.operator, parsed.operator_seeds),
        ("temporal_pino", parsed.pino, parsed.pino_seeds),
    ]

    for name, single_ckpt, multi_seeds in model_targets:
        # Determine candidate checkpoint list
        ckpt_paths: List[Path] = []
        if multi_seeds:
            for pattern in multi_seeds:
                matched = glob.glob(pattern)
                if matched:
                    ckpt_paths.extend([Path(p).resolve() for p in matched])
                else:
                    ckpt_paths.append(Path(pattern).resolve())
        else:
            ckpt_paths = [Path(single_ckpt).resolve()]

        # Check for checkpoint existence - fail fast if missing
        for cp in ckpt_paths:
            if not cp.exists():
                raise FileNotFoundError(
                    f"Required checkpoint for '{name}' does not exist: {cp}. "
                    f"Check path or provide a valid trained model."
                )

        print(f"Evaluating {name} ({len(ckpt_paths)} checkpoint(s))...")
        seed_evals = []
        for cp in ckpt_paths:
            res = evaluate_checkpoint_model(
                data=data,
                checkpoint_path=cp,
                split_indices=split_idx,
                batch_size=parsed.batch_size,
                n_bootstraps=parsed.n_bootstraps,
                device=device,
            )
            seed_evals.append(res)

        # Aggregate across seeds
        if len(seed_evals) == 1:
            comparison_results[name] = seed_evals[0]
        else:
            # Multi-seed mean and standard deviation
            agg: Dict[str, Any] = {"multi_seed_count": len(seed_evals), "seeds": []}
            for ch in ("concentration", "surface_coverage", "film_thickness"):
                maes = [se[ch]["MAE"] for se in seed_evals]
                rmses = [se[ch]["RMSE"] for se in seed_evals]
                agg[ch] = {
                    "MAE": float(np.mean(maes)),
                    "MAE_std_across_seeds": float(np.std(maes)),
                    "RMSE": float(np.mean(rmses)),
                    "RMSE_std_across_seeds": float(np.std(rmses)),
                }
            comparison_results[name] = agg

    # Print human-readable summary table with test size
    print(
        format_evaluation_summary(
            model_name="Neural Operator Benchmark",
            metrics=comparison_results.get("temporal_operator", {}),
            baselines=baselines,
            test_size=test_size,
        )
    )

    # Save complete JSON comparison
    comparison_json = out_dir / "comparison_metrics.json"
    comparison_json.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    # Generate 3-panel figure with per-channel scales
    plot_comparison_panels(comparison_results, out_dir / "physical_mae_comparison.png")

    print(f"Comparison metrics and per-channel panels written to: {out_dir}")


if __name__ == "__main__":
    main()

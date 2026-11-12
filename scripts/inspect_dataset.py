"""Diagnostic inspection, statistics reporting, and visualization of ALD datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.data.validate import validate_dataset


def print_summary(data: Mapping[str, Any]) -> None:
    """Print tensor schema, shapes, and dtypes for all archived arrays."""
    print("\nDataset Summary:")
    for key in sorted(data.keys()):
        arr = data[key]
        if hasattr(arr, "shape"):
            print(f"{key:24s} shape={str(arr.shape):20s} dtype={arr.dtype}")

    print("\nPhysical Parameter Distributions:")
    param_key = "raw_parameters" if "raw_parameters" in data else "parameters"
    if param_key in data:
        params = data[param_key]
        names = (
            data["parameter_names"].tolist()
            if "parameter_names" in data
            else [f"p_{i}" for i in range(params.shape[1])]
        )
        for i, name in enumerate(names[: params.shape[1]]):
            vals = params[:, i] if params.ndim > 1 else np.array([params[i]])
            print(
                f"{name:16s} min={vals.min():.6e} max={vals.max():.6e} "
                f"mean={vals.mean():.6e} std={vals.std():.6e}"
            )

    print("\nField Statistics:")
    for key in (
        "concentration",
        "surface_coverage",
        "film_thickness",
        "metric_gpc",
        "metric_step_coverage",
    ):
        if key in data:
            vals = data[key]
            print(
                f"{key:24s} min={vals.min():.6e} max={vals.max():.6e} "
                f"mean={vals.mean():.6e} std={vals.std():.6e}"
            )


def plot_conformality(data: Mapping[str, Any], output_dir: Path) -> None:
    """Render conformality distribution histogram if step coverage metrics are present."""
    sc_key = "metric_step_coverage" if "metric_step_coverage" in data else "conformality"
    if sc_key not in data:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    conformality = data[sc_key]
    if conformality.ndim > 1:
        # Final cycle conformality
        vals = conformality[:, -1]
    else:
        vals = conformality

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=25, edgecolor="black", alpha=0.7, color="teal")
    plt.xlabel("Step Coverage (Floor / Field Thickness)")
    plt.ylabel("Sample Frequency")
    plt.title("Step Coverage Distribution Across Dataset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "step_coverage_distribution.png", dpi=200)
    plt.close()


def plot_sample(data: Mapping[str, Any], sample_idx: int, output_dir: Path) -> None:
    """Render 2D multi-field spatial contours for a designated sample index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    solid_mask = data["solid_mask"].astype(bool)

    c = data["concentration"][sample_idx]
    theta = data["surface_coverage"][sample_idx]
    h = data["film_thickness"][sample_idx]

    # If 3D (T, ny, nx), take last snapshot
    if c.ndim == 3:
        c = c[-1]
        theta = theta[-1]
        h = h[-1]

    c_plot = c.copy().astype(np.float64)
    c_plot[solid_mask] = np.nan
    theta_plot = theta.copy().astype(np.float64)
    theta_plot[solid_mask] = np.nan
    h_plot = h.copy().astype(np.float64)
    h_plot[solid_mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(c_plot, cmap="viridis", origin="lower")
    axes[0].set_title(f"Sample {sample_idx}: Concentration")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(theta_plot, cmap="plasma", origin="lower", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Sample {sample_idx}: Coverage theta")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(h_plot, cmap="inferno", origin="lower")
    axes[2].set_title(f"Sample {sample_idx}: Film Thickness")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / f"sample_{sample_idx:03d}_fields.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ALD dataset schema, distributions, and spatial fields."
    )
    parser.add_argument("--input", required=True, help="Path to input .npz dataset.")
    parser.add_argument(
        "--output-dir",
        default="results/figures/dataset_inspection",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Requested dataset file not found: {input_path}")

    out_dir = Path(args.output_dir).resolve()
    data = np.load(input_path)

    print(f"Inspecting dataset: {input_path}")
    print_summary(data)

    # Delegate validation directly to validator rather than duplicating checks
    passed, report = validate_dataset(data, input_path=input_path)
    print("\nDataset Verification Summary:")
    print(f"Overall Status: {'PASS' if passed else 'FAIL'}")
    for name, res in report["checks"].items():
        print(f"[{'PASS' if res else 'FAIL':4s}] {name}")

    plot_conformality(data, out_dir)

    param_key = "raw_parameters" if "raw_parameters" in data else "parameters"
    if param_key in data:
        n_samples = len(data[param_key])
        sample_indices = (
            [0, n_samples // 2, n_samples - 1] if n_samples >= 3 else list(range(n_samples))
        )
        for idx in sample_indices:
            plot_sample(data, idx, out_dir)

    print("\nDataset inspection complete.")
    print(f"Diagnostic figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

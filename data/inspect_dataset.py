"""Diagnostic inspection, statistics reporting, and visualization of ALD datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAMETER_NAMES_8 = [
    "diffusivity",
    "k_ads",
    "k_des",
    "k_rxn",
    "k_growth",
    "pulse_time",
    "purge_time",
    "reaction_time",
]

PARAMETER_NAMES_5 = [
    "diffusivity",
    "k_ads",
    "k_des",
    "k_rxn",
    "k_growth",
]


def print_summary(data: Mapping[str, Any]) -> None:
    """Print tensor schema, shapes, and dtypes for all archived arrays."""
    print("Dataset Summary")
    print("-" * 60)
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"{key:24s} shape={str(arr.shape):20s} dtype={arr.dtype}")

    print("\nPhysical Parameter Distributions")
    print("-" * 60)
    if "parameters" in data:
        params = data["parameters"]
        num_params = params.shape[1] if params.ndim > 1 else len(params)
        names = PARAMETER_NAMES_8 if num_params >= 8 else PARAMETER_NAMES_5[:num_params]

        for i, name in enumerate(names):
            vals = params[:, i] if params.ndim > 1 else np.array([params[i]])
            print(
                f"{name:16s} min={vals.min():.6e} max={vals.max():.6e} "
                f"mean={vals.mean():.6e} std={vals.std():.6e}"
            )

    print("\nField Statistics")
    print("-" * 60)
    for key in ("concentration", "surface_coverage", "film_thickness", "gpc", "conformality"):
        if key in data:
            vals = data[key]
            print(
                f"{key:24s} min={vals.min():.6e} max={vals.max():.6e} "
                f"mean={vals.mean():.6e} std={vals.std():.6e}"
            )


def validate_physics(data: Mapping[str, Any]) -> bool:
    """Verify physical invariants across concentration, coverage, and thickness arrays."""
    checks = {}

    for key in data.keys():
        arr = data[key]
        if np.issubdtype(arr.dtype, np.number):
            checks[f"Finite values in {key}"] = bool(np.all(np.isfinite(arr)))

    is_normalized = "field_mean" in data and "field_std" in data
    if "concentration" in data:
        c = (data["concentration"] * data["field_std"][0] + data["field_mean"][0]) if is_normalized else data["concentration"]
        checks["Concentration C >= 0"] = bool(np.all(c >= -1e-6))
    if "surface_coverage" in data:
        cov = (data["surface_coverage"] * data["field_std"][1] + data["field_mean"][1]) if is_normalized else data["surface_coverage"]
        checks["0 <= Surface coverage theta <= 1"] = bool(np.all((cov >= -1e-6) & (cov <= 1.0 + 1e-6)))
    if "film_thickness" in data:
        h = (data["film_thickness"] * data["field_std"][2] + data["field_mean"][2]) if is_normalized else data["film_thickness"]
        checks["Film thickness h >= 0"] = bool(np.all(h >= -1e-8))
    if "conformality" in data:
        checks["Conformality >= 0"] = bool(np.all(data["conformality"] >= 0.0))

    print("\nPhysical Invariant Verification")
    print("-" * 60)
    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status:4s}] {name}")

    return all_passed


def plot_conformality(data: Mapping[str, Any], output_dir: Path) -> None:
    """Render conformality distribution histogram if conformality array is present."""
    if "conformality" not in data:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    conformality = data["conformality"]

    plt.figure(figsize=(8, 5))
    plt.hist(conformality, bins=25, edgecolor="black", alpha=0.7)
    plt.xlabel("Step Coverage Conformality (Bottom / Top Thickness)")
    plt.ylabel("Sample Frequency")
    plt.title("Conformality Distribution Across Dataset")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "conformality_distribution.png", dpi=200)
    plt.close()


def plot_sample(data: Mapping[str, Any], index: int, output_dir: Path) -> None:
    """Render 2D spatial maps for an individual sample index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("concentration", "Precursor Concentration C", "concentration"),
        ("surface_coverage", "Surface Coverage theta", "surface_coverage"),
        ("film_thickness", "Film Thickness h", "film_thickness"),
    ]

    for key, title, fname in fields:
        if key not in data:
            continue
        arr = data[key][index]
        # If transient 3D slice (T, H, W), take final time step
        if arr.ndim == 3:
            arr = arr[-1]

        plt.figure(figsize=(8, 5))
        if key == "surface_coverage":
            im = plt.imshow(arr, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="plasma")
        elif key == "film_thickness":
            im = plt.imshow(arr, origin="lower", aspect="auto", cmap="magma")
        else:
            im = plt.imshow(arr, origin="lower", aspect="auto", cmap="viridis")

        plt.colorbar(im)
        plt.xlabel("Horizontal Grid Coordinate (x)")
        plt.ylabel("Vertical Grid Coordinate (y)")
        plt.title(f"{title} (Sample {index})")
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{index}_{fname}.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ALD dataset properties, verify invariants, and generate diagnostic figures."
    )
    parser.add_argument("--input", default="data/final_dataset.npz", help="Path to input .npz dataset.")
    parser.add_argument("--output-dir", default="results/figures/dataset_inspection", help="Output directory for plots.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        # Check alternative common locations
        alt_paths = [
            Path("data/deposition_dataset.npz").resolve(),
            Path("data/transient_ald_dataset.npz").resolve(),
            Path("data/transient_ald_prepared.npz").resolve(),
        ]
        found = False
        for alt in alt_paths:
            if alt.exists():
                input_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset archive not found at: {input_path}")

    out_dir = Path(args.output_dir).resolve()
    data = np.load(input_path)

    print(f"Inspecting dataset: {input_path}")
    print_summary(data)
    validate_physics(data)
    plot_conformality(data, out_dir)

    if "parameters" in data:
        n_samples = len(data["parameters"])
        sample_indices = [0, n_samples // 2, n_samples - 1] if n_samples >= 3 else list(range(n_samples))
        for idx in sample_indices:
            plot_sample(data, idx, out_dir)

    print("\nDataset inspection complete.")
    print(f"Diagnostic figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
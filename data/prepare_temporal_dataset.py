"""Split and normalize transient ALD trajectory datasets without cross-split data leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

FIELD_NAMES = ("concentration", "surface_coverage", "film_thickness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split and normalize a transient ALD dataset by trajectory."
    )
    parser.add_argument("--input", required=True, help="Path to input raw transient ALD .npz dataset.")
    parser.add_argument("--output", default="data/transient_ald_prepared.npz", help="Output path for prepared dataset.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for trajectory permutation.")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Fraction of trajectories for training.")
    parser.add_argument("--validation-fraction", type=float, default=0.15, help="Fraction of trajectories for validation.")
    return parser.parse_args()


def split_indices(
    num_trajectories: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition trajectory indices into disjoint train, validation, and test subsets.

    Parameters
    ----------
    num_trajectories : int
        Total number of independent simulation trajectories.
    train_fraction : float
        Proportion allocated to training.
    validation_fraction : float
        Proportion allocated to validation.
    seed : int
        RNG seed for reproducible permutation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Sorted arrays of (train_indices, validation_indices, test_indices).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.")
    if num_trajectories < 3:
        raise ValueError("At least three trajectories are required for train/val/test splitting.")

    indices = np.random.default_rng(seed).permutation(num_trajectories)

    num_train = int(np.floor(train_fraction * num_trajectories))
    num_validation = int(np.floor(validation_fraction * num_trajectories))
    num_test = num_trajectories - num_train - num_validation

    if num_validation < 1:
        num_validation = 1
        if num_train > 1:
            num_train -= 1
        else:
            num_test -= 1

    if num_test < 1:
        num_test = 1
        if num_train > 1:
            num_train -= 1
        else:
            num_validation -= 1

    num_train = num_trajectories - num_validation - num_test

    return (
        np.sort(indices[:num_train]),
        np.sort(indices[num_train : num_train + num_validation]),
        np.sort(indices[num_train + num_validation :]),
    )


def masked_stats(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation strictly over physically active cells."""
    selected = values[:, :, mask]
    mean = float(selected.mean())
    std = float(selected.std())
    return mean, max(std, 1.0e-8)


def normalize_field(
    values: np.ndarray, mask: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Z-score standardize field values using training statistics, zeroing inactive regions."""
    normalized = (values.astype(np.float32) - mean) / std
    normalized[:, :, ~mask] = 0.0
    return normalized


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    source = np.load(input_path)
    required = {
        "parameters",
        "time",
        "concentration",
        "surface_coverage",
        "film_thickness",
        "solid_mask",
        "surface_mask",
    }
    missing = sorted(required.difference(source.files))
    if missing:
        raise KeyError(f"Input dataset is missing required arrays: {', '.join(missing)}")

    parameters = source["parameters"].astype(np.float32)
    time = source["time"].astype(np.float32)
    concentration = source["concentration"]
    surface_coverage = source["surface_coverage"]
    film_thickness = source["film_thickness"]
    phase_names = (
        source["phase_names"]
        if "phase_names" in source.files
        else np.asarray(("initial", "pulse", "purge", "reaction", "post_reaction_purge"))
    )

    num_trajectories = parameters.shape[0]
    train, validation, test = split_indices(
        num_trajectories,
        args.train_fraction,
        args.validation_fraction,
        args.seed,
    )

    gas_mask = ~source["solid_mask"].astype(bool)
    surface_mask = source["surface_mask"].astype(bool)
    field_values = (concentration, surface_coverage, film_thickness)
    field_masks = (gas_mask, surface_mask, surface_mask)
    field_means = []
    field_stds = []
    normalized_fields = []

    for values, mask in zip(field_values, field_masks):
        mean, std = masked_stats(values[train], mask)
        field_means.append(mean)
        field_stds.append(std)
        normalized_fields.append(normalize_field(values, mask, mean, std))

    parameter_mean = parameters[train].mean(axis=0)
    parameter_std = np.maximum(parameters[train].std(axis=0), 1.0e-8)
    normalized_parameters = (parameters - parameter_mean) / parameter_std

    time_mean = float(time[train].mean())
    time_std = max(float(time[train].std()), 1.0e-8)
    normalized_time = (time - time_mean) / time_std

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        parameters=normalized_parameters.astype(np.float32),
        time=normalized_time.astype(np.float32),
        cycle=source["cycle"] if "cycle" in source.files else np.zeros((num_trajectories, 1), dtype=np.int16),
        phase=source["phase"] if "phase" in source.files else np.zeros((num_trajectories, 1), dtype=np.int8),
        phase_names=phase_names,
        concentration=normalized_fields[0],
        surface_coverage=normalized_fields[1],
        film_thickness=normalized_fields[2],
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        parameter_mean=parameter_mean.astype(np.float32),
        parameter_std=parameter_std.astype(np.float32),
        field_mean=np.asarray(field_means, dtype=np.float32),
        field_std=np.asarray(field_stds, dtype=np.float32),
        time_mean=np.float32(time_mean),
        time_std=np.float32(time_std),
        parameter_names=source["parameter_names"] if "parameter_names" in source.files else np.asarray([]),
        x=source["x"] if "x" in source.files else np.linspace(0.0, 1.0, concentration.shape[-1]),
        y=source["y"] if "y" in source.files else np.linspace(0.0, 1.0, concentration.shape[-2]),
        solid_mask=source["solid_mask"],
        surface_mask=source["surface_mask"],
        seed=np.int64(args.seed),
    )

    metadata = {
        "schema_version": 1,
        "source": str(input_path),
        "num_trajectories": int(num_trajectories),
        "train_trajectories": int(train.size),
        "validation_trajectories": int(validation.size),
        "test_trajectories": int(test.size),
        "normalization": "train_trajectory_statistics_only",
        "field_names": FIELD_NAMES,
        "seed": args.seed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Temporal dataset preparation completed successfully.")
    print(f"Trajectory partitioning: train={train.size}, validation={validation.size}, test={test.size}")
    print(f"Prepared dataset archive: {output}")
    print(f"Metadata summary: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

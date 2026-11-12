"""Split and normalize transient ALD trajectory datasets without cross-split data leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thin_film_pino.data.parameters import ACTIVE_PARAMETER_NAMES
from thin_film_pino.data.prepare import (
    apply_normalization,
    compute_statistics,
    stratified_split_indices,
)

FIELD_NAMES = ("concentration", "surface_coverage", "film_thickness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify and normalize a transient ALD dataset by trajectory."
    )
    parser.add_argument(
        "--input", required=True, help="Path to input raw transient ALD .npz dataset."
    )
    parser.add_argument(
        "--output",
        default="data/transient_ald_prepared.npz",
        help="Output path for prepared dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for trajectory permutation."
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.70, help="Fraction of trajectories for training."
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of trajectories for validation.",
    )
    parser.add_argument(
        "--ood-fraction",
        type=float,
        default=0.10,
        help="Fraction of longest pulse trajectories reserved for out-of-distribution evaluation.",
    )
    return parser.parse_args()


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

    raw_parameters = source["parameters"].astype(np.float32)
    time = source["time"].astype(np.float32)
    concentration = source["concentration"]
    surface_coverage = source["surface_coverage"]
    film_thickness = source["film_thickness"]

    solid_mask = source["solid_mask"].astype(bool)
    gas_mask = ~solid_mask
    surface_mask = source["surface_mask"].astype(bool)

    # 1. Stratified partitioning with out-of-distribution holdout
    train, validation, test, ood = stratified_split_indices(
        raw_parameters,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        ood_fraction=args.ood_fraction,
        seed=args.seed,
    )

    # 2. Compute statistics strictly from the training partition
    stats = compute_statistics(
        train_c=concentration[train],
        train_theta=surface_coverage[train],
        train_h=film_thickness[train],
        train_params=raw_parameters[train],
        train_time=time[train],
        gas_mask=gas_mask,
        surface_mask=surface_mask,
    )

    # 3. Apply normalization
    c_norm, theta_norm, h_norm, p_norm, t_norm = apply_normalization(
        c=concentration,
        theta=surface_coverage,
        h=film_thickness,
        params=raw_parameters,
        time=time,
        gas_mask=gas_mask,
        surface_mask=surface_mask,
        stats=stats,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Forward metric arrays if present
    extra_arrays = {
        k: source[k]
        for k in source.files
        if k.startswith("metric_") or k in ("x", "y", "phase_names", "cycle", "phase")
    }

    np.savez_compressed(
        output,
        # Normalized inputs for neural operator
        parameters=p_norm,
        time=t_norm,
        concentration=c_norm,
        surface_coverage=theta_norm,
        film_thickness=h_norm,
        # Raw physical parameters preserved for PDE residuals
        raw_parameters=raw_parameters,
        # Disjoint index splits
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        ood_indices=ood,
        # Normalization statistics
        field_mean=stats["field_mean"],
        field_std=stats["field_std"],
        parameter_mean=stats["parameter_mean"],
        parameter_std=stats["parameter_std"],
        time_mean=stats["time_mean"],
        time_std=stats["time_std"],
        active_parameter_names=np.asarray(ACTIVE_PARAMETER_NAMES),
        # Geometry masks
        solid_mask=solid_mask,
        gas_mask=gas_mask,
        surface_mask=surface_mask,
        seed=np.int64(args.seed),
        **extra_arrays,
    )

    # Relative paths for portable metadata
    try:
        rel_source = str(input_path.relative_to(Path.cwd()))
    except ValueError:
        rel_source = input_path.name

    try:
        rel_output = str(output.relative_to(Path.cwd()))
    except ValueError:
        rel_output = output.name

    metadata = {
        "schema_version": "2.0.0",
        "source": rel_source,
        "output": rel_output,
        "num_trajectories": int(raw_parameters.shape[0]),
        "train_trajectories": int(train.size),
        "validation_trajectories": int(validation.size),
        "test_trajectories": int(test.size),
        "ood_trajectories": int(ood.size),
        "normalization": "train_trajectory_statistics_only",
        "field_names": FIELD_NAMES,
        "active_parameters": list(ACTIVE_PARAMETER_NAMES),
        "seed": args.seed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Temporal dataset preparation completed.")
    print(
        f"Partitioning: train={train.size}, val={validation.size}, test={test.size}, ood={ood.size}"
    )
    print(f"Prepared dataset archive: {output}")
    print(f"Metadata summary: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np


FIELD_NAMES = ("concentration", "surface_coverage", "film_thickness")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split and normalize a transient ALD dataset by trajectory."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/transient_ald_prepared.npz")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def split_indices(num_trajectories, train_fraction, validation_fraction, seed):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.")
    if num_trajectories < 3:
        raise ValueError("At least three trajectories are required for splitting.")

    indices = np.random.default_rng(seed).permutation(num_trajectories)
    num_train = max(1, int(round(train_fraction * num_trajectories)))
    num_validation = max(1, int(round(validation_fraction * num_trajectories)))
    if num_train + num_validation >= num_trajectories:
        num_validation = num_trajectories - num_train - 1
    return (
        np.sort(indices[:num_train]),
        np.sort(indices[num_train : num_train + num_validation]),
        np.sort(indices[num_train + num_validation :]),
    )


def masked_stats(values, mask):
    selected = values[:, :, mask]
    mean = float(selected.mean())
    std = float(selected.std())
    return mean, max(std, 1.0e-8)


def normalize_field(values, mask, mean, std):
    normalized = (values.astype(np.float32) - mean) / std
    normalized[:, :, ~mask] = 0.0
    return normalized


def main():
    args = parse_args()
    source = np.load(args.input)
    required = {"parameters", "time", "concentration", "surface_coverage", "film_thickness"}
    missing = sorted(required.difference(source.files))
    if missing:
        raise KeyError(f"Input dataset is missing: {', '.join(missing)}")

    parameters = source["parameters"].astype(np.float32)
    time = source["time"].astype(np.float32)
    concentration = source["concentration"]
    surface_coverage = source["surface_coverage"]
    film_thickness = source["film_thickness"]
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        parameters=normalized_parameters.astype(np.float32),
        time=normalized_time.astype(np.float32),
        cycle=source["cycle"],
        phase=source["phase"],
        phase_names=source["phase_names"],
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
        parameter_names=source["parameter_names"],
        x=source["x"],
        y=source["y"],
        solid_mask=source["solid_mask"],
        surface_mask=source["surface_mask"],
        seed=np.int64(args.seed),
    )

    metadata = {
        "schema_version": 1,
        "source": str(args.input),
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
    print("Temporal dataset preparation complete.")
    print(f"Trajectories: train={train.size}, validation={validation.size}, test={test.size}")
    print(f"Saved data: {output}")
    print(f"Saved metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

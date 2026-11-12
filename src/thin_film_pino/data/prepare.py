"""Trajectory dataset partitioning and leakage-free normalization utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


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
        RNG seed for permutation.

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


def normalize_field(values: np.ndarray, mask: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize field values using training statistics, zeroing inactive regions."""
    normalized = (values.astype(np.float32) - mean) / std
    normalized[:, :, ~mask] = 0.0
    return normalized


def compute_statistics(
    train_c: np.ndarray,
    train_theta: np.ndarray,
    train_h: np.ndarray,
    train_params: np.ndarray,
    train_time: np.ndarray,
    gas_mask: np.ndarray,
    surface_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute normalization statistics from training trajectories only."""
    c_mean, c_std = masked_stats(train_c, gas_mask)
    theta_mean, theta_std = masked_stats(train_theta, surface_mask)
    h_mean, h_std = masked_stats(train_h, surface_mask)

    p_mean = train_params.mean(axis=0)
    p_std = train_params.std(axis=0)
    p_std[p_std < 1.0e-8] = 1.0

    t_mean = float(train_time.mean())
    t_std = max(float(train_time.std()), 1.0e-8)

    return {
        "field_mean": np.array([c_mean, theta_mean, h_mean], dtype=np.float32),
        "field_std": np.array([c_std, theta_std, h_std], dtype=np.float32),
        "parameter_mean": p_mean.astype(np.float32),
        "parameter_std": p_std.astype(np.float32),
        "time_mean": np.float32(t_mean),
        "time_std": np.float32(t_std),
    }


def apply_normalization(
    c: np.ndarray,
    theta: np.ndarray,
    h: np.ndarray,
    params: np.ndarray,
    time: np.ndarray,
    gas_mask: np.ndarray,
    surface_mask: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize arrays using provided training statistics."""
    f_mean = stats["field_mean"]
    f_std = stats["field_std"]
    c_norm = normalize_field(c, gas_mask, f_mean[0], f_std[0])
    theta_norm = normalize_field(theta, surface_mask, f_mean[1], f_std[1])
    h_norm = normalize_field(h, surface_mask, f_mean[2], f_std[2])

    p_norm = (params - stats["parameter_mean"]) / stats["parameter_std"]
    t_norm = (time - stats["time_mean"]) / stats["time_std"]
    return c_norm, theta_norm, h_norm, p_norm, t_norm

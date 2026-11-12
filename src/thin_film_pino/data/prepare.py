"""Trajectory dataset partitioning, stratification, and leakage-free normalization utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def stratified_split_indices(
    parameters: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    ood_fraction: float = 0.10,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition trajectory indices into train, validation, test, and out-of-distribution (OOD) subsets.

    Stratifies in-distribution samples by pulse-time regime (short vs. long pulse) to guarantee
    balanced representation across splits, and reserves the longest pulse trajectories for the OOD set.

    Parameters
    ----------
    parameters : np.ndarray
        Array of shape (N, 8) or (N, 5+) containing parameter values.
    train_fraction : float, default=0.70
        Fraction of in-distribution samples for training.
    validation_fraction : float, default=0.15
        Fraction of in-distribution samples for validation.
    ood_fraction : float, default=0.10
        Fraction of total trajectories reserved for the long-pulse OOD test set.
    seed : int, default=123
        Random seed for shuffling.

    Returns
    -------
    train_indices : np.ndarray
    validation_indices : np.ndarray
    test_indices : np.ndarray
    ood_indices : np.ndarray
    """
    num_trajectories = int(parameters.shape[0])
    if num_trajectories < 4:
        raise ValueError(
            "At least four trajectories are required for train/val/test/ood splitting."
        )

    # Locate pulse_time index (index 5 in standard 8-parameter array, or last column)
    pulse_idx = 5 if parameters.shape[1] > 5 else parameters.shape[1] - 1
    pulse_times = parameters[:, pulse_idx]

    rng = np.random.default_rng(seed)

    # 1. Select Out-of-Distribution (OOD) trajectories: longest pulses
    num_ood = max(1, int(np.round(ood_fraction * num_trajectories)))
    sorted_by_pulse = np.argsort(pulse_times)
    ood_indices = sorted_by_pulse[-num_ood:]
    in_distribution_indices = sorted_by_pulse[:-num_ood]

    # 2. Stratify in-distribution samples by pulse-time regime (cutoff at 60.0)
    in_pulses = pulse_times[in_distribution_indices]
    short_mask = in_pulses <= 60.0
    long_mask = in_pulses > 60.0

    short_indices = in_distribution_indices[short_mask]
    long_indices = in_distribution_indices[long_mask]

    def _split_subset(subset: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        shuffled = rng.permutation(subset)
        n = len(shuffled)
        if n == 0:
            return [], [], []
        n_tr = max(1, int(np.floor(train_fraction * n)))
        n_val = max(1 if n >= 3 else 0, int(np.round(validation_fraction * n)))
        # Ensure at least 1 test sample if subset is reasonably sized
        if n - n_tr - n_val < 1 and n >= 3:
            n_tr = max(1, n_tr - 1)
        tr = shuffled[:n_tr].tolist()
        val = shuffled[n_tr : n_tr + n_val].tolist()
        te = shuffled[n_tr + n_val :].tolist()
        return tr, val, te

    tr_short, val_short, te_short = _split_subset(short_indices)
    tr_long, val_long, te_long = _split_subset(long_indices)

    train_all = np.array(sorted(tr_short + tr_long), dtype=np.int64)
    val_all = np.array(sorted(val_short + val_long), dtype=np.int64)
    test_all = np.array(sorted(te_short + te_long), dtype=np.int64)
    ood_all = np.array(sorted(ood_indices.tolist()), dtype=np.int64)

    # Safety: ensure non-empty validation and test splits
    if len(val_all) == 0 and len(train_all) > 1:
        val_all = np.array([train_all[-1]])
        train_all = train_all[:-1]
    if len(test_all) == 0 and len(train_all) > 1:
        test_all = np.array([train_all[-1]])
        train_all = train_all[:-1]

    return train_all, val_all, test_all, ood_all


def split_indices(
    num_trajectories: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for random trajectory splitting without stratification."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_trajectories)

    n_train = int(np.floor(train_fraction * num_trajectories))
    n_val = int(np.floor(validation_fraction * num_trajectories))

    train = np.sort(shuffled[:n_train])
    val = np.sort(shuffled[n_train : n_train + n_val])
    test = np.sort(shuffled[n_train + n_val :])
    return train, val, test


def extract_active_parameters(parameters: np.ndarray) -> np.ndarray:
    """Extract varying parameter columns, dropping constant parameters."""
    if parameters.shape[1] >= 8:
        # Indices 0 (diffusivity), 1 (k_ads), 2 (k_des), 3 (k_rxn), 5 (pulse_time)
        return parameters[:, [0, 1, 2, 3, 5]].astype(np.float32)
    return parameters.astype(np.float32)


def masked_stats(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation strictly over physically active cells."""
    if values.ndim == 4:
        selected = values[:, :, mask]
    elif values.ndim == 3:
        selected = values[:, mask]
    else:
        selected = values[mask]

    mean = float(selected.mean())
    std = float(selected.std())
    return mean, max(std, 1.0e-8)


def normalize_field(values: np.ndarray, mask: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize field values using training statistics, zeroing inactive regions."""
    normalized = (values.astype(np.float32) - mean) / std
    if normalized.ndim == 4:
        normalized[:, :, ~mask] = 0.0
    elif normalized.ndim == 3:
        normalized[:, ~mask] = 0.0
    else:
        normalized[~mask] = 0.0
    return normalized


def denormalize_field(
    normalized: np.ndarray, mask: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Denormalize field values and re-apply zero mask to inactive cells.

    Because normalized zero is not physical zero (0 * std + mean = mean != 0),
    cells outside the mask must be explicitly re-zeroed.
    """
    denorm = normalized.astype(np.float32) * std + mean
    if denorm.ndim == 4:
        denorm[:, :, ~mask] = 0.0
    elif denorm.ndim == 3:
        denorm[:, ~mask] = 0.0
    else:
        denorm[~mask] = 0.0
    return denorm


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

    active_params = extract_active_parameters(train_params)
    p_mean = active_params.mean(axis=0)
    p_std = active_params.std(axis=0)
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
    c_norm = normalize_field(c, gas_mask, float(f_mean[0]), float(f_std[0]))
    theta_norm = normalize_field(theta, surface_mask, float(f_mean[1]), float(f_std[1]))
    h_norm = normalize_field(h, surface_mask, float(f_mean[2]), float(f_std[2]))

    active_params = extract_active_parameters(params)
    p_norm = (active_params - stats["parameter_mean"]) / stats["parameter_std"]
    t_norm = (time.astype(np.float32) - stats["time_mean"]) / stats["time_std"]

    return c_norm, theta_norm, h_norm, p_norm.astype(np.float32), t_norm.astype(np.float32)

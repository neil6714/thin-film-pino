"""Evaluation metrics, diagnostic checks, and baseline predictors for thin-film models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_channel_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Compute physical metrics, trajectory-wise relative L2, and guarded R-squared.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth physical field array of shape (N, T, H, W).
    y_pred : np.ndarray
        Predicted physical field array of shape (N, T, H, W).
    mask : np.ndarray
        Boolean spatial domain mask of shape (H, W).
    epsilon : float, default=1.0e-12
        Numerical stability constant for denominators.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing MAE, RMSE, relative L2 spread statistics, and guarded R2.
    """
    n_traj, n_time, height, width = y_true.shape
    mask_bool = mask.astype(bool)
    n_active_spatial = int(np.count_nonzero(mask_bool))
    if n_active_spatial == 0:
        raise ValueError("Provided spatial mask contains zero active cells.")

    # Flatten active spatial cells across all trajectories and time steps
    y_true_active = y_true[:, :, mask_bool]  # (N, T, M)
    y_pred_active = y_pred[:, :, mask_bool]  # (N, T, M)

    diff = y_pred_active - y_true_active
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))

    # Trajectory-wise relative L2 norm: ||y_pred_i - y_true_i||_2 / ||y_true_i||_2
    diff_per_traj = diff.reshape(n_traj, -1)
    true_per_traj = y_true_active.reshape(n_traj, -1)
    l2_diff = np.sqrt(np.sum(np.square(diff_per_traj), axis=1))
    l2_true = np.sqrt(np.sum(np.square(true_per_traj), axis=1))
    rel_l2 = l2_diff / np.maximum(l2_true, epsilon)

    relative_l2_stats = {
        "mean": float(np.mean(rel_l2)),
        "std": float(np.std(rel_l2)),
        "min": float(np.min(rel_l2)),
        "max": float(np.max(rel_l2)),
        "median": float(np.median(rel_l2)),
        "per_trajectory": [float(val) for val in rel_l2],
    }

    # Guarded R-squared computation
    y_flat = y_true_active.ravel()
    pred_flat = y_pred_active.ravel()
    ss_tot = float(np.sum(np.square(y_flat - np.mean(y_flat))))
    ss_res = float(np.sum(np.square(y_flat - pred_flat)))
    total_samples = y_flat.size
    sample_variance = ss_tot / max(total_samples, 1)

    r2_dict: Dict[str, Any] = {
        "R2_denominator_SS_tot": ss_tot,
        "sample_variance": sample_variance,
    }

    # Drop R-squared when target variance is near zero to prevent spurious metrics
    if sample_variance < 1.0e-10 or ss_tot < 1.0e-8:
        r2_dict["R2"] = None
        r2_dict["R2_status"] = "dropped_due_to_near_zero_variance"
    else:
        r2_dict["R2"] = float(1.0 - (ss_res / max(ss_tot, epsilon)))
        r2_dict["R2_status"] = "computed"

    return {
        "MAE": mae,
        "RMSE": rmse,
        "relative_L2": relative_l2_stats,
        **r2_dict,
    }


def compute_per_phase_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    phase: np.ndarray,
    phase_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute channel MAE and RMSE broken down by ALD process phase.

    Parameters
    ----------
    y_true : np.ndarray
        Array of shape (N, T, H, W).
    y_pred : np.ndarray
        Array of shape (N, T, H, W).
    mask : np.ndarray
        Spatial mask of shape (H, W).
    phase : np.ndarray
        Phase indicator array of shape (N, T) or (T,).
    phase_names : Optional[Dict[int, str]], default=None
        Mapping of phase integers to descriptive names.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Metrics mapped by phase name or phase index.
    """
    if phase_names is None:
        phase_names = {1: "dose", 2: "post_dose_purge", 3: "reaction", 4: "post_rxn_purge"}

    if phase.ndim == 1:
        phase = np.broadcast_to(phase[None, :], (y_true.shape[0], y_true.shape[1]))

    mask_bool = mask.astype(bool)
    results: Dict[str, Dict[str, float]] = {}

    unique_phases = sorted(int(p) for p in np.unique(phase))
    for p_val in unique_phases:
        p_name = phase_names.get(p_val, f"phase_{p_val}")
        time_mask = phase == p_val  # (N, T)
        if not np.any(time_mask):
            continue

        # Extract elements matching phase and spatial mask
        sub_true = y_true[time_mask][:, mask_bool]
        sub_pred = y_pred[time_mask][:, mask_bool]
        diff = sub_pred - sub_true

        results[p_name] = {
            "phase_index": p_val,
            "MAE": float(np.mean(np.abs(diff))),
            "RMSE": float(np.sqrt(np.mean(np.square(diff)))),
            "snapshots_evaluated": int(np.count_nonzero(time_mask)),
        }

    return results


def compute_derived_cycle_quantities(
    pred_h: np.ndarray,
    ref_h: np.ndarray,
    surface_mask: np.ndarray,
    field_mask: np.ndarray,
    floor_mask: np.ndarray,
    cycle: Optional[np.ndarray] = None,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Compute Growth Per Cycle (GPC) and step coverage errors per cycle.

    Parameters
    ----------
    pred_h : np.ndarray
        Predicted film thickness field of shape (N, T, H, W).
    ref_h : np.ndarray
        Reference film thickness field of shape (N, T, H, W).
    surface_mask : np.ndarray
        Active surface boundary mask (H, W).
    field_mask : np.ndarray
        Top wafer surface mask outside the trench (H, W).
    floor_mask : np.ndarray
        Bottom trench floor surface mask (H, W).
    cycle : Optional[np.ndarray], default=None
        Cycle index array of shape (N, T) or (T,).
    epsilon : float, default=1.0e-12
        Tolerance to avoid division by zero.

    Returns
    -------
    Dict[str, Any]
        Per-cycle and aggregate GPC and step coverage errors.
    """
    n_traj, n_time, _, _ = pred_h.shape
    surf = surface_mask.astype(bool)
    field = field_mask.astype(bool)
    floor = floor_mask.astype(bool)

    if cycle is None:
        # If cycle is not provided, evaluate final state as cycle 1
        cycle = np.ones((n_traj, n_time), dtype=np.int64)
    elif cycle.ndim == 1:
        cycle = np.broadcast_to(cycle[None, :], (n_traj, n_time))

    unique_cycles = sorted(int(c) for c in np.unique(cycle) if c > 0)
    if not unique_cycles:
        unique_cycles = [1]

    gpc_errors: List[float] = []
    sc_errors: List[float] = []
    cycle_records: List[Dict[str, Any]] = []

    # Track preceding cycle thickness for incremental GPC calculation
    prev_h_pred_mean = np.zeros(n_traj, dtype=np.float64)
    prev_h_ref_mean = np.zeros(n_traj, dtype=np.float64)

    for c_val in unique_cycles:
        # Identify the terminal snapshot for this cycle along the trajectory
        # Terminal snapshot is the last time step corresponding to cycle c_val
        c_matches = np.where(cycle == c_val)
        time_indices = c_matches[1]
        if len(time_indices) == 0:
            continue
        terminal_t = int(np.max(time_indices))

        pred_t = pred_h[:, terminal_t]  # (N, H, W)
        ref_t = ref_h[:, terminal_t]  # (N, H, W)

        # 1. Surface mean thickness and GPC
        mean_h_pred = np.mean(pred_t[:, surf], axis=1)  # (N,)
        mean_h_ref = np.mean(ref_t[:, surf], axis=1)  # (N,)

        gpc_pred = mean_h_pred - prev_h_pred_mean
        gpc_ref = mean_h_ref - prev_h_ref_mean

        prev_h_pred_mean = mean_h_pred
        prev_h_ref_mean = mean_h_ref

        # 2. Step coverage: floor thickness divided by field thickness
        floor_h_pred = np.mean(pred_t[:, floor], axis=1) if np.any(floor) else mean_h_pred
        field_h_pred = np.mean(pred_t[:, field], axis=1) if np.any(field) else mean_h_pred
        sc_pred = floor_h_pred / np.maximum(field_h_pred, epsilon)

        floor_h_ref = np.mean(ref_t[:, floor], axis=1) if np.any(floor) else mean_h_ref
        field_h_ref = np.mean(ref_t[:, field], axis=1) if np.any(field) else mean_h_ref
        sc_ref = floor_h_ref / np.maximum(field_h_ref, epsilon)

        gpc_diff = gpc_pred - gpc_ref
        sc_diff = sc_pred - sc_ref

        gpc_mae = float(np.mean(np.abs(gpc_diff)))
        sc_mae = float(np.mean(np.abs(sc_diff)))

        gpc_errors.extend(np.abs(gpc_diff).tolist())
        sc_errors.extend(np.abs(sc_diff).tolist())

        cycle_records.append(
            {
                "cycle": c_val,
                "terminal_time_index": terminal_t,
                "gpc_mae": gpc_mae,
                "gpc_ref_mean": float(np.mean(gpc_ref)),
                "gpc_pred_mean": float(np.mean(gpc_pred)),
                "step_coverage_mae": sc_mae,
                "step_coverage_ref_mean": float(np.mean(sc_ref)),
                "step_coverage_pred_mean": float(np.mean(sc_pred)),
            }
        )

    return {
        "gpc_mae_overall": float(np.mean(gpc_errors)) if gpc_errors else 0.0,
        "gpc_rmse_overall": float(np.sqrt(np.mean(np.square(gpc_errors)))) if gpc_errors else 0.0,
        "step_coverage_mae_overall": float(np.mean(sc_errors)) if sc_errors else 0.0,
        "step_coverage_rmse_overall": (
            float(np.sqrt(np.mean(np.square(sc_errors)))) if sc_errors else 0.0
        ),
        "per_cycle": cycle_records,
    }


def compute_bounds_violations(
    pred_c: np.ndarray,
    pred_theta: np.ndarray,
    gas_mask: np.ndarray,
    surface_mask: np.ndarray,
    phase: Optional[np.ndarray] = None,
    phase_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute physical domain boundary violations for concentration and surface coverage.

    Parameters
    ----------
    pred_c : np.ndarray
        Predicted concentration in physical units (N, T, H, W).
    pred_theta : np.ndarray
        Predicted coverage in physical units (N, T, H, W).
    gas_mask : np.ndarray
        Active gas mask (H, W).
    surface_mask : np.ndarray
        Active surface boundary mask (H, W).
    phase : Optional[np.ndarray], default=None
        Phase indicator array of shape (N, T) or (T,).
    phase_names : Optional[Dict[int, str]], default=None
        Mapping of phase integers to names.

    Returns
    -------
    Dict[str, Any]
        Diagnostics summarizing lower and upper bounds violations overall and per phase.
    """
    if phase_names is None:
        phase_names = {1: "dose", 2: "post_dose_purge", 3: "reaction", 4: "post_rxn_purge"}

    gas_bool = gas_mask.astype(bool)
    surf_bool = surface_mask.astype(bool)

    c_active = pred_c[:, :, gas_bool]
    theta_active = pred_theta[:, :, surf_bool]

    # Global bounds violations
    c_negative = c_active < 0.0
    c_neg_fraction = float(np.mean(c_negative))
    c_max_neg = float(np.max(-c_active[c_negative])) if np.any(c_negative) else 0.0

    theta_lower = theta_active < 0.0
    theta_lower_fraction = float(np.mean(theta_lower))
    theta_max_lower = float(np.max(-theta_active[theta_lower])) if np.any(theta_lower) else 0.0

    theta_upper = theta_active > 1.0
    theta_upper_fraction = float(np.mean(theta_upper))
    theta_max_upper = float(np.max(theta_active[theta_upper] - 1.0)) if np.any(theta_upper) else 0.0

    diagnostics: Dict[str, Any] = {
        "overall": {
            "concentration_negative_fraction": c_neg_fraction,
            "concentration_max_negative_violation": c_max_neg,
            "coverage_below_zero_fraction": theta_lower_fraction,
            "coverage_max_below_zero_violation": theta_max_lower,
            "coverage_above_one_fraction": theta_upper_fraction,
            "coverage_max_above_one_violation": theta_max_upper,
        }
    }

    if phase is not None:
        if phase.ndim == 1:
            phase = np.broadcast_to(phase[None, :], (pred_c.shape[0], pred_c.shape[1]))

        per_phase: Dict[str, Dict[str, float]] = {}
        for p_val in sorted(int(p) for p in np.unique(phase)):
            p_name = phase_names.get(p_val, f"phase_{p_val}")
            t_mask = phase == p_val
            if not np.any(t_mask):
                continue

            c_sub = pred_c[t_mask][:, gas_bool]
            theta_sub = pred_theta[t_mask][:, surf_bool]

            c_sub_neg = c_sub < 0.0
            th_sub_low = theta_sub < 0.0
            th_sub_high = theta_sub > 1.0

            per_phase[p_name] = {
                "concentration_negative_fraction": float(np.mean(c_sub_neg)),
                "concentration_max_negative": (
                    float(np.max(-c_sub[c_sub_neg])) if np.any(c_sub_neg) else 0.0
                ),
                "coverage_below_zero_fraction": float(np.mean(th_sub_low)),
                "coverage_max_below_zero": (
                    float(np.max(-theta_sub[th_sub_low])) if np.any(th_sub_low) else 0.0
                ),
                "coverage_above_one_fraction": float(np.mean(th_sub_high)),
                "coverage_max_above_one": (
                    float(np.max(theta_sub[th_sub_high] - 1.0)) if np.any(th_sub_high) else 0.0
                ),
            }

        diagnostics["per_phase"] = per_phase

    return diagnostics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 123,
) -> Dict[str, float]:
    """Compute non-parametric bootstrap confidence intervals across test trajectories.

    Resamples trajectories with replacement to evaluate test performance uncertainty.

    Parameters
    ----------
    y_true : np.ndarray
        Array of shape (N, T, H, W).
    y_pred : np.ndarray
        Array of shape (N, T, H, W).
    mask : np.ndarray
        Spatial mask of shape (H, W).
    n_bootstraps : int, default=500
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level for two-tailed (1 - alpha) confidence interval.
    seed : int, default=123
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Dictionary with bootstrap mean, standard deviation, and lower/upper bounds for MAE.
    """
    n_traj = y_true.shape[0]
    if n_traj < 2:
        diff = np.abs(y_pred[:, :, mask.astype(bool)] - y_true[:, :, mask.astype(bool)])
        val = float(np.mean(diff))
        return {"mae_mean": val, "mae_std": 0.0, "mae_ci_lower": val, "mae_ci_upper": val}

    rng = np.random.default_rng(seed)
    mask_bool = mask.astype(bool)

    # Compute per-trajectory mean absolute error
    diff_per_traj = np.abs(y_pred - y_true)[:, :, mask_bool]  # (N, T, M)
    traj_maes = np.mean(diff_per_traj, axis=(1, 2))  # (N,)

    boot_maes = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        resample_idx = rng.choice(n_traj, size=n_traj, replace=True)
        boot_maes[b] = np.mean(traj_maes[resample_idx])

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    return {
        "mae_mean": float(np.mean(boot_maes)),
        "mae_std": float(np.std(boot_maes)),
        "mae_ci_lower": float(np.percentile(boot_maes, lower_pct)),
        "mae_ci_upper": float(np.percentile(boot_maes, upper_pct)),
    }


def compute_baseline_predictions(
    data: Dict[str, Any],
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate train-mean and persistence baseline predictions in physical units.

    Parameters
    ----------
    data : Dict[str, Any]
        Prepared dataset dictionary containing normalization statistics and targets.
    test_idx : np.ndarray
        Indices of test trajectories.

    Returns
    -------
    train_mean_pred : np.ndarray
        Array of shape (N_test, T, 3, H, W) in physical units.
    persistence_pred : np.ndarray
        Array of shape (N_test, T, 3, H, W) in physical units.
    """
    c_target = data["concentration"][test_idx]
    theta_target = data["surface_coverage"][test_idx]
    h_target = data["film_thickness"][test_idx]

    f_mean = data["field_mean"]
    f_std = data["field_std"]
    gas_mask = data["gas_mask"].astype(bool)
    surf_mask = data["surface_mask"].astype(bool)

    n_test, n_time, height, width = c_target.shape

    # Denormalize targets to exact physical units
    c_phys = c_target * f_std[0] + f_mean[0]
    c_phys[:, :, ~gas_mask] = 0.0

    theta_phys = theta_target * f_std[1] + f_mean[1]
    theta_phys[:, :, ~surf_mask] = 0.0

    h_phys = h_target * f_std[2] + f_mean[2]
    h_phys[:, :, ~surf_mask] = 0.0

    # 1. Train-mean predictor: predicts training mean on active domain, 0 outside
    c_mean_pred = np.zeros_like(c_phys)
    c_mean_pred[:, :, gas_mask] = f_mean[0]

    theta_mean_pred = np.zeros_like(theta_phys)
    theta_mean_pred[:, :, surf_mask] = f_mean[1]

    h_mean_pred = np.zeros_like(h_phys)
    h_mean_pred[:, :, surf_mask] = f_mean[2]

    train_mean_pred = np.stack([c_mean_pred, theta_mean_pred, h_mean_pred], axis=2)

    # 2. Persistence predictor: holds initial snapshot t=0 constant for all t
    c_persist = np.repeat(c_phys[:, :1], n_time, axis=1)
    theta_persist = np.repeat(theta_phys[:, :1], n_time, axis=1)
    h_persist = np.repeat(h_phys[:, :1], n_time, axis=1)

    persistence_pred = np.stack([c_persist, theta_persist, h_persist], axis=2)

    return train_mean_pred, persistence_pred

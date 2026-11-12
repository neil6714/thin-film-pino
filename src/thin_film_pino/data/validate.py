"""Integrity, physical invariant, and split verification suite for ALD datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from thin_film_pino.data.prepare import denormalize_field, masked_stats


def validate_dataset(
    data: Mapping[str, Any], input_path: Path | str = "dataset"
) -> Tuple[bool, Dict[str, Any]]:
    """Execute verification checks across raw or prepared ALD dataset.

    Parameters
    ----------
    data : Mapping[str, Any]
        Dictionary or np.lib.npyio.NpzFile containing dataset arrays.
    input_path : Path or str, default='dataset'
        Label or path for reporting.

    Returns
    -------
    passed : bool
        True if all verification checks pass, False otherwise.
    report : dict
        Structured report of informational flags, array shapes, and check results.
    """
    info: Dict[str, Any] = {}
    checks: Dict[str, bool] = {}

    # Check finite numbers across all numerical arrays
    no_nan_or_inf = True
    for key, arr in data.items():
        if np.issubdtype(arr.dtype, np.number):
            if not np.all(np.isfinite(arr)):
                no_nan_or_inf = False
                break
    checks["no_nan_or_inf"] = no_nan_or_inf

    raw_c = data["concentration"]
    raw_theta = data["surface_coverage"]
    raw_h = data["film_thickness"]

    # 1. Shape consistency
    shape_ok = raw_c.shape == raw_theta.shape == raw_h.shape
    if "time" in data:
        if raw_c.ndim == 4:
            shape_ok = shape_ok and (data["time"].shape == raw_c.shape[:2])
        elif raw_c.ndim == 3:
            shape_ok = shape_ok and (data["time"].shape == (raw_c.shape[0],))
    if "parameters" in data:
        shape_ok = shape_ok and (data["parameters"].shape[0] == raw_c.shape[0])
    checks["shape_consistency"] = bool(shape_ok)

    # 2. Geometry and masks
    solid_mask = data["solid_mask"].astype(bool)
    gas_mask = ~solid_mask if "gas_mask" not in data else data["gas_mask"].astype(bool)
    surface_mask = data["surface_mask"].astype(bool)

    # 3. Normalization detection & denormalization for physical checks
    is_normalized = "field_mean" in data and "field_std" in data
    info["is_normalized"] = bool(is_normalized)

    if is_normalized:
        f_mean = data["field_mean"]
        f_std = data["field_std"]
        c = denormalize_field(raw_c, gas_mask, float(f_mean[0]), float(f_std[0]))
        theta = denormalize_field(raw_theta, surface_mask, float(f_mean[1]), float(f_std[1]))
        h = denormalize_field(raw_h, surface_mask, float(f_mean[2]), float(f_std[2]))
        info["fields_denormalized_for_physics_checks"] = True
    else:
        c = raw_c
        theta = raw_theta
        h = raw_h
        info["fields_denormalized_for_physics_checks"] = False

    # 4. Physical field bounds
    checks["concentration_nonnegative"] = bool(np.all(c >= -1.0e-6))
    checks["coverage_bounded"] = bool(np.all((theta >= -1.0e-6) & (theta <= 1.0 + 1.0e-6)))
    checks["thickness_nonnegative"] = bool(np.all(h >= -1.0e-8))

    # Mask compliance
    checks["c_zero_in_solid"] = bool(np.all(np.abs(c[..., solid_mask]) < 1.0e-6))
    checks["theta_zero_off_surface"] = bool(np.all(np.abs(theta[..., ~surface_mask]) < 1.0e-6))
    checks["h_zero_off_surface"] = bool(np.all(np.abs(h[..., ~surface_mask]) < 1.0e-6))

    # Monotonic thickness growth on reacting surface
    if h.ndim == 4 and h.shape[1] > 1:
        diff_h = np.diff(h[..., surface_mask], axis=1)
        checks["thickness_non_decreasing"] = bool(np.all(diff_h >= -1.0e-7))
    else:
        checks["thickness_non_decreasing"] = True

    # 5. Temporal and Phase Consistency
    if "phase" in data and data["phase"].ndim >= 2:
        phase = data["phase"]
        checks["phase_ids_valid"] = bool(np.all(np.isin(phase, np.arange(5))))

        # C must be zero at end of purge phases (last snapshot of phase 2 and phase 4)
        is_purge = (phase == 2) | (phase == 4)
        is_purge_end = np.zeros_like(is_purge, dtype=bool)
        is_purge_end[:, -1] = is_purge[:, -1]
        is_purge_end[:, :-1] = is_purge[:, :-1] & (phase[:, 1:] != phase[:, :-1])
        c_at_purge = c[is_purge_end]
        checks["c_zero_at_purge_ends"] = bool(np.all(np.abs(c_at_purge) < 1.0e-6))

        # Phase ordering within cycle: phase IDs non-decreasing within each cycle
        if "cycle" in data:
            cycle = data["cycle"]
            ordering_ok = True
            for c_id in np.unique(cycle):
                if c_id == 0:
                    continue
                c_mask = cycle == c_id
                for b in range(phase.shape[0]):
                    sample_phases = phase[b, c_mask[b]]
                    if len(sample_phases) > 1 and np.any(np.diff(sample_phases) < 0):
                        ordering_ok = False
                        break
                if not ordering_ok:
                    break
            checks["phase_ordering_within_cycle"] = bool(ordering_ok)
        else:
            checks["phase_ordering_within_cycle"] = True
    else:
        checks["phase_ids_valid"] = True
        checks["c_zero_at_purge_ends"] = True
        checks["phase_ordering_within_cycle"] = True

    # Time monotonicity within phase
    if "time" in data and data["time"].ndim >= 2:
        diff_t = np.diff(data["time"], axis=1)
        checks["time_increasing_within_phase"] = bool(np.all(diff_t >= -1.0e-8))
    else:
        checks["time_increasing_within_phase"] = True

    # GPC non-negative
    if "metric_gpc" in data:
        checks["gpc_nonnegative"] = bool(np.all(data["metric_gpc"] >= -1.0e-8))
    else:
        checks["gpc_nonnegative"] = True

    # 6. Partition and Split Verification
    is_prepared = "train_indices" in data
    info["is_prepared"] = bool(is_prepared)

    if is_prepared:
        num_trajectories = raw_c.shape[0]
        train = set(data["train_indices"].tolist())
        val = set(data["validation_indices"].tolist())
        test = set(data["test_indices"].tolist())
        ood = set(data["ood_indices"].tolist()) if "ood_indices" in data else set()

        # Disjointness
        disjoint = (
            len(train & val) == 0
            and len(train & test) == 0
            and len(train & ood) == 0
            and len(val & test) == 0
            and len(val & ood) == 0
            and len(test & ood) == 0
        )
        checks["splits_disjoint"] = bool(disjoint)

        # Coverage: union of splits must cover all trajectory indices
        all_indices = train | val | test | ood
        checks["split_coverage"] = bool(all_indices == set(range(num_trajectories)))

        # Verify stored normalization statistics match recomputed training stats
        if "raw_parameters" in data:
            tr_idx = np.array(sorted(train), dtype=np.int64)
            recomputed_c_mean, recomputed_c_std = masked_stats(c[tr_idx], gas_mask)
            recomputed_th_mean, recomputed_th_std = masked_stats(theta[tr_idx], surface_mask)
            recomputed_h_mean, recomputed_h_std = masked_stats(h[tr_idx], surface_mask)

            stored_f_mean = data["field_mean"]
            stored_f_std = data["field_std"]

            mean_match = np.allclose(
                [recomputed_c_mean, recomputed_th_mean, recomputed_h_mean],
                stored_f_mean,
                rtol=1.0e-4,
                atol=1.0e-4,
            )
            std_match = np.allclose(
                [recomputed_c_std, recomputed_th_std, recomputed_h_std],
                stored_f_std,
                rtol=1.0e-4,
                atol=1.0e-4,
            )
            checks["train_statistics_match"] = bool(mean_match and std_match)
        else:
            checks["train_statistics_match"] = True

    all_passed = bool(all(checks.values()))

    report = {
        "input": str(input_path),
        "info": info,
        "shapes": {k: list(data[k].shape) for k in data.keys() if hasattr(data[k], "shape")},
        "checks": checks,
        "passed": all_passed,
    }
    return all_passed, report

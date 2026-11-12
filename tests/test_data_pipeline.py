"""Unit tests for dataset generation, reproducible sampling, stratification, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thin_film_pino.data.generate import generate
from thin_film_pino.data.parameters import (
    ACTIVE_PARAMETER_NAMES,
    PARAMETER_NAMES,
    load_priors,
    sample_parameter_ensemble,
)
from thin_film_pino.data.prepare import (
    apply_normalization,
    compute_statistics,
    denormalize_field,
    extract_active_parameters,
    stratified_split_indices,
)
from thin_film_pino.data.validate import validate_dataset


def test_priors_loading_and_reproducible_sampling() -> None:
    """Verify priors loading and exact SeedSequence sample reproducibility."""
    priors = load_priors()
    assert "parameters" in priors
    for name in PARAMETER_NAMES:
        assert name in priors["parameters"]

    # Sample reproducibility: two ensemble calls with same seed must yield identical arrays
    ens1, _ = sample_parameter_ensemble(num_samples=10, seed=42, priors=priors)
    ens2, _ = sample_parameter_ensemble(num_samples=10, seed=42, priors=priors)
    np.testing.assert_array_equal(ens1, ens2)

    # Different seeds must differ
    ens3, _ = sample_parameter_ensemble(num_samples=10, seed=99, priors=priors)
    assert not np.array_equal(ens1, ens3)

    # Constant parameters must hold nominal values
    np.testing.assert_allclose(ens1[:, 4], 0.02)  # k_growth
    np.testing.assert_allclose(ens1[:, 6], 1.0)  # purge_time
    np.testing.assert_allclose(ens1[:, 7], 1.0)  # reaction_time


def test_stratified_splitting_and_active_parameters() -> None:
    """Verify stratified splitting by pulse time regime and active parameter filtering."""
    priors = load_priors()
    params, _ = sample_parameter_ensemble(num_samples=20, seed=123, priors=priors)

    train, val, test, ood = stratified_split_indices(
        params,
        train_fraction=0.60,
        validation_fraction=0.20,
        ood_fraction=0.10,
        seed=123,
    )

    # Disjointness
    s_tr, s_val, s_te, s_ood = set(train), set(val), set(test), set(ood)
    assert len(s_tr & s_val) == 0
    assert len(s_tr & s_te) == 0
    assert len(s_tr & s_ood) == 0
    assert len(s_val & s_te) == 0
    assert len(s_val & s_ood) == 0
    assert len(s_te & s_ood) == 0

    # Complete coverage
    assert (s_tr | s_val | s_te | s_ood) == set(range(20))

    # OOD contains the largest pulse times
    pulse_times = params[:, 5]
    ood_min_pulse = pulse_times[ood].min()
    in_dist_max_pulse = pulse_times[list(s_tr | s_val | s_te)].max()
    assert ood_min_pulse >= in_dist_max_pulse

    # Active parameters extraction: drops columns 4, 6, 7
    active = extract_active_parameters(params)
    assert active.shape == (20, len(ACTIVE_PARAMETER_NAMES))
    np.testing.assert_allclose(active[:, 0], params[:, 0], rtol=1e-5)  # D
    np.testing.assert_allclose(active[:, 4], params[:, 5], rtol=1e-5)  # pulse_time


def test_masked_normalization_and_denormalization() -> None:
    """Verify that denormalizing correctly preserves zero on inactive cells outside masks."""
    ny, nx = 15, 20
    gas_mask = np.ones((ny, nx), dtype=bool)
    gas_mask[10:, :] = False  # solid substrate
    surface_mask = np.zeros((ny, nx), dtype=bool)
    surface_mask[9, :] = True  # single line of surface cells

    # Artificial field: 1.0 on gas, 0.0 on solid
    c_raw = np.where(gas_mask, 2.5, 0.0).astype(np.float32)[None, None, :, :]

    stats = compute_statistics(
        train_c=c_raw,
        train_theta=np.zeros_like(c_raw),
        train_h=np.zeros_like(c_raw),
        train_params=np.zeros((1, 8), dtype=np.float32),
        train_time=np.zeros((1, 1), dtype=np.float32),
        gas_mask=gas_mask,
        surface_mask=surface_mask,
    )

    c_norm, _, _, _, _ = apply_normalization(
        c=c_raw,
        theta=np.zeros_like(c_raw),
        h=np.zeros_like(c_raw),
        params=np.zeros((1, 8), dtype=np.float32),
        time=np.zeros((1, 1), dtype=np.float32),
        gas_mask=gas_mask,
        surface_mask=surface_mask,
        stats=stats,
    )

    # Inactive cells in normalized array are 0
    assert np.all(c_norm[..., ~gas_mask] == 0.0)

    # Denormalizing must re-zero inactive cells, not leave them at field_mean
    c_denorm = denormalize_field(
        c_norm, gas_mask, float(stats["field_mean"][0]), float(stats["field_std"][0])
    )
    assert np.all(c_denorm[..., ~gas_mask] == 0.0)
    np.testing.assert_allclose(c_denorm[..., gas_mask], c_raw[..., gas_mask], atol=1e-5)


def test_unified_generator_and_validator(tmp_path: Path) -> None:
    """Verify unified dataset generator and validator on small CPU grid."""
    out_traj = tmp_path / "test_traj.npz"
    out_final = tmp_path / "test_final.npz"

    # 1. Trajectory mode generation
    generate(
        mode="trajectory",
        num_samples=2,
        batch_size=2,
        num_cycles=2,
        snapshots_per_phase=2,
        nx=20,
        ny=15,
        seed=42,
        output_path=out_traj,
        device="cpu",
    )
    assert out_traj.is_file()
    assert out_traj.with_suffix(".json").is_file()

    # Verify metadata JSON structure and relative paths
    meta = json.loads(out_traj.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == "2.0.0"
    assert meta["mode"] == "trajectory"
    assert "archive_path" in meta
    assert not Path(meta["archive_path"]).is_absolute()

    # Validate raw trajectory dataset
    with np.load(out_traj) as data_traj:
        passed_traj, report_traj = validate_dataset(data_traj, input_path=out_traj)
        assert passed_traj
        assert report_traj["checks"]["concentration_nonnegative"]
        assert report_traj["checks"]["coverage_bounded"]
        assert report_traj["checks"]["thickness_non_decreasing"]
        assert report_traj["checks"]["c_zero_at_purge_ends"]

    # 2. Final state mode generation
    generate(
        mode="final_state",
        num_samples=2,
        batch_size=2,
        num_cycles=2,
        nx=20,
        ny=15,
        seed=42,
        output_path=out_final,
        device="cpu",
    )
    assert out_final.is_file()
    with np.load(out_final) as data_final:
        passed_final, _ = validate_dataset(data_final, input_path=out_final)
        assert passed_final

        # In final_state, concentration is non-zero because it was snapshotted at end of dose
        assert data_final["concentration"].max() > 0.0

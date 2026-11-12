"""Unit tests for Phase 6 evaluation metrics, baselines, diagnostics, and comparisons."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thin_film_pino.evaluation.evaluate import (
    evaluate_baselines,
    evaluate_checkpoint,
    evaluate_single_model,
    load_model_from_checkpoint,
    predict_batched,
)
from thin_film_pino.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_baseline_predictions,
    compute_bounds_violations,
    compute_channel_errors,
    compute_derived_cycle_quantities,
    compute_per_phase_metrics,
)
from thin_film_pino.evaluation.visualize import plot_field_comparison_trajectory
from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.physics.geometry import build_geometry
from thin_film_pino.training.train import compute_normalized_coordinates


@pytest.fixture
def mock_evaluation_data():
    """Build a lightweight synthetic dataset dictionary on a 20x15 grid for testing."""
    ny, nx = 15, 20
    geom = build_geometry(nx=nx, ny=ny)

    n_trajectories = 6
    n_time = 8

    # Synthetic physical fields
    c = np.ones((n_trajectories, n_time, ny, nx), dtype=np.float32) * 0.8
    c[:, :, ~geom.gas_mask] = 0.0

    theta = np.linspace(0.1, 0.9, n_time, dtype=np.float32)[None, :, None, None]
    theta = np.broadcast_to(theta, (n_trajectories, n_time, ny, nx)).copy()
    theta[:, :, ~geom.surface_mask] = 0.0

    h = np.linspace(0.0, 0.05, n_time, dtype=np.float32)[None, :, None, None]
    h = np.broadcast_to(h, (n_trajectories, n_time, ny, nx)).copy()
    h[:, :, ~geom.surface_mask] = 0.0

    # Phases: 1 (dose, 2 steps), 2 (purge1, 2 steps), 3 (rxn, 2 steps), 4 (purge2, 2 steps)
    phase = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    phase_all = np.broadcast_to(phase[None, :], (n_trajectories, n_time))

    cycle = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    cycle_all = np.broadcast_to(cycle[None, :], (n_trajectories, n_time))

    raw_params = np.ones((n_trajectories, 8), dtype=np.float32)
    norm_params = np.zeros((n_trajectories, 5), dtype=np.float32)

    time_arr = np.linspace(0.0, 40.0, n_time, dtype=np.float32)
    time_all = np.broadcast_to(time_arr[None, :], (n_trajectories, n_time))

    return {
        "x": geom.x,
        "y": geom.y,
        "parameters": norm_params,
        "raw_parameters": raw_params,
        "time": time_all,
        "concentration": c,
        "surface_coverage": theta,
        "film_thickness": h,
        "phase": phase_all,
        "cycle": cycle_all,
        "solid_mask": geom.solid_mask,
        "gas_mask": geom.gas_mask,
        "surface_mask": geom.surface_mask,
        "field_mask": geom.field_mask,
        "floor_mask": geom.floor_mask,
        "train_indices": np.array([0, 1, 2]),
        "validation_indices": np.array([3]),
        "test_indices": np.array([4, 5]),
        "field_mean": np.array([0.5, 0.5, 0.025], dtype=np.float32),
        "field_std": np.array([0.4, 0.3, 0.02], dtype=np.float32),
        "time_mean": np.float32(20.0),
        "time_std": np.float32(12.0),
    }


def test_compute_channel_errors():
    """Verify MAE, RMSE, relative L2 spread, and guarded R2 computation."""
    ny, nx = 5, 5
    mask = np.ones((ny, nx), dtype=bool)

    y_true = np.ones((3, 4, ny, nx), dtype=np.float32) * 2.0
    # Add constant prediction offset
    y_pred = y_true + 0.1

    # When y_true is constant, sample variance is 0, so R2 should be dropped
    res_const = compute_channel_errors(y_true, y_pred, mask)
    assert np.isclose(res_const["MAE"], 0.1, atol=1e-6)
    assert np.isclose(res_const["RMSE"], 0.1, atol=1e-6)
    assert res_const["R2"] is None
    assert res_const["R2_status"] == "dropped_due_to_near_zero_variance"
    assert res_const["R2_denominator_SS_tot"] == 0.0
    assert "mean" in res_const["relative_L2"]
    assert "std" in res_const["relative_L2"]

    # When y_true varies, R2 must be computed
    y_var_true = np.arange(3 * 4 * ny * nx, dtype=np.float32).reshape(3, 4, ny, nx)
    y_var_pred = y_var_true + 0.5
    res_var = compute_channel_errors(y_var_true, y_var_pred, mask)
    assert res_var["R2"] is not None
    assert res_var["R2_status"] == "computed"
    assert res_var["R2"] > 0.99
    assert res_var["R2_denominator_SS_tot"] > 0.0


def test_per_phase_metrics(mock_evaluation_data):
    """Verify per-phase MAE and RMSE breakdowns."""
    data = mock_evaluation_data
    c_true = data["concentration"]
    c_pred = c_true + 0.05
    phase = data["phase"]
    mask = data["gas_mask"]

    phase_metrics = compute_per_phase_metrics(c_true, c_pred, mask, phase)
    assert "dose" in phase_metrics
    assert "post_dose_purge" in phase_metrics
    assert "reaction" in phase_metrics
    assert "post_rxn_purge" in phase_metrics

    for p_name, metrics in phase_metrics.items():
        assert np.isclose(metrics["MAE"], 0.05, atol=1e-5)
        assert metrics["snapshots_evaluated"] == 2 * c_true.shape[0]


def test_derived_cycle_quantities(mock_evaluation_data):
    """Verify GPC and step coverage calculations."""
    data = mock_evaluation_data
    h_ref = data["film_thickness"]
    # Introduce small perturbation
    h_pred = h_ref * 1.05

    cycle_quantities = compute_derived_cycle_quantities(
        pred_h=h_pred,
        ref_h=h_ref,
        surface_mask=data["surface_mask"],
        field_mask=data["field_mask"],
        floor_mask=data["floor_mask"],
        cycle=data["cycle"],
    )

    assert "gpc_mae_overall" in cycle_quantities
    assert "step_coverage_mae_overall" in cycle_quantities
    assert len(cycle_quantities["per_cycle"]) > 0
    cycle_1 = cycle_quantities["per_cycle"][0]
    assert cycle_1["cycle"] == 1
    assert cycle_1["gpc_mae"] >= 0.0
    assert cycle_1["step_coverage_mae"] >= 0.0


def test_bounds_violations(mock_evaluation_data):
    """Verify bounds violation diagnostics on unphysical predictions."""
    data = mock_evaluation_data
    c = data["concentration"].copy()
    theta = data["surface_coverage"].copy()

    # Create deliberate violations on verified active cells:
    # C < 0 on gas cells, theta < 0 and theta > 1 on surface cells
    gas_coords = np.argwhere(data["gas_mask"])
    yg, xg = gas_coords[0]
    c[:, :, yg, xg] = -0.5

    surf_coords = np.argwhere(data["surface_mask"])
    y0, x0 = surf_coords[0]
    y1, x1 = surf_coords[1]
    theta[:, :, y0, x0] = -0.2
    theta[:, :, y1, x1] = 1.3

    diag = compute_bounds_violations(
        pred_c=c,
        pred_theta=theta,
        gas_mask=data["gas_mask"],
        surface_mask=data["surface_mask"],
        phase=data["phase"],
    )

    overall = diag["overall"]
    assert overall["concentration_negative_fraction"] > 0.0
    assert np.isclose(overall["concentration_max_negative_violation"], 0.5, atol=1e-5)
    assert overall["coverage_below_zero_fraction"] > 0.0
    assert np.isclose(overall["coverage_max_below_zero_violation"], 0.2, atol=1e-5)
    assert overall["coverage_above_one_fraction"] > 0.0
    assert np.isclose(overall["coverage_max_above_one_violation"], 0.3, atol=1e-5)
    assert "per_phase" in diag


def test_bootstrap_confidence_intervals():
    """Verify that bootstrap confidence intervals produce consistent bounds."""
    y_true = np.random.RandomState(42).randn(10, 5, 10, 10)
    y_pred = y_true + 0.2
    mask = np.ones((10, 10), dtype=bool)

    ci = bootstrap_confidence_intervals(y_true, y_pred, mask, n_bootstraps=200, seed=123)
    assert "mae_mean" in ci
    assert "mae_ci_lower" in ci
    assert "mae_ci_upper" in ci
    assert ci["mae_ci_lower"] <= ci["mae_mean"] <= ci["mae_ci_upper"]


def test_baseline_predictions(mock_evaluation_data):
    """Verify evaluation of train-mean and persistence baseline predictors."""
    data = mock_evaluation_data
    test_idx = data["test_indices"]

    train_mean_pred, persistence_pred = compute_baseline_predictions(data, test_idx)
    assert train_mean_pred.shape == (len(test_idx), 8, 3, 15, 20)
    assert persistence_pred.shape == (len(test_idx), 8, 3, 15, 20)

    # Persistence should hold time step 0 constant across all time
    for t in range(8):
        assert np.allclose(persistence_pred[:, t], persistence_pred[:, 0])

    baselines_eval = evaluate_baselines(data, test_idx, n_bootstraps=50)
    assert "train_mean" in baselines_eval
    assert "persistence" in baselines_eval
    assert "MAE" in baselines_eval["train_mean"]["concentration"]
    assert "MAE" in baselines_eval["persistence"]["film_thickness"]


def test_model_batched_evaluation(mock_evaluation_data):
    """Verify model-agnostic evaluation using a small TemporalFourierOperator in batches."""
    data = mock_evaluation_data
    test_idx = data["test_indices"]

    device = torch.device("cpu")
    model = TemporalFourierOperator(
        parameter_dim=5,
        spatial_channels=3,
        num_phases=5,
        include_cycle=True,
        output_channels=3,
        width=8,
        modes=4,
        num_blocks=1,
    ).to(device)

    # Test predict_batched
    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)
    gas_mask_t = torch.as_tensor(data["gas_mask"], device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    p_t = torch.as_tensor(data["parameters"][test_idx], device=device).float()
    t_t = torch.as_tensor(data["time"][test_idx], device=device).float()
    ph_t = torch.as_tensor(data["phase"][test_idx], device=device).long()
    cyc_t = torch.as_tensor(data["cycle"][test_idx], device=device).float()

    preds = predict_batched(
        model=model,
        parameters=p_t,
        time=t_t,
        spatial_coords=spatial_coords,
        phase=ph_t,
        cycle_frac=cyc_t,
        batch_size=1,  # Mini-batch size 1
    )
    assert preds.shape == (len(test_idx), 8, 3, 15, 20)

    # Evaluate single model
    eval_res = evaluate_single_model(
        model=model,
        data=data,
        split_indices=test_idx,
        batch_size=1,
        n_bootstraps=50,
        device=device,
    )

    assert "concentration" in eval_res
    assert "surface_coverage" in eval_res
    assert "film_thickness" in eval_res
    assert "derived_cycle_quantities" in eval_res
    assert "bounds_violations" in eval_res
    assert eval_res["test_trajectories"] == len(test_idx)


def test_missing_checkpoint_fails(tmp_path):
    """Verify that attempting to load a non-existent checkpoint raises FileNotFoundError immediately."""
    fake_ckpt = tmp_path / "does_not_exist.pt"
    data = {"parameters": np.zeros((2, 5))}
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_model_from_checkpoint(fake_ckpt, data, torch.device("cpu"))

    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint("data/nonexistent.npz", fake_ckpt)


def test_qualitative_plotting(tmp_path, mock_evaluation_data):
    """Verify qualitative plot generation executes and writes PNG files."""
    data = mock_evaluation_data
    ref_f = data["concentration"][0]
    pred_f = ref_f * 0.95
    out_png = tmp_path / "test_qualitative.png"

    plot_field_comparison_trajectory(
        ref_field=ref_f,
        pred_field=pred_f,
        mask=data["gas_mask"],
        solid_mask=data["solid_mask"],
        x_coords=data["x"],
        y_coords=data["y"],
        time_values=data["time"][0],
        time_indices=[0, 2, 4],
        field_name="Concentration",
        field_unit="mol/m^3",
        output_path=out_png,
        phase_labels=["dose", "purge1", "rxn"],
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0

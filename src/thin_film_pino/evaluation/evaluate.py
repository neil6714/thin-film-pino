"""Model-agnostic batched evaluation suite for thin-film neural operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from thin_film_pino.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_baseline_predictions,
    compute_bounds_violations,
    compute_channel_errors,
    compute_derived_cycle_quantities,
    compute_per_phase_metrics,
)
from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.physics.geometry import build_geometry
from thin_film_pino.training.pino import compute_physics_loss
from thin_film_pino.training.train import compute_normalized_coordinates, load_dataset_in_memory


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Batched model-agnostic evaluation with physical metrics, baselines, and diagnostics."
    )
    parser.add_argument(
        "--input", required=True, help="Path to prepared transient ALD dataset (.npz)."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/fno/best_model.pt",
        help="Path to trained operator model checkpoint (.pt).",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional list of multiple seed checkpoints to evaluate and aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/evaluation",
        help="Destination directory for evaluation metrics.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "ood", "validation", "train"],
        help="Trajectory split partition to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of trajectories to process per forward batch.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=500,
        help="Number of bootstrap resamples across trajectories.",
    )
    return parser.parse_args(args)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    dataset_metadata: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate and load a neural operator from a checkpoint path.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to PyTorch checkpoint file.
    dataset_metadata : Dict[str, Any]
        Metadata and shape information from dataset.
    device : torch.device
        Compute device.

    Returns
    -------
    torch.nn.Module
        Loaded model in eval mode.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    config_path = ckpt_path.parent / "best_model_config.json"
    if not config_path.exists():
        config_path = ckpt_path.parent / "checkpoint_config.json"

    cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}

    arch_cfg = cfg.get("architecture", {})
    width = arch_cfg.get("width", 32)
    modes = arch_cfg.get("modes", 12)
    num_blocks = arch_cfg.get("num_blocks", 4)
    param_dim = arch_cfg.get(
        "parameter_dim",
        int(dataset_metadata["parameters"].shape[-1]),
    )
    num_phases = arch_cfg.get("num_phases", 5)
    include_cycle = arch_cfg.get("include_cycle", True)

    model = TemporalFourierOperator(
        parameter_dim=param_dim,
        spatial_channels=3,
        num_phases=num_phases,
        include_cycle=include_cycle,
        output_channels=3,
        width=width,
        modes=modes,
        num_blocks=num_blocks,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_batched(
    model: torch.nn.Module,
    parameters: torch.Tensor,
    time: torch.Tensor,
    spatial_coords: torch.Tensor,
    phase: Optional[torch.Tensor] = None,
    cycle_frac: Optional[torch.Tensor] = None,
    batch_size: int = 4,
) -> torch.Tensor:
    """Compute model predictions across trajectories in manageable batches.

    Parameters
    ----------
    model : torch.nn.Module
        Neural operator model.
    parameters : torch.Tensor
        Normalized parameters (N, P).
    time : torch.Tensor
        Normalized time coordinates (N, T).
    spatial_coords : torch.Tensor
        Spatial grid coordinates (3, H, W).
    phase : Optional[torch.Tensor], default=None
        Phase indicator array (N, T).
    cycle_frac : Optional[torch.Tensor], default=None
        Cycle fraction array (N, T).
    batch_size : int, default=4
        Number of trajectories per mini-batch.

    Returns
    -------
    torch.Tensor
        Batched predictions of shape (N, T, 3, H, W).
    """
    n_trajectories = parameters.shape[0]
    preds: List[torch.Tensor] = []

    with torch.no_grad():
        for start_idx in range(0, n_trajectories, batch_size):
            end_idx = min(start_idx + batch_size, n_trajectories)
            p_batch = parameters[start_idx:end_idx]
            t_batch = time[start_idx:end_idx]
            ph_batch = phase[start_idx:end_idx] if phase is not None else None
            cyc_batch = cycle_frac[start_idx:end_idx] if cycle_frac is not None else None

            out = model(
                p_batch,
                t_batch,
                spatial_coords,
                phase=ph_batch,
                cycle_fraction=cyc_batch,
            )
            preds.append(out)

    return torch.cat(preds, dim=0)


def evaluate_single_model(
    model: torch.nn.Module,
    data: Dict[str, Any],
    split_indices: np.ndarray,
    batch_size: int = 4,
    n_bootstraps: int = 500,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run model-agnostic evaluation over designated trajectory split.

    Parameters
    ----------
    model : torch.nn.Module
        Operator model to evaluate.
    data : Dict[str, Any]
        In-memory dataset dictionary.
    split_indices : np.ndarray
        Indices of trajectories to evaluate.
    batch_size : int, default=4
        Trajectory batch size for forward passes.
    n_bootstraps : int, default=500
        Number of bootstrap samples for uncertainty estimation.
    device : Optional[torch.device], default=None
        Compute device.

    Returns
    -------
    Dict[str, Any]
        Complete evaluation report with metrics, diagnostics, and bootstrap CIs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)
    gas_mask_np = data["gas_mask"].astype(bool)
    surface_mask_np = data["surface_mask"].astype(bool)

    # Reconstruct floor and field masks if not explicitly in dataset
    if "floor_mask" in data and "field_mask" in data:
        floor_mask_np = data["floor_mask"].astype(bool)
        field_mask_np = data["field_mask"].astype(bool)
    else:
        ny, nx = gas_mask_np.shape
        geom = build_geometry(nx=nx, ny=ny)
        floor_mask_np = geom.floor_mask.astype(bool)
        field_mask_np = geom.field_mask.astype(bool)

    gas_mask_t = torch.as_tensor(gas_mask_np, device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    parameters_t = torch.as_tensor(data["parameters"][split_indices], device=device).float()
    time_t = torch.as_tensor(data["time"][split_indices], device=device).float()

    phase_t = (
        torch.as_tensor(data["phase"][split_indices], device=device).long()
        if "phase" in data
        else None
    )

    if "cycle" in data:
        max_c = float(np.max(data["cycle"])) or 10.0
        cycle_frac_t = torch.as_tensor(
            data["cycle"][split_indices].astype(np.float32) / max_c, device=device
        )
    else:
        cycle_frac_t = None

    # Predict in batches over test trajectories
    pred_norm = predict_batched(
        model=model,
        parameters=parameters_t,
        time=time_t,
        spatial_coords=spatial_coords,
        phase=phase_t,
        cycle_frac=cycle_frac_t,
        batch_size=batch_size,
    )

    # Denormalize predictions and ground truth to physical units
    f_mean = data["field_mean"]
    f_std = data["field_std"]
    pred_np = pred_norm.detach().cpu().numpy()

    # Targets
    c_true_norm = data["concentration"][split_indices]
    th_true_norm = data["surface_coverage"][split_indices]
    h_true_norm = data["film_thickness"][split_indices]

    c_true_phys = c_true_norm * f_std[0] + f_mean[0]
    c_true_phys[:, :, ~gas_mask_np] = 0.0

    th_true_phys = th_true_norm * f_std[1] + f_mean[1]
    th_true_phys[:, :, ~surface_mask_np] = 0.0

    h_true_phys = h_true_norm * f_std[2] + f_mean[2]
    h_true_phys[:, :, ~surface_mask_np] = 0.0

    # Predictions in physical units
    c_pred_phys = pred_np[:, :, 0] * f_std[0] + f_mean[0]
    c_pred_phys[:, :, ~gas_mask_np] = 0.0

    th_pred_phys = pred_np[:, :, 1] * f_std[1] + f_mean[1]
    th_pred_phys[:, :, ~surface_mask_np] = 0.0

    h_pred_phys = pred_np[:, :, 2] * f_std[2] + f_mean[2]
    h_pred_phys[:, :, ~surface_mask_np] = 0.0

    # 1. Physical metrics per channel
    c_metrics = compute_channel_errors(c_true_phys, c_pred_phys, gas_mask_np)
    th_metrics = compute_channel_errors(th_true_phys, th_pred_phys, surface_mask_np)
    h_metrics = compute_channel_errors(h_true_phys, h_pred_phys, surface_mask_np)

    # Bootstrap confidence intervals on channel MAE
    c_boot = bootstrap_confidence_intervals(
        c_true_phys, c_pred_phys, gas_mask_np, n_bootstraps=n_bootstraps
    )
    th_boot = bootstrap_confidence_intervals(
        th_true_phys, th_pred_phys, surface_mask_np, n_bootstraps=n_bootstraps
    )
    h_boot = bootstrap_confidence_intervals(
        h_true_phys, h_pred_phys, surface_mask_np, n_bootstraps=n_bootstraps
    )

    c_metrics["bootstrap_mae_ci"] = c_boot
    th_metrics["bootstrap_mae_ci"] = th_boot
    h_metrics["bootstrap_mae_ci"] = h_boot

    # 2. Per-phase breakdown
    phase_arr = data["phase"][split_indices] if "phase" in data else None
    if phase_arr is not None:
        c_phase = compute_per_phase_metrics(c_true_phys, c_pred_phys, gas_mask_np, phase_arr)
        th_phase = compute_per_phase_metrics(th_true_phys, th_pred_phys, surface_mask_np, phase_arr)
        h_phase = compute_per_phase_metrics(h_true_phys, h_pred_phys, surface_mask_np, phase_arr)
    else:
        c_phase, th_phase, h_phase = {}, {}, {}

    # 3. Derived quantities: GPC and step coverage per cycle
    cycle_arr = data["cycle"][split_indices] if "cycle" in data else None
    cycle_metrics = compute_derived_cycle_quantities(
        pred_h=h_pred_phys,
        ref_h=h_true_phys,
        surface_mask=surface_mask_np,
        field_mask=field_mask_np,
        floor_mask=floor_mask_np,
        cycle=cycle_arr,
    )

    # 4. Bounds violations
    bounds_diag = compute_bounds_violations(
        pred_c=c_pred_phys,
        pred_theta=th_pred_phys,
        gas_mask=gas_mask_np,
        surface_mask=surface_mask_np,
        phase=phase_arr,
    )

    # 5. Physics residual diagnostics via autograd
    p_phys = data["raw_parameters"][split_indices]
    dx = float(data["x"][1] - data["x"][0]) if len(data["x"]) > 1 else 1.0 / 20.0
    dy = float(data["y"][1] - data["y"][0]) if len(data["y"]) > 1 else 1.0 / 15.0

    stats_tuple = (
        torch.as_tensor(data["field_mean"], device=device),
        torch.as_tensor(data["field_std"], device=device),
        torch.as_tensor(data["time_mean"], device=device),
        torch.as_tensor(data["time_std"], device=device),
    )

    with torch.enable_grad():
        try:
            _, pde_components = compute_physics_loss(
                model=model,
                parameters_norm=parameters_t[: min(4, len(split_indices))],
                parameters_phys=torch.as_tensor(
                    p_phys[: min(4, len(split_indices))], device=device
                ).float(),
                spatial_coords=spatial_coords,
                gas_mask=torch.as_tensor(gas_mask_np, device=device),
                surface_mask=torch.as_tensor(surface_mask_np, device=device),
                stats=stats_tuple,
                dx=dx,
                dy=dy,
                num_collocation_per_phase=2,
                num_cycles=10,
            )
            physics_residuals = {
                "transport_residual_norm": float(np.sqrt(pde_components["transport_mse"])),
                "coverage_residual_norm": float(np.sqrt(pde_components["coverage_mse"])),
                "growth_residual_norm": float(np.sqrt(pde_components["growth_mse"])),
                "transport_mse": pde_components["transport_mse"],
                "coverage_mse": pde_components["coverage_mse"],
                "growth_mse": pde_components["growth_mse"],
            }
        except Exception as exc:
            physics_residuals = {"status": f"diagnostic_unavailable: {exc}"}

    return {
        "concentration": c_metrics,
        "surface_coverage": th_metrics,
        "film_thickness": h_metrics,
        "per_phase_breakdown": {
            "concentration": c_phase,
            "surface_coverage": th_phase,
            "film_thickness": h_phase,
        },
        "derived_cycle_quantities": cycle_metrics,
        "bounds_violations": bounds_diag,
        "physics_residuals": physics_residuals,
        "test_trajectories": int(len(split_indices)),
    }


def evaluate_baselines(
    data: Dict[str, Any],
    split_indices: np.ndarray,
    n_bootstraps: int = 500,
) -> Dict[str, Any]:
    """Evaluate train-mean and persistence baseline predictors on identical test data.

    Parameters
    ----------
    data : Dict[str, Any]
        In-memory dataset dictionary.
    split_indices : np.ndarray
        Split indices to evaluate.
    n_bootstraps : int, default=500
        Number of bootstrap samples.

    Returns
    -------
    Dict[str, Any]
        Baseline metrics for train-mean and persistence predictors.
    """
    f_mean = data["field_mean"]
    f_std = data["field_std"]
    gas_mask = data["gas_mask"].astype(bool)
    surface_mask = data["surface_mask"].astype(bool)

    c_true = data["concentration"][split_indices] * f_std[0] + f_mean[0]
    c_true[:, :, ~gas_mask] = 0.0

    th_true = data["surface_coverage"][split_indices] * f_std[1] + f_mean[1]
    th_true[:, :, ~surface_mask] = 0.0

    h_true = data["film_thickness"][split_indices] * f_std[2] + f_mean[2]
    h_true[:, :, ~surface_mask] = 0.0

    train_mean_pred, persistence_pred = compute_baseline_predictions(data, split_indices)

    # 1. Train-mean metrics
    tm_c = compute_channel_errors(c_true, train_mean_pred[:, :, 0], gas_mask)
    tm_th = compute_channel_errors(th_true, train_mean_pred[:, :, 1], surface_mask)
    tm_h = compute_channel_errors(h_true, train_mean_pred[:, :, 2], surface_mask)

    tm_c["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        c_true, train_mean_pred[:, :, 0], gas_mask, n_bootstraps=n_bootstraps
    )
    tm_th["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        th_true, train_mean_pred[:, :, 1], surface_mask, n_bootstraps=n_bootstraps
    )
    tm_h["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        h_true, train_mean_pred[:, :, 2], surface_mask, n_bootstraps=n_bootstraps
    )

    # 2. Persistence metrics
    pers_c = compute_channel_errors(c_true, persistence_pred[:, :, 0], gas_mask)
    pers_th = compute_channel_errors(th_true, persistence_pred[:, :, 1], surface_mask)
    pers_h = compute_channel_errors(h_true, persistence_pred[:, :, 2], surface_mask)

    pers_c["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        c_true, persistence_pred[:, :, 0], gas_mask, n_bootstraps=n_bootstraps
    )
    pers_th["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        th_true, persistence_pred[:, :, 1], surface_mask, n_bootstraps=n_bootstraps
    )
    pers_h["bootstrap_mae_ci"] = bootstrap_confidence_intervals(
        h_true, persistence_pred[:, :, 2], surface_mask, n_bootstraps=n_bootstraps
    )

    return {
        "train_mean": {
            "concentration": tm_c,
            "surface_coverage": tm_th,
            "film_thickness": tm_h,
        },
        "persistence": {
            "concentration": pers_c,
            "surface_coverage": pers_th,
            "film_thickness": pers_h,
        },
    }


def format_evaluation_summary(
    model_name: str,
    metrics: Dict[str, Any],
    baselines: Dict[str, Any],
    test_size: int,
) -> str:
    """Format evaluation metrics and baseline comparisons as a human-readable table."""
    lines = [
        "=" * 82,
        f"Evaluation Summary: {model_name}",
        f"Evaluation Partition Size: N = {test_size} trajectories",
        "Notice: When test set size is small, differences within bootstrap intervals are not statistically significant.",
        "-" * 82,
        f"{'Model / Baseline':<20} | {'Field':<18} | {'MAE (Physical)':<18} | {'95% Bootstrap CI':<18}",
        "-" * 82,
    ]

    models_to_print = [
        (model_name, metrics),
        ("Train-Mean Baseline", baselines["train_mean"]),
        ("Persistence Baseline", baselines["persistence"]),
    ]

    for m_label, m_dict in models_to_print:
        for ch, ch_name in [
            ("concentration", "Concentration"),
            ("surface_coverage", "Coverage"),
            ("film_thickness", "Thickness"),
        ]:
            ch_data = m_dict.get(ch, {})
            mae_val = ch_data.get("MAE", float("nan"))
            ci_data = ch_data.get("bootstrap_mae_ci", {})
            ci_str = (
                f"[{ci_data.get('mae_ci_lower', 0.0):.3e}, {ci_data.get('mae_ci_upper', 0.0):.3e}]"
                if ci_data
                else "N/A"
            )
            lines.append(f"{m_label:<20} | {ch_name:<18} | {mae_val:<18.4e} | {ci_str:<18}")

    lines.append("-" * 82)
    # Add GPC and Step Coverage Summary if present
    if "derived_cycle_quantities" in metrics:
        dc = metrics["derived_cycle_quantities"]
        lines.append(
            f"Derived Metrics: GPC MAE = {dc.get('gpc_mae_overall', 0.0):.4e} | "
            f"Step Coverage MAE = {dc.get('step_coverage_mae_overall', 0.0):.4e}"
        )

    # Add Physics Residuals and Bounds Summary if present
    if "physics_residuals" in metrics and "transport_residual_norm" in metrics["physics_residuals"]:
        pr = metrics["physics_residuals"]
        lines.append(
            f"Physics Residuals: Transport = {pr['transport_residual_norm']:.4e} | "
            f"Coverage = {pr['coverage_residual_norm']:.4e} | Growth = {pr['growth_residual_norm']:.4e}"
        )

    if "bounds_violations" in metrics:
        bv = metrics["bounds_violations"].get("overall", {})
        lines.append(
            f"Bounds Violations: C < 0 rate = {bv.get('concentration_negative_fraction', 0.0):.2%} | "
            f"theta outside [0, 1] rate = {bv.get('coverage_below_zero_fraction', 0.0) + bv.get('coverage_above_one_fraction', 0.0):.2%}"
        )

    lines.append("=" * 82)
    return "\n".join(lines)


def evaluate_checkpoint(
    input_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path] = "runs/evaluation",
    split: str = "test",
    batch_size: int = 4,
    n_bootstraps: int = 500,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint against held-out trajectory data.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to prepared dataset (.npz).
    checkpoint_path : Union[str, Path]
        Path to checkpoint file (.pt).
    output_dir : Union[str, Path], default='runs/evaluation'
        Destination folder for evaluation artifacts.
    split : str, default='test'
        Split to evaluate: 'test', 'ood', 'validation', or 'train'.
    batch_size : int, default=4
        Trajectory batch size.
    n_bootstraps : int, default=500
        Number of bootstrap iterations.
    device : Optional[torch.device], default=None
        Device for tensor computations.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics and baseline comparisons.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = Path(input_path).resolve()
    data = load_dataset_in_memory(input_path)

    split_key = f"{split}_indices"
    if split_key not in data:
        raise KeyError(
            f"Requested split '{split}' not found in dataset. Available keys: {list(data.keys())}"
        )
    split_idx = data[split_key]

    model = load_model_from_checkpoint(checkpoint_path, data, device)
    metrics = evaluate_single_model(
        model=model,
        data=data,
        split_indices=split_idx,
        batch_size=batch_size,
        n_bootstraps=n_bootstraps,
        device=device,
    )
    baselines = evaluate_baselines(data, split_idx, n_bootstraps=n_bootstraps)

    test_size = int(len(split_idx))
    summary_text = format_evaluation_summary(
        model_name=Path(checkpoint_path).stem,
        metrics=metrics,
        baselines=baselines,
        test_size=test_size,
    )
    print(summary_text)

    full_output = {
        "model": metrics,
        "baselines": baselines,
        "metadata": {
            "input": str(input_path),
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "split": split,
            "test_trajectories": test_size,
            "device": str(device),
        },
    }

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    (out_path / "summary.txt").write_text(summary_text, encoding="utf-8")

    return full_output


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for evaluation."""
    parsed = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if parsed.checkpoints:
        # Multi-seed evaluation
        all_seed_metrics = []
        for ckpt in parsed.checkpoints:
            print(f"Evaluating checkpoint: {ckpt}")
            res = evaluate_checkpoint(
                input_path=parsed.input,
                checkpoint_path=ckpt,
                output_dir=Path(parsed.output_dir) / Path(ckpt).parent.name,
                split=parsed.split,
                batch_size=parsed.batch_size,
                n_bootstraps=parsed.n_bootstraps,
                device=device,
            )
            all_seed_metrics.append(res)

        print(
            f"Multi-seed evaluation over {len(all_seed_metrics)} checkpoints completed. "
            f"Reports stored under {parsed.output_dir}."
        )
    else:
        evaluate_checkpoint(
            input_path=parsed.input,
            checkpoint_path=parsed.checkpoint,
            output_dir=parsed.output_dir,
            split=parsed.split,
            batch_size=parsed.batch_size,
            n_bootstraps=parsed.n_bootstraps,
            device=device,
        )


if __name__ == "__main__":
    main()

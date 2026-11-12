"""Evaluation module for trained temporal Fourier neural operator checkpoints on held-out test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.training.train import compute_normalized_coordinates, load_dataset_in_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal Fourier operator checkpoints with physical and normalized metrics."
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
        "--output-dir",
        default="runs/evaluation",
        help="Destination directory for evaluation metrics.",
    )
    return parser.parse_args()


def evaluate_checkpoint(
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path = "runs/evaluation",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint against held-out test trajectories."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = Path(input_path).resolve()
    data = load_dataset_in_memory(input_path)

    checkpoint_path = Path(checkpoint_path).resolve()
    # Load weights safely with weights_only=True
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Load configuration JSON if present alongside checkpoint
    config_path = checkpoint_path.parent / "best_model_config.json"
    if not config_path.exists():
        config_path = checkpoint_path.parent / "checkpoint_config.json"

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
    param_dim = arch_cfg.get("parameter_dim", data["parameters"].shape[-1])
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

    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)
    gas_mask_t = torch.as_tensor(~data["solid_mask"], device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    test_idx = data["test_indices"]
    parameters = torch.as_tensor(data["parameters"][test_idx], device=device).float()
    time = torch.as_tensor(data["time"][test_idx], device=device).float()

    phase = (
        torch.as_tensor(data["phase"][test_idx], device=device).long() if "phase" in data else None
    )

    if "cycle" in data:
        max_c = float(np.max(data["cycle"])) or 10.0
        cycle_frac = torch.as_tensor(
            data["cycle"][test_idx].astype(np.float32) / max_c, device=device
        )
    else:
        cycle_frac = None

    target_fields = np.stack(
        [
            data["concentration"][test_idx],
            data["surface_coverage"][test_idx],
            data["film_thickness"][test_idx],
        ],
        axis=2,
    )
    target = torch.as_tensor(target_fields, device=device).float()

    surface_mask_t = torch.as_tensor(data["surface_mask"], device=device).float().unsqueeze(0)
    channel_masks = torch.cat([gas_mask_t, surface_mask_t, surface_mask_t], dim=0)
    mask = channel_masks[None, None]

    with torch.no_grad():
        prediction = model(
            parameters,
            time,
            spatial_coords,
            phase=phase,
            cycle_fraction=cycle_frac,
        )

    diff = prediction - target
    valid = mask.expand_as(diff)
    denom = valid.sum().clamp_min(1.0)
    scale = torch.as_tensor(data["field_std"], device=device)[None, None, :, None, None]
    physical_diff = diff * scale

    metrics: Dict[str, Any] = {}
    channel_names = ("concentration", "surface_coverage", "film_thickness")

    for c, name in enumerate(channel_names):
        v = valid[:, :, c]
        e = physical_diff[:, :, c]
        n = v.sum().clamp_min(1.0)

        y_true_phys = (target[:, :, c] * scale[:, :, c]).masked_select(v.bool())
        e_phys = (e * v).masked_select(v.bool())

        mae_phys = float(e_phys.abs().sum() / n)
        rmse_phys = float(torch.sqrt(e_phys.square().sum() / n))
        mae_norm = float((diff[:, :, c].abs() * v).sum() / n)
        rmse_norm = float(torch.sqrt((diff[:, :, c].square() * v).sum() / n))

        ss_tot = (y_true_phys - y_true_phys.mean()).square().sum().clamp_min(1.0e-12)
        r2_phys = float(1.0 - (e_phys.square().sum() / ss_tot))

        metrics[name] = {
            "MAE_physical": mae_phys,
            "RMSE_physical": rmse_phys,
            "MAE_normalized": mae_norm,
            "RMSE_normalized": rmse_norm,
            "R2_physical": r2_phys,
        }

    metrics["overall"] = {
        "MSE_normalized": float((diff.square() * valid).sum() / denom),
        "MAE_normalized": float((diff.abs() * valid).sum() / denom),
        "test_trajectories": int(len(test_idx)),
        "grid": [int(data["concentration"].shape[-2]), int(data["concentration"].shape[-1])],
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    metrics = evaluate_checkpoint(args.input, args.checkpoint, args.output_dir, device)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {Path(args.output_dir).resolve() / 'metrics.json'}")


if __name__ == "__main__":
    main()

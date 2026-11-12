"""Evaluation module for trained temporal Fourier neural operator checkpoints on held-out test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from thin_film_pino.models.fno import TemporalFourierOperator


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
    data = np.load(input_path)

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("args", {})

    width = config.get("width", 32)
    modes = config.get("modes", 12)
    param_dim = data["parameters"].shape[-1]
    model = TemporalFourierOperator(parameter_dim=param_dim, width=width, modes=modes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    coords = (
        torch.stack(
            torch.meshgrid(
                torch.as_tensor(data["y"], device=device),
                torch.as_tensor(data["x"], device=device),
                indexing="ij",
            ),
            dim=0,
        ).float()
        * 2.0
        - 1.0
    )

    test_idx = data["test_indices"]
    parameters = torch.as_tensor(data["parameters"][test_idx], device=device)
    time = torch.as_tensor(data["time"][test_idx], device=device)

    target_fields = np.stack(
        [
            data["concentration"][test_idx],
            data["surface_coverage"][test_idx],
            data["film_thickness"][test_idx],
        ],
        axis=2,
    )
    target = torch.as_tensor(target_fields, device=device)

    mask = torch.as_tensor(
        np.stack([~data["solid_mask"], data["surface_mask"], data["surface_mask"]]),
        device=device,
    ).float()[None, None]

    with torch.no_grad():
        prediction = model(parameters, time, coords)

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

        ss_tot = (y_true_phys - y_true_phys.mean()).square().sum().clamp_min(1e-12)
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

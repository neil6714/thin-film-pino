"""Generate qualitative comparison figures for a selected test trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from thin_film_pino.evaluation.evaluate import load_model_from_checkpoint
from thin_film_pino.evaluation.visualize import generate_qualitative_figures
from thin_film_pino.training.train import compute_normalized_coordinates, load_dataset_in_memory


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse qualitative plotting arguments."""
    parser = argparse.ArgumentParser(
        description="Generate qualitative predicted versus reference field plots on a test trajectory."
    )
    parser.add_argument(
        "--input", required=True, help="Path to prepared transient ALD dataset (.npz)."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained operator model checkpoint (.pt)."
    )
    parser.add_argument(
        "--output-dir",
        default="runs/evaluation/qualitative",
        help="Destination directory for qualitative figures.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Index of test trajectory to visualize (0 to N_test - 1).",
    )
    return parser.parse_args(args)


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for qualitative plotting."""
    parsed = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_path = Path(parsed.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    ckpt_path = Path(parsed.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data = load_dataset_in_memory(input_path)
    test_idx = data["test_indices"]
    if parsed.trajectory_index >= len(test_idx):
        raise IndexError(
            f"Trajectory index {parsed.trajectory_index} out of range (test set size: {len(test_idx)})."
        )

    raw_traj_idx = test_idx[parsed.trajectory_index]
    model = load_model_from_checkpoint(ckpt_path, data, device)

    # Prepare inputs for single trajectory
    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)
    gas_mask_t = torch.as_tensor(data["gas_mask"], device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    p_tensor = torch.as_tensor(
        data["parameters"][raw_traj_idx : raw_traj_idx + 1], device=device
    ).float()
    t_tensor = torch.as_tensor(data["time"][raw_traj_idx : raw_traj_idx + 1], device=device).float()

    ph_tensor = (
        torch.as_tensor(data["phase"][raw_traj_idx : raw_traj_idx + 1], device=device).long()
        if "phase" in data
        else None
    )

    if "cycle" in data:
        max_c = float(data["cycle"].max()) or 10.0
        cyc_tensor = torch.as_tensor(
            data["cycle"][raw_traj_idx : raw_traj_idx + 1].astype(float) / max_c, device=device
        ).float()
    else:
        cyc_tensor = None

    with torch.no_grad():
        pred_norm = model(
            p_tensor,
            t_tensor,
            spatial_coords,
            phase=ph_tensor,
            cycle_fraction=cyc_tensor,
        )

    # Denormalize to physical units
    f_mean = data["field_mean"]
    f_std = data["field_std"]
    pred_np = pred_norm[0].detach().cpu().numpy()  # (T, 3, H, W)

    c_pred = pred_np[:, 0] * f_std[0] + f_mean[0]
    th_pred = pred_np[:, 1] * f_std[1] + f_mean[1]
    h_pred = pred_np[:, 2] * f_std[2] + f_mean[2]

    gas_mask = data["gas_mask"].astype(bool)
    surf_mask = data["surface_mask"].astype(bool)

    c_pred[:, ~gas_mask] = 0.0
    th_pred[:, ~surf_mask] = 0.0
    h_pred[:, ~surf_mask] = 0.0

    pred_physical = np.stack([c_pred, th_pred, h_pred], axis=1)

    out_dir = Path(parsed.output_dir).resolve()
    saved = generate_qualitative_figures(
        data=data,
        pred_physical=pred_physical,
        trajectory_index=parsed.trajectory_index,
        output_dir=out_dir,
    )

    print(f"Qualitative figures generated for test trajectory {parsed.trajectory_index}:")
    for s in saved:
        print(f"  - {s}")


if __name__ == "__main__":
    main()

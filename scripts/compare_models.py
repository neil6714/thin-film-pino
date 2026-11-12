"""Comparative benchmark evaluating trained neural operator models on held-out trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.training.train import compute_normalized_coordinates, load_dataset_in_memory


def evaluate(
    data: Mapping[str, Any],
    checkpoint_path: Union[str, Path],
    width: int = 32,
    modes: int = 12,
) -> Dict[str, float]:
    """Compute physical MAE metrics across concentration, coverage, and thickness channels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path).resolve()

    # Load model weights safely with weights_only=True
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    config_path = checkpoint_path.parent / "best_model_config.json"
    cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}

    arch_cfg = cfg.get("architecture", {})
    model_width = arch_cfg.get("width", width)
    model_modes = arch_cfg.get("modes", modes)
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
        width=model_width,
        modes=model_modes,
        num_blocks=num_blocks,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)
    gas_mask_t = torch.as_tensor(~data["solid_mask"], device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    test_idx = data["test_indices"]
    params = torch.as_tensor(data["parameters"][test_idx], device=device).float()
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

    with torch.no_grad():
        pred = model(
            params,
            time,
            spatial_coords,
            phase=phase,
            cycle_fraction=cycle_frac,
        )

    target = torch.as_tensor(
        np.stack(
            [
                data["concentration"][test_idx],
                data["surface_coverage"][test_idx],
                data["film_thickness"][test_idx],
            ],
            axis=2,
        ),
        device=device,
    ).float()

    surface_mask_t = torch.as_tensor(data["surface_mask"], device=device).float().unsqueeze(0)
    channel_masks = torch.cat([gas_mask_t, surface_mask_t, surface_mask_t], dim=0)
    mask = channel_masks[None, None]
    scale = torch.as_tensor(data["field_std"], device=device)[None, None, :, None, None]

    err = (pred - target).abs() * scale
    valid = mask.expand_as(err)
    vals = (err * valid).sum(dim=(0, 1, 3, 4)) / valid.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)

    return {
        "concentration_mae": float(vals[0]),
        "coverage_mae": float(vals[1]),
        "thickness_mae": float(vals[2]),
        "overall_mae": float(vals.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and compare trained neural operators against held-out ALD test trajectories."
    )
    parser.add_argument(
        "--input", required=True, help="Path to prepared transient ALD dataset (.npz)."
    )
    parser.add_argument(
        "--output", default="runs/comparison", help="Output directory for comparison metrics."
    )
    parser.add_argument(
        "--operator",
        default="runs/fno/best_model.pt",
        help="Path to baseline temporal operator checkpoint.",
    )
    parser.add_argument(
        "--pino",
        default="runs/pino/best_model.pt",
        help="Path to physics-informed temporal PINO checkpoint.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset_in_memory(Path(args.input).resolve())

    rows: Dict[str, Dict[str, Any]] = {}
    models_to_evaluate = [
        ("temporal_operator", args.operator),
        ("temporal_pino", args.pino),
    ]

    for name, path_str in models_to_evaluate:
        ckpt_path = Path(path_str).resolve()
        if ckpt_path.exists():
            print(f"Evaluating {name} from {ckpt_path}...")
            rows[name] = evaluate(data, ckpt_path)

    for name, path_str in models_to_evaluate:
        ckpt_dir = Path(path_str).resolve().parent
        metrics_file = ckpt_dir / "metrics.json"
        if metrics_file.exists() and name in rows:
            try:
                m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                reported_loss = m_data.get("test_loss", m_data.get("best_validation_loss"))
                rows[name]["reported_loss"] = reported_loss
            except (OSError, json.JSONDecodeError):
                pass

    comparison_json = out_dir / "comparison_metrics.json"
    comparison_json.write_text(json.dumps(rows, indent=2))

    if rows:
        names = list(rows.keys())
        x = np.arange(len(names))
        w = 0.25

        plt.figure(figsize=(9, 5))
        for j, key in enumerate(("concentration_mae", "coverage_mae", "thickness_mae")):
            plt.bar(
                x + (j - 1) * w,
                [rows[n][key] for n in names],
                w,
                label=key.replace("_", " ").title(),
            )
        plt.xticks(x, [n.replace("_", " ").title() for n in names])
        plt.ylabel("Physical Mean Absolute Error (MAE)")
        plt.title("Comparative Performance on Held-Out Test Trajectories")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "physical_mae_comparison.png", dpi=200)
        plt.close()

    print(json.dumps(rows, indent=2))
    print(f"Comparison metrics and figure written to: {out_dir}")


if __name__ == "__main__":
    main()

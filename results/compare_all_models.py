"""Comparative benchmark evaluating trained neural operator models on held-out trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_temporal_operator import TemporalFourierOperator


def evaluate(
    data: Mapping[str, Any], checkpoint_path: Union[str, Path], width: int = 32, modes: int = 12
) -> Dict[str, float]:
    """Compute physical MAE metrics across concentration, coverage, and thickness channels.

    Parameters
    ----------
    data : Mapping[str, Any]
        Loaded NPZ archive containing test split trajectories and normalization statistics.
    checkpoint_path : str or Path
        Path to the saved PyTorch model checkpoint.
    width : int, default=32
        Hidden channel width if not specified in checkpoint config.
    modes : int, default=12
        Fourier Fourier modes if not specified in checkpoint config.

    Returns
    -------
    Dict[str, float]
        Dictionary of channel-wise and aggregate physical MAE values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})

    model = TemporalFourierOperator(
        width=cfg.get("width", width),
        modes=cfg.get("modes", modes),
    ).to(device)
    model.load_state_dict(ckpt["model"])
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

    idx = data["test_indices"]
    p = torch.as_tensor(data["parameters"][idx], device=device)
    t = torch.as_tensor(data["time"][idx], device=device)

    target_fields = np.stack(
        [
            data["concentration"][idx],
            data["surface_coverage"][idx],
            data["film_thickness"][idx],
        ],
        axis=2,
    )
    target = torch.as_tensor(target_fields, device=device)

    with torch.no_grad():
        pred = model(p, t, coords)

    mask = (
        torch.as_tensor(
            np.stack([~data["solid_mask"], data["surface_mask"], data["surface_mask"]]),
            device=device,
        )
        .float()[None, None]
    )
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
    parser.add_argument("--input", required=True, help="Path to prepared transient ALD dataset (.npz).")
    parser.add_argument("--output", default="results/model_comparison", help="Output directory for comparison metrics.")
    parser.add_argument(
        "--operator",
        default="results/stage4c_temporal_operator/best_model.pt",
        help="Path to baseline temporal operator checkpoint.",
    )
    parser.add_argument(
        "--pino",
        default="results/stage4d_temporal_pino/best_model.pt",
        help="Path to physics-informed temporal PINO checkpoint.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)

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
                reported_loss = m_data.get("test_loss", m_data.get("best_validation_data_loss"))
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

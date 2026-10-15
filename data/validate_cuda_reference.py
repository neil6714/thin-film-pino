"""Validation script comparing matched trajectories from reference and accelerated simulators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generate_transient_dataset import sample_parameters
from physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from physics.simulator import ThinFilmDepositionSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify trajectory parity between reference CPU and accelerated PyTorch simulators."
    )
    parser.add_argument(
        "--output",
        default="results/stage4_validation",
        help="Directory to save comparison metrics and figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parameter sampling.",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=10,
        help="Number of ALD cycles to simulate for validation.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    p = sample_parameters(rng)

    sim_kwargs = dict(
        diffusivity=p[0],
        k_ads=p[1],
        k_des=p[2],
        k_rxn=p[3],
        k_growth=p[4],
        pulse_time=p[5],
        purge_time=p[6],
        reaction_time=p[7],
        num_cycles=args.num_cycles,
    )

    print("Executing reference CPU simulation...")
    ref_sim = ThinFilmDepositionSimulator(**sim_kwargs)
    _, ref_theta, ref_h = ref_sim.run()

    print("Executing accelerated PyTorch simulation...")
    fast_sim = BatchedTransientALDCudaSimulator(np.asarray([p]), nx=80, ny=60)
    pulse_steps = fast_sim.pulse_steps()
    purge_steps = fast_sim.fixed_steps(fast_sim.purge_time)
    reaction_steps = fast_sim.fixed_steps(fast_sim.reaction_time)
    active_mask = torch.ones(1, dtype=torch.bool, device=fast_sim.device)

    max_pulse = int(pulse_steps.max().item())
    for _ in range(args.num_cycles):
        for s in range(1, max_pulse + 1):
            fast_sim.transport_step(pulse_steps >= s, pulse=True)
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()
        for _ in range(reaction_steps):
            fast_sim.reaction_step()
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()

    _, theta_batch, h_batch = fast_sim.snapshot()
    theta = theta_batch[0]
    h = h_batch[0]

    ref_theta_resampled = zoom(ref_theta, (60 / 120, 80 / 160), order=1)
    ref_h_resampled = zoom(ref_h, (60 / 120, 80 / 160), order=1)

    err_theta = np.abs(ref_theta_resampled - theta)
    err_h = np.abs(ref_h_resampled - h)

    metrics = {
        "surface_coverage_mae": float(err_theta.mean()),
        "surface_coverage_rmse": float(np.sqrt((err_theta**2).mean())),
        "film_thickness_mae": float(err_h.mean()),
        "film_thickness_rmse": float(np.sqrt((err_h**2).mean())),
        "reference_grid": [120, 160],
        "accelerated_grid": [60, 80],
        "parameters": p.tolist(),
        "device": str(fast_sim.device),
    }

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "reference_vs_cuda.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    for a, v, t in zip(
        ax[0],
        [ref_h_resampled, h, err_h],
        ["Reference Thickness", "Accelerated Thickness", "Absolute Error"],
    ):
        im = a.imshow(v)
        a.set_title(t)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    for a, v, t in zip(
        ax[1],
        [ref_theta_resampled, theta, err_theta],
        ["Reference Coverage", "Accelerated Coverage", "Absolute Error"],
    ):
        im = a.imshow(v)
        a.set_title(t)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig_path = output_dir / "reference_vs_cuda.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Validation summary saved to: {metrics_path}")


if __name__ == "__main__":
    main()

"""Validation script verifying trajectory parity between reference CPU and accelerated simulators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from thin_film_pino.data.parameters import sample_parameters
from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify trajectory parity between reference CPU and PyTorch simulators on matched grids."
    )
    parser.add_argument(
        "--output",
        default="results/validation",
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
        default=5,
        help="Number of ALD cycles to simulate for validation.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=80,
        help="Horizontal grid resolution.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=60,
        help="Vertical grid resolution.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point precision for parity comparison.",
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
        nx=args.nx,
        ny=args.ny,
        dtype=args.dtype,
    )

    print("Executing reference CPU simulation...")
    ref_sim = ThinFilmDepositionSimulator(**sim_kwargs)
    ref_sim.run()
    ref_theta = ref_sim.theta
    ref_h = ref_sim.h
    ref_c = ref_sim.C

    print("Executing accelerated PyTorch simulation...")
    fast_sim = BatchedTransientALDCudaSimulator(
        np.asarray([p]), nx=args.nx, ny=args.ny, dtype=args.dtype, dt=ref_sim.dt
    )
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

    c_batch, theta_batch, h_batch = fast_sim.snapshot()
    theta = theta_batch[0]
    h = h_batch[0]
    c = c_batch[0]

    err_theta = np.abs(ref_theta - theta)
    err_h = np.abs(ref_h - h)
    err_c = np.abs(ref_c - c)

    norm_ref_theta = float(np.linalg.norm(ref_theta))
    norm_ref_h = float(np.linalg.norm(ref_h))

    rel_l2_theta = float(np.linalg.norm(err_theta) / norm_ref_theta) if norm_ref_theta > 0 else 0.0
    rel_l2_h = float(np.linalg.norm(err_h) / norm_ref_h) if norm_ref_h > 0 else 0.0

    metrics = {
        "surface_coverage_max_error": float(err_theta.max()),
        "surface_coverage_mae": float(err_theta.mean()),
        "surface_coverage_rel_l2": rel_l2_theta,
        "film_thickness_max_error": float(err_h.max()),
        "film_thickness_mae": float(err_h.mean()),
        "film_thickness_rel_l2": rel_l2_h,
        "concentration_max_error": float(err_c.max()),
        "grid": [args.ny, args.nx],
        "dtype": args.dtype,
        "dt": float(ref_sim.dt),
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
        [ref_h, h, err_h],
        ["Reference Thickness", "Accelerated Thickness", "Absolute Error"],
    ):
        im = a.imshow(v, cmap="viridis")
        a.set_title(t)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    for a, v, t in zip(
        ax[1],
        [ref_theta, theta, err_theta],
        ["Reference Coverage", "Accelerated Coverage", "Absolute Error"],
    ):
        im = a.imshow(v, cmap="magma")
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

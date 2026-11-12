"""Unified multi-parameter sensitivity and kinetic sweeps for ALD simulation analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def cell_edge_extent(x: np.ndarray, y: np.ndarray) -> Sequence[float]:
    """Calculate exact cell-edge bounding extent for matplotlib imshow."""
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1.0
    return [
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    ]


def run_kinetic_sweep(args: argparse.Namespace) -> None:
    """Execute 2D parameter sweep over diffusivity D and reaction rate k_rxn."""
    grid_size = args.grid_size
    pulse_time = float(args.pulse_time)
    num_cycles = int(args.num_cycles)

    diffusivities = np.linspace(1.0e-3, 8.0e-3, grid_size)
    reaction_rates = np.linspace(0.1, 0.8, grid_size)

    step_coverage = np.zeros((grid_size, grid_size), dtype=np.float64)
    gpc = np.zeros((grid_size, grid_size), dtype=np.float64)
    field_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
    floor_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)

    total_runs = grid_size * grid_size
    print(
        f"Beginning kinetic sweep ({grid_size}x{grid_size} = {total_runs} simulations, "
        f"pulse_time={pulse_time:.1f}, num_cycles={num_cycles})..."
    )

    for i, k_rxn in enumerate(reaction_rates):
        for j, diff in enumerate(diffusivities):
            sim = ThinFilmDepositionSimulator(
                diffusivity=diff,
                k_rxn=k_rxn,
                k_ads=args.k_ads,
                k_des=args.k_des,
                k_growth=0.02,
                pulse_time=pulse_time,
                purge_time=1.0,
                reaction_time=1.0,
                num_cycles=num_cycles,
                nx=args.nx,
                ny=args.ny,
            )
            sim.run()

            field_h = float(np.mean(sim.h[sim.field_mask])) if np.any(sim.field_mask) else 0.0
            floor_h = float(np.mean(sim.h[sim.floor_mask])) if np.any(sim.floor_mask) else 0.0
            sc = floor_h / field_h if field_h > 0.0 else 0.0

            step_coverage[i, j] = sc
            gpc[i, j] = sim.history["gpc"][-1]
            field_thickness[i, j] = field_h
            floor_thickness[i, j] = floor_h

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        diffusivities=diffusivities,
        reaction_rates=reaction_rates,
        step_coverage=step_coverage,
        gpc=gpc,
        field_thickness=field_thickness,
        floor_thickness=floor_thickness,
        pulse_time=pulse_time,
    )

    extent = cell_edge_extent(diffusivities, reaction_rates)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        step_coverage,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, label="Step Coverage (Floor / Field)")
    plt.xlabel("Precursor Diffusivity D")
    plt.ylabel("Reaction Rate k_rxn")
    plt.title(f"Step Coverage vs D and k_rxn (pulse_time = {pulse_time:.1f})")
    plt.tight_layout()
    plt.savefig(out_dir / "kinetic_sweep_step_coverage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        gpc,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="plasma",
    )
    plt.colorbar(im, label="Growth Per Cycle (thickness/cycle)")
    plt.xlabel("Precursor Diffusivity D")
    plt.ylabel("Reaction Rate k_rxn")
    plt.title(f"Growth Per Cycle vs D and k_rxn (pulse_time = {pulse_time:.1f})")
    plt.tight_layout()
    plt.savefig(out_dir / "kinetic_sweep_gpc.png", dpi=200)
    plt.close()

    print(f"Kinetic sweep completed. Results saved to: {out_npz}")
    print(f"Figures saved to: {out_dir}")


def run_pulse_sweep(args: argparse.Namespace) -> None:
    """Execute 1D sensitivity sweep across precursor pulse exposure durations."""
    pulse_times = np.asarray(args.pulse_times, dtype=np.float64)
    num_cycles = int(args.num_cycles)

    step_coverages = []
    gpcs = []
    field_thicknesses = []
    floor_thicknesses = []
    dose_coverages = []

    print(
        f"Beginning pulse duration sweep ({len(pulse_times)} pulse durations, "
        f"num_cycles={num_cycles})..."
    )

    for pt in pulse_times:
        sim = ThinFilmDepositionSimulator(
            diffusivity=args.diffusivity,
            k_ads=args.k_ads,
            k_des=args.k_des,
            k_rxn=args.k_rxn,
            k_growth=0.02,
            pulse_time=float(pt),
            purge_time=1.0,
            reaction_time=1.0,
            num_cycles=num_cycles,
            nx=args.nx,
            ny=args.ny,
        )
        sim.run()

        field_h = float(np.mean(sim.h[sim.field_mask])) if np.any(sim.field_mask) else 0.0
        floor_h = float(np.mean(sim.h[sim.floor_mask])) if np.any(sim.floor_mask) else 0.0
        sc = floor_h / field_h if field_h > 0.0 else 0.0

        step_coverages.append(sc)
        gpcs.append(sim.history["gpc"][-1])
        field_thicknesses.append(field_h)
        floor_thicknesses.append(floor_h)
        dose_coverages.append(sim.history["dose_coverage"][-1])

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        pulse_times=pulse_times,
        step_coverage=np.asarray(step_coverages),
        gpc=np.asarray(gpcs),
        field_thickness=np.asarray(field_thicknesses),
        floor_thickness=np.asarray(floor_thicknesses),
        dose_coverage=np.asarray(dose_coverages),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(pulse_times, step_coverages, marker="o", color="purple")
    axes[0].set_xlabel("Precursor Pulse Time")
    axes[0].set_ylabel("Step Coverage (Floor / Field)")
    axes[0].set_title(f"Step Coverage vs Pulse Time (D = {args.diffusivity:.1e})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(pulse_times, gpcs, marker="s", color="darkblue", label="GPC")
    axes[1].plot(
        pulse_times, dose_coverages, marker="^", color="darkgreen", label="Dose Saturation theta"
    )
    axes[1].set_xlabel("Precursor Pulse Time")
    axes[1].set_ylabel("GPC / Saturation Coverage")
    axes[1].set_title(f"Deposition Metrics vs Pulse Time (D = {args.diffusivity:.1e})")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "pulse_duration_sweep.png", dpi=200)
    plt.close()

    print(f"Pulse sweep completed. Results saved to: {out_npz}")
    print(f"Figures saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified ALD parameter and sensitivity sweep suite."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sweep type to execute.")

    # Subcommand: kinetic
    p_kin = subparsers.add_parser("kinetic", help="2D sweep over diffusivity and reaction rate.")
    p_kin.add_argument(
        "--grid-size", type=int, default=8, help="Number of points along each parameter axis."
    )
    p_kin.add_argument(
        "--pulse-time", type=float, default=10.0, help="Explicit precursor dose duration."
    )
    p_kin.add_argument("--k-ads", type=float, default=1.0, help="Adsorption rate constant.")
    p_kin.add_argument("--k-des", type=float, default=0.05, help="Desorption rate constant.")
    p_kin.add_argument(
        "--num-cycles", type=int, default=10, help="Number of ALD cycles per evaluation."
    )
    p_kin.add_argument("--nx", type=int, default=40, help="Horizontal grid resolution.")
    p_kin.add_argument("--ny", type=int, default=30, help="Vertical grid resolution.")
    p_kin.add_argument(
        "--output-dir", default="results/figures/sweeps", help="Directory for saved figures."
    )
    p_kin.add_argument(
        "--output-npz", default="data/kinetic_sweep.npz", help="Output path for .npz results."
    )

    # Subcommand: pulse
    p_pulse = subparsers.add_parser("pulse", help="1D sweep over precursor exposure durations.")
    p_pulse.add_argument(
        "--pulse-times",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0],
        help="List of precursor dose exposure durations to simulate.",
    )
    p_pulse.add_argument("--diffusivity", type=float, default=5.0e-3, help="Precursor diffusivity.")
    p_pulse.add_argument("--k-ads", type=float, default=1.0, help="Adsorption rate constant.")
    p_pulse.add_argument("--k-des", type=float, default=0.05, help="Desorption rate constant.")
    p_pulse.add_argument("--k-rxn", type=float, default=0.4, help="Reaction rate constant.")
    p_pulse.add_argument(
        "--num-cycles", type=int, default=10, help="Number of ALD cycles per evaluation."
    )
    p_pulse.add_argument("--nx", type=int, default=40, help="Horizontal grid resolution.")
    p_pulse.add_argument("--ny", type=int, default=30, help="Vertical grid resolution.")
    p_pulse.add_argument(
        "--output-dir", default="results/figures/sweeps", help="Directory for saved figures."
    )
    p_pulse.add_argument(
        "--output-npz", default="data/pulse_sweep.npz", help="Output path for .npz results."
    )

    args = parser.parse_args()
    if args.command == "kinetic":
        run_kinetic_sweep(args)
    elif args.command == "pulse":
        run_pulse_sweep(args)


if __name__ == "__main__":
    main()

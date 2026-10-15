"""Multi-cycle ALD validation: thickness linearity, GPC stabilization, and conformality."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-cycle ALD validation to verify linear thickness growth and GPC stabilization."
    )
    parser.add_argument("--num-cycles", type=int, default=50, help="Total ALD cycles to evaluate.")
    parser.add_argument("--pulse-time", type=float, default=60.0, help="Precursor pulse duration in seconds.")
    parser.add_argument(
        "--output-dir",
        default="results/figures/cycle_validation",
        help="Directory to save diagnostic validation figures.",
    )
    parser.add_argument(
        "--output-npz",
        default="data/cycle_validation.npz",
        help="Path for archived simulation output array.",
    )
    args = parser.parse_args()

    simulator = ThinFilmDepositionSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=args.pulse_time,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=args.num_cycles,
    )

    print("Executing multi-cycle ALD validation...")
    print(f"Precursor pulse duration: {simulator.pulse_time} s")
    print(f"Total cycles: {simulator.num_cycles}")

    simulator.run()

    cycles = np.asarray(simulator.history["cycle"], dtype=np.float64)
    gpc = np.asarray(simulator.history["gpc"], dtype=np.float64)
    mean_thickness = np.asarray(simulator.history["mean_thickness"], dtype=np.float64)
    top_thickness = np.asarray(simulator.history["top_thickness"], dtype=np.float64)
    bottom_thickness = np.asarray(simulator.history["bottom_thickness"], dtype=np.float64)
    conformality = np.asarray(simulator.history["conformality"], dtype=np.float64)
    coverage = np.asarray(simulator.history["surface_coverage"], dtype=np.float64)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        cycles=cycles,
        gpc=gpc,
        mean_thickness=mean_thickness,
        top_thickness=top_thickness,
        bottom_thickness=bottom_thickness,
        conformality=conformality,
        surface_coverage=coverage,
    )

    # Thickness linearity
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, mean_thickness, label="Mean Surface Thickness", marker="o")
    plt.plot(cycles, top_thickness, label="Top Trench Thickness", marker="s")
    plt.plot(cycles, bottom_thickness, label="Bottom Trench Thickness", marker="^")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Film Thickness (m)")
    plt.title("Film Thickness Evolution vs ALD Cycle")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "thickness_vs_cycle.png", dpi=200)
    plt.close()

    # GPC stabilization
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, gpc, marker="o", color="green")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Growth Per Cycle (m/cycle)")
    plt.title("GPC Stabilization Across Cycles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gpc_vs_cycle.png", dpi=200)
    plt.close()

    # Conformality evolution
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, conformality, marker="o", color="purple")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Step Coverage Conformality (Bottom / Top)")
    plt.title("Conformality vs ALD Cycle")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "conformality_vs_cycle.png", dpi=200)
    plt.close()

    # Coverage evolution
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, coverage, marker="o", color="orange")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Fractional Surface Coverage theta")
    plt.title("Surface Coverage vs ALD Cycle")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_vs_cycle.png", dpi=200)
    plt.close()

    print("\nValidation completed successfully.")
    print(f"Data archived to: {out_npz}")
    print(f"Diagnostic figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
"""Multi-cycle ALD validation: thickness linearity, GPC stabilization, and conformality."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-cycle ALD validation to verify linear thickness growth and GPC stabilization."
    )
    parser.add_argument("--num-cycles", type=int, default=50, help="Total ALD cycles to evaluate.")
    parser.add_argument(
        "--pulse-time", type=float, default=60.0, help="Precursor pulse duration in seconds."
    )
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

    print(f"Executing multi-cycle validation ({args.num_cycles} cycles)...")
    history = simulator.run()

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{k: np.asarray(v) for k, v in history.items()})

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = np.asarray(history["cycle"])
    mean_thickness = np.asarray(history["mean_thickness"])
    top_thickness = np.asarray(history["top_thickness"])
    bottom_thickness = np.asarray(history["bottom_thickness"])
    gpc = np.asarray(history["gpc"])
    conformality = np.asarray(history["conformality"])
    coverage = np.asarray(history["surface_coverage"])

    # Linear regression on cycles 10+ to evaluate steady-state linearity
    valid_mask = cycles >= 10
    slope, intercept = np.polyfit(cycles[valid_mask], mean_thickness[valid_mask], 1)
    y_pred = slope * cycles[valid_mask] + intercept
    ss_tot = np.sum((mean_thickness[valid_mask] - np.mean(mean_thickness[valid_mask])) ** 2)
    ss_res = np.sum((mean_thickness[valid_mask] - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    print("\nSteady-State Growth Dynamics (Cycles >= 10)")
    print("-" * 50)
    print(f"Growth Rate Slope: {slope:.6e} m/cycle")
    print(f"Linear Fit R^2:    {r2:.6f}")
    print(f"Cycle 50 GPC:      {gpc[-1]:.6e} m/cycle")
    print(f"Final Conformality:{conformality[-1]:.4f}")

    # Thickness linearity
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, mean_thickness, label="Mean Surface Thickness", marker="o", markersize=3)
    plt.plot(cycles, top_thickness, label="Top Region Thickness", linestyle="--")
    plt.plot(cycles, bottom_thickness, label="Bottom Region Thickness", linestyle=":")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Film Thickness (m)")
    plt.title(f"Thickness vs ALD Cycle (R^2 = {r2:.6f})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "thickness_vs_cycle.png", dpi=200)
    plt.close()

    # GPC stabilization
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, gpc, marker="o", color="darkgreen")
    plt.axhline(slope, color="red", linestyle="--", label=f"Steady Growth ({slope:.4e})")
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

    print("\nValidation completed.")
    print(f"Data archived to: {out_npz}")
    print(f"Diagnostic figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

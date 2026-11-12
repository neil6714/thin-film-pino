"""Multi-cycle ALD validation evaluating thickness linearity, GPC stabilization, and conformality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify linear thickness growth and GPC stabilization across multiple ALD cycles."
    )
    parser.add_argument("--num-cycles", type=int, default=50, help="Total ALD cycles to evaluate.")
    parser.add_argument("--pulse-time", type=float, default=60.0, help="Precursor pulse duration.")
    parser.add_argument("--nx", type=int, default=160, help="Horizontal grid resolution.")
    parser.add_argument("--ny", type=int, default=120, help="Vertical grid resolution.")
    parser.add_argument(
        "--gpc-tolerance",
        type=float,
        default=0.02,
        help="Maximum allowable relative GPC variation over the final 5 cycles.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.999,
        help="Minimum acceptable R^2 coefficient for post-incubation thickness linearity.",
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
        nx=args.nx,
        ny=args.ny,
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
    field_thickness = np.asarray(history["field_thickness"])
    floor_thickness = np.asarray(history["floor_thickness"])
    gpc = np.asarray(history["gpc"])
    step_coverage = np.asarray(history["step_coverage"])
    coverage = np.asarray(history["surface_coverage"])

    # Linear regression on cycles >= 10 to evaluate linearized multi-cycle growth linearity
    min_fit_cycle = min(10, args.num_cycles // 2)
    valid_mask = cycles >= min_fit_cycle
    slope, intercept = np.polyfit(cycles[valid_mask], mean_thickness[valid_mask], 1)
    y_pred = slope * cycles[valid_mask] + intercept
    ss_tot = np.sum((mean_thickness[valid_mask] - np.mean(mean_thickness[valid_mask])) ** 2)
    ss_res = np.sum((mean_thickness[valid_mask] - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0)

    # Relative GPC variation across final cycles
    final_window = min(5, len(gpc))
    rel_gpc_var = float(np.max(np.abs(gpc[-final_window:] - gpc[-1])) / max(abs(gpc[-1]), 1.0e-8))

    r2_passed = r2 >= args.r2_threshold
    gpc_passed = rel_gpc_var <= args.gpc_tolerance
    all_passed = r2_passed and gpc_passed

    print(f"\nLinearized Multi-Cycle Growth Dynamics (Cycles >= {min_fit_cycle})")
    print(f"Growth Rate Slope:        {slope:.6e} thickness/cycle")
    print(f"Linear Fit R^2:           {r2:.6f} (threshold: {args.r2_threshold:.4f})")
    print(f"Final Cycle GPC:          {gpc[-1]:.6e}")
    print(f"Final GPC Relative Var:   {rel_gpc_var:.4e} (tolerance: {args.gpc_tolerance:.4f})")
    print(f"Final Step Coverage:      {step_coverage[-1]:.4f}")
    print(f"Status:                   {'PASS' if all_passed else 'FAIL'}")

    # 1. Thickness linearity
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, mean_thickness, label="Mean Surface Thickness", marker="o", markersize=3)
    plt.plot(cycles, field_thickness, label="Field Region Thickness", linestyle="--")
    plt.plot(cycles, floor_thickness, label="Floor Region Thickness", linestyle=":")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Film Thickness")
    plt.title(f"Thickness vs ALD Cycle (R^2 = {r2:.6f})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "thickness_vs_cycle.png", dpi=200)
    plt.close()

    # 2. GPC stabilization
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, gpc, marker="o", color="darkgreen")
    plt.axhline(slope, color="red", linestyle="--", label=f"Asymptotic Growth ({slope:.4e})")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Growth Per Cycle (thickness/cycle)")
    plt.title("GPC Stabilization Across Cycles")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gpc_vs_cycle.png", dpi=200)
    plt.close()

    # 3. Step coverage evolution
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, step_coverage, marker="o", color="purple")
    plt.xlabel("ALD Cycle Index")
    plt.ylabel("Step Coverage (Floor / Field)")
    plt.title("Step Coverage Evolution Across Cycles")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "conformality_vs_cycle.png", dpi=200)
    plt.close()

    # 4. Coverage evolution
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

    if not all_passed:
        print("Cycle validation failed acceptance criteria.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

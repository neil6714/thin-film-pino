"""Pulse saturation curve analysis verifying Langmuir self-limiting deposition behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator

DEFAULT_PULSE_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 25.0])


def build_simulator(pulse_time: float, num_cycles: int = 10) -> ThinFilmDepositionSimulator:
    """Instantiate reference simulator with designated dose duration."""
    return ThinFilmDepositionSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=pulse_time,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=num_cycles,
    )


def saturation_coverage_by_region(simulator: ThinFilmDepositionSimulator) -> Tuple[float, float]:
    """Calculate mean saturation surface coverage at dose conclusion in field and floor regions."""
    field_cov = (
        float(np.mean(simulator.dose_theta[simulator.field_mask]))
        if np.any(simulator.field_mask)
        else 0.0
    )
    floor_cov = (
        float(np.mean(simulator.dose_theta[simulator.floor_mask]))
        if np.any(simulator.floor_mask)
        else 0.0
    )
    return field_cov, floor_cov


def save_figures(results: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Render and save diagnostic pulse saturation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pulses = results["pulse_time"]

    plt.figure(figsize=(8, 5))
    plt.plot(pulses, results["gpc"], marker="o", color="blue")
    plt.xlabel("Precursor Dose Duration")
    plt.ylabel("Growth Per Cycle (thickness/cycle)")
    plt.title("GPC vs Precursor Dose Duration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pulse_vs_gpc.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pulses, results["step_coverage"], marker="s", color="purple")
    plt.xlabel("Precursor Dose Duration")
    plt.ylabel("Step Coverage (Floor / Field)")
    plt.title("Step Coverage vs Precursor Dose Duration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pulse_vs_conformality.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        pulses, results["field_saturation_coverage"], marker="^", label="Field Saturation Coverage"
    )
    plt.plot(
        pulses, results["floor_saturation_coverage"], marker="v", label="Floor Saturation Coverage"
    )
    plt.xlabel("Precursor Dose Duration")
    plt.ylabel("Saturation Surface Coverage")
    plt.title("Dose Saturation Coverage vs Precursor Dose Duration")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pulse_vs_coverage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pulses, results["field_thickness"], marker="^", label="Field Thickness")
    plt.plot(pulses, results["floor_thickness"], marker="v", label="Floor Thickness")
    plt.xlabel("Precursor Dose Duration")
    plt.ylabel("Film Thickness")
    plt.title("Film Thickness vs Precursor Dose Duration")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pulse_vs_thickness.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(pulses, results["gpc"], marker="o", color="blue")
    axes[0, 0].set_title("Growth Per Cycle")
    axes[0, 0].set_xlabel("Dose Duration")
    axes[0, 0].set_ylabel("GPC (thickness/cycle)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(pulses, results["step_coverage"], marker="s", color="purple")
    axes[0, 1].set_title("Step Coverage")
    axes[0, 1].set_xlabel("Dose Duration")
    axes[0, 1].set_ylabel("Floor / Field")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(pulses, results["field_saturation_coverage"], marker="^", label="Field")
    axes[1, 0].plot(pulses, results["floor_saturation_coverage"], marker="v", label="Floor")
    axes[1, 0].set_title("Dose Saturation Coverage")
    axes[1, 0].set_xlabel("Dose Duration")
    axes[1, 0].set_ylabel("Saturation Coverage")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(pulses, results["field_thickness"], marker="^", label="Field")
    axes[1, 1].plot(pulses, results["floor_thickness"], marker="v", label="Floor")
    axes[1, 1].set_title("Film Thickness")
    axes[1, 1].set_xlabel("Dose Duration")
    axes[1, 1].set_ylabel("Thickness")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pulse_saturation_diagnostics.png", dpi=200)
    plt.close()


def run_sweep(
    pulse_times: np.ndarray = DEFAULT_PULSE_TIMES,
    num_cycles: int = 10,
    output_dir: str = "results/figures/pulse_saturation",
    output_npz: str = "data/pulse_saturation.npz",
) -> None:
    results: Dict[str, list] = {
        "pulse_time": list(pulse_times),
        "gpc": [],
        "field_thickness": [],
        "floor_thickness": [],
        "step_coverage": [],
        "field_saturation_coverage": [],
        "floor_saturation_coverage": [],
    }

    print("Beginning precursor pulse saturation sweep...")
    for pt in pulse_times:
        sim = build_simulator(pt, num_cycles=num_cycles)
        sim.run()

        field_cov, floor_cov = saturation_coverage_by_region(sim)
        results["gpc"].append(sim.history["gpc"][-1])
        results["field_thickness"].append(sim.history["field_thickness"][-1])
        results["floor_thickness"].append(sim.history["floor_thickness"][-1])
        results["step_coverage"].append(sim.history["step_coverage"][-1])
        results["field_saturation_coverage"].append(field_cov)
        results["floor_saturation_coverage"].append(floor_cov)

        print(
            f"Pulse={pt:6.1f} | GPC={results['gpc'][-1]:.6e} | "
            f"Step Coverage={results['step_coverage'][-1]:.6f} | "
            f"Field sat theta={field_cov:.4f} | Floor sat theta={floor_cov:.4f}"
        )

    out_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in results.items()}
    out_file = Path(output_npz).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **out_arrays)

    out_fig_dir = Path(output_dir).resolve()
    save_figures(out_arrays, out_fig_dir)

    print("\nPulse saturation sweep completed.")
    print(f"Data archived to: {out_file}")
    print(f"Diagnostic figures saved to: {out_fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Langmuir self-limiting precursor saturation behavior in trench features."
    )
    parser.add_argument(
        "--num-cycles", type=int, default=10, help="Cycles simulated per pulse duration."
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/pulse_saturation",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--output-npz", default="data/pulse_saturation.npz", help="Path for archived .npz results."
    )
    args = parser.parse_args()

    run_sweep(num_cycles=args.num_cycles, output_dir=args.output_dir, output_npz=args.output_npz)


if __name__ == "__main__":
    main()

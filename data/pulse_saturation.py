"""Pulse saturation curve analysis verifying Langmuir self-limiting deposition behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator

DEFAULT_PULSE_TIMES = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0])


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


def surface_coverage_by_region(simulator: ThinFilmDepositionSimulator) -> Tuple[float, float]:
    """Calculate mean fractional surface coverage in top and bottom trench regions."""
    top_region = simulator.surface_mask & (simulator.Y < simulator.trench_top + 2.5 * simulator.dy)
    bottom_region = simulator.surface_mask & (simulator.Y > simulator.trench_bottom - 2.5 * simulator.dy)
    top_cov = float(np.mean(simulator.theta[top_region])) if np.any(top_region) else 0.0
    bottom_cov = float(np.mean(simulator.theta[bottom_region])) if np.any(bottom_region) else 0.0
    return top_cov, bottom_cov


def save_figures(results: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Render and save diagnostic pulse saturation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pulse_times = results["pulse_times"]

    figure_specs = (
        ("gpc", "Growth Per Cycle (m/cycle)", "Pulse Saturation: GPC", "pulse_vs_gpc.png"),
        ("conformality", "Step Coverage Conformality", "Pulse Saturation: Conformality", "pulse_vs_conformality.png"),
    )

    for metric, ylabel, title, filename in figure_specs:
        plt.figure(figsize=(8, 5))
        plt.plot(pulse_times, results[metric], marker="o")
        plt.xlabel("Precursor Pulse Duration (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()

    for top_metric, bottom_metric, ylabel, title, filename in (
        ("top_thickness", "bottom_thickness", "Film Thickness (m)", "Regional Film Thickness Saturation", "pulse_vs_thickness.png"),
        ("top_coverage", "bottom_coverage", "Fractional Surface Coverage theta", "Regional Surface Coverage Saturation", "pulse_vs_coverage.png"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(pulse_times, results[top_metric], marker="o", label="Trench Opening (Top)")
        plt.plot(pulse_times, results[bottom_metric], marker="s", label="Trench Floor (Bottom)")
        plt.xlabel("Precursor Pulse Duration (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def run_sweep(
    pulse_times: np.ndarray = DEFAULT_PULSE_TIMES,
    num_cycles: int = 10,
    output_dir: str = "results/figures/pulse_saturation",
    output_npz: str = "data/pulse_saturation.npz",
) -> None:
    """Execute pulse saturation sweep, validating self-limiting Langmuir kinetics."""
    results: Dict[str, list] = {
        "pulse_times": list(pulse_times),
        "gpc": [],
        "top_thickness": [],
        "bottom_thickness": [],
        "conformality": [],
        "top_coverage": [],
        "bottom_coverage": [],
    }

    print("Beginning precursor pulse saturation sweep...")
    for pt in pulse_times:
        sim = build_simulator(pt, num_cycles=num_cycles)
        sim.run()

        top_cov, bottom_cov = surface_coverage_by_region(sim)
        results["gpc"].append(sim.history["gpc"][-1])
        results["top_thickness"].append(sim.history["top_thickness"][-1])
        results["bottom_thickness"].append(sim.history["bottom_thickness"][-1])
        results["conformality"].append(sim.history["conformality"][-1])
        results["top_coverage"].append(top_cov)
        results["bottom_coverage"].append(bottom_cov)

        print(
            f"Pulse={pt:6.1f} s | GPC={results['gpc'][-1]:.6e} m/cycle | "
            f"Conformality={results['conformality'][-1]:.6f} | "
            f"Top theta={top_cov:.4f} | Bottom theta={bottom_cov:.4f}"
        )

    out_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in results.items()}
    out_file = Path(output_npz).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **out_arrays)

    out_fig_dir = Path(output_dir).resolve()
    save_figures(out_arrays, out_fig_dir)

    print("\nPulse saturation sweep completed successfully.")
    print(f"Data archived to: {out_file}")
    print(f"Diagnostic figures saved to: {out_fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Langmuir self-limiting precursor saturation behavior in high-aspect-ratio features."
    )
    parser.add_argument("--num-cycles", type=int, default=10, help="Cycles simulated per pulse duration.")
    parser.add_argument("--output-dir", default="results/figures/pulse_saturation", help="Output directory for plots.")
    parser.add_argument("--output-npz", default="data/pulse_saturation.npz", help="Path for archived .npz results.")
    args = parser.parse_args()

    run_sweep(num_cycles=args.num_cycles, output_dir=args.output_dir, output_npz=args.output_npz)


if __name__ == "__main__":
    main()

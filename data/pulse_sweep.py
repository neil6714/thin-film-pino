"""Dose duration sensitivity sweep measuring conformality and growth across short-to-medium pulses."""

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

DEFAULT_PULSE_TIMES = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 50.0])


def run_sweep(
    pulse_times: np.ndarray = DEFAULT_PULSE_TIMES,
    num_cycles: int = 10,
    output_dir: str = "results/figures/pulse_sweep",
    output_npz: str = "data/pulse_sweep.npz",
) -> None:
    """Execute pulse exposure sweep across selected pulse times."""
    conformality = []
    gpc = []
    top_thickness = []
    bottom_thickness = []

    print("Beginning precursor pulse exposure duration sweep...")
    for pt in pulse_times:
        sim = ThinFilmDepositionSimulator(
            diffusivity=5.0e-3,
            k_ads=1.0,
            k_des=0.05,
            k_rxn=0.4,
            k_growth=0.02,
            pulse_time=pt,
            purge_time=1.0,
            reaction_time=1.0,
            num_cycles=num_cycles,
        )
        sim.run()

        conformality.append(sim.history["conformality"][-1])
        gpc.append(sim.history["gpc"][-1])
        top_thickness.append(sim.history["top_thickness"][-1])
        bottom_thickness.append(sim.history["bottom_thickness"][-1])

        print(
            f"Pulse={pt:6.1f} s | GPC={gpc[-1]:.6e} m/cycle | "
            f"Top={top_thickness[-1]:.6e} m | Bottom={bottom_thickness[-1]:.6e} m | "
            f"Conformality={conformality[-1]:.6f}"
        )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(output_npz).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        pulse_times=pulse_times,
        conformality=np.asarray(conformality, dtype=np.float64),
        gpc=np.asarray(gpc, dtype=np.float64),
        top_thickness=np.asarray(top_thickness, dtype=np.float64),
        bottom_thickness=np.asarray(bottom_thickness, dtype=np.float64),
    )

    plt.figure(figsize=(8, 5))
    plt.plot(pulse_times, conformality, marker="o", color="purple")
    plt.xlabel("Precursor Pulse Duration (s)")
    plt.ylabel("Step Coverage Conformality")
    plt.title("Step Coverage Conformality vs Pulse Duration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "conformality_vs_pulse.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pulse_times, gpc, marker="s", color="green")
    plt.xlabel("Precursor Pulse Duration (s)")
    plt.ylabel("Growth Per Cycle (m/cycle)")
    plt.title("Growth Per Cycle vs Pulse Duration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "gpc_vs_pulse.png", dpi=200)
    plt.close()

    print("\nPulse sweep completed successfully.")
    print(f"Archived dataset saved to: {out_file}")
    print(f"Diagnostic figures saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate step coverage conformality over short-to-medium precursor pulses.")
    parser.add_argument("--num-cycles", type=int, default=10, help="Cycles per simulation sample.")
    parser.add_argument("--output-dir", default="results/figures/pulse_sweep", help="Directory for output figures.")
    parser.add_argument("--output-npz", default="data/pulse_sweep.npz", help="Path for archived .npz results.")
    args = parser.parse_args()

    run_sweep(num_cycles=args.num_cycles, output_dir=args.output_dir, output_npz=args.output_npz)


if __name__ == "__main__":
    main()
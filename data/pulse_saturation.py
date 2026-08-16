import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.simulator import ThinFilmDepositionSimulator


OUTPUT_DIR = "results/figures/pulse_saturation"
PULSE_TIMES = np.array(
    [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
)


def build_simulator(pulse_time):
    return ThinFilmDepositionSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=pulse_time,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=10,
    )


def surface_coverage_by_region(simulator):
    top_region = simulator.surface_mask & (
        simulator.Y < simulator.trench_top + 2.5 * simulator.dy
    )
    bottom_region = simulator.surface_mask & (
        simulator.Y > simulator.trench_bottom - 2.5 * simulator.dy
    )

    return (
        np.mean(simulator.theta[top_region]),
        np.mean(simulator.theta[bottom_region]),
    )


def save_figures(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pulse_times = results["pulse_times"]

    figure_specs = (
        ("gpc", "Growth per cycle", "Pulse Saturation: GPC", "pulse_vs_gpc.png"),
        (
            "conformality",
            "Conformality",
            "Pulse Saturation: Conformality",
            "pulse_vs_conformality.png",
        ),
    )

    for metric, ylabel, title, filename in figure_specs:
        plt.figure(figsize=(8, 5))
        plt.plot(pulse_times, results[metric], marker="o")
        plt.xlabel("Pulse time")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
        plt.close()

    for top_metric, bottom_metric, ylabel, title, filename in (
        (
            "top_thickness",
            "bottom_thickness",
            "Film thickness",
            "Final Film Thickness",
            "pulse_vs_thickness.png",
        ),
        (
            "top_coverage",
            "bottom_coverage",
            "Surface coverage",
            "Final Surface Coverage",
            "pulse_vs_coverage.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(pulse_times, results[top_metric], marker="o", label="Top")
        plt.plot(pulse_times, results[bottom_metric], marker="o", label="Bottom")
        plt.xlabel("Pulse time")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
        plt.close()


def run():
    results = {
        "pulse_times": PULSE_TIMES,
        "gpc": [],
        "top_thickness": [],
        "bottom_thickness": [],
        "conformality": [],
        "top_coverage": [],
        "bottom_coverage": [],
    }

    for pulse_time in PULSE_TIMES:
        simulator = build_simulator(pulse_time)
        simulator.run()

        top_coverage, bottom_coverage = surface_coverage_by_region(simulator)
        results["gpc"].append(simulator.history["gpc"][-1])
        results["top_thickness"].append(simulator.history["top_thickness"][-1])
        results["bottom_thickness"].append(simulator.history["bottom_thickness"][-1])
        results["conformality"].append(simulator.history["conformality"][-1])
        results["top_coverage"].append(top_coverage)
        results["bottom_coverage"].append(bottom_coverage)

        print(
            f"Pulse={pulse_time:6.1f} | GPC={results['gpc'][-1]:.6e} | "
            f"Top={results['top_thickness'][-1]:.6e} | "
            f"Bottom={results['bottom_thickness'][-1]:.6e} | "
            f"Conformality={results['conformality'][-1]:.6f} | "
            f"Top coverage={top_coverage:.6f} | "
            f"Bottom coverage={bottom_coverage:.6f}"
        )

    results = {name: np.asarray(values) for name, values in results.items()}
    np.savez("data/pulse_saturation.npz", **results)
    save_figures(results)

    print("\nPulse saturation sweep complete.")
    print("Saved to: data/pulse_saturation.npz")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()

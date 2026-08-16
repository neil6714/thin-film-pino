import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.simulator import ThinFilmDepositionSimulator


OUTPUT_DIR = "results/figures/cycle_validation"


def main():
    simulator = ThinFilmDepositionSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=60.0,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=50,
    )

    print("Starting multi-cycle ALD validation...")
    print(f"Pulse time = {simulator.pulse_time}")
    print(f"Cycles = {simulator.num_cycles}")

    simulator.run()

    cycles = np.asarray(simulator.history["cycle"])
    gpc = np.asarray(simulator.history["gpc"])
    mean_thickness = np.asarray(
        simulator.history["mean_thickness"]
    )
    top_thickness = np.asarray(
        simulator.history["top_thickness"]
    )
    bottom_thickness = np.asarray(
        simulator.history["bottom_thickness"]
    )
    conformality = np.asarray(
        simulator.history["conformality"]
    )
    coverage = np.asarray(
        simulator.history["surface_coverage"]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez(
        "data/cycle_validation.npz",
        cycles=cycles,
        gpc=gpc,
        mean_thickness=mean_thickness,
        top_thickness=top_thickness,
        bottom_thickness=bottom_thickness,
        conformality=conformality,
        coverage=coverage,
    )

    stable_gpc = np.mean(gpc[-10:])
    gpc_variation = (
        np.std(gpc[-10:]) / stable_gpc
        if stable_gpc > 0.0
        else np.inf
    )

    thickness_fit = np.polyfit(
        cycles[-20:],
        mean_thickness[-20:],
        1,
    )

    predicted = np.polyval(
        thickness_fit,
        cycles[-20:],
    )

    ss_res = np.sum(
        (mean_thickness[-20:] - predicted) ** 2
    )

    ss_tot = np.sum(
        (
            mean_thickness[-20:]
            - np.mean(mean_thickness[-20:])
        ) ** 2
    )

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0.0
        else 1.0
    )

    coverage_valid = (
        np.all(coverage >= 0.0)
        and np.all(coverage <= 1.0)
    )

    conformality_stable = np.all(
        conformality[-10:] > 0.5
    )

    print("\nValidation results")
    print("-" * 55)
    print(
        f"Stable GPC:          {stable_gpc:.6e}"
    )
    print(
        f"GPC variation:       {gpc_variation:.4f}"
    )
    print(
        f"Thickness R^2:       {r_squared:.6f}"
    )
    print(
        f"Final conformality:  {conformality[-1]:.4f}"
    )
    print(
        f"Final top coverage:  "
        f"{coverage[-1]:.6f}"
    )
    print(
        f"Coverage bounded:    "
        f"{coverage_valid}"
    )
    print(
        f"High conformality:   "
        f"{conformality_stable}"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(cycles, gpc, marker="o")
    plt.xlabel("ALD cycle")
    plt.ylabel("Growth per cycle")
    plt.title("GPC vs ALD Cycle")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gpc_vs_cycle.png",
        ),
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        cycles,
        mean_thickness,
        marker="o",
    )
    plt.xlabel("ALD cycle")
    plt.ylabel("Mean film thickness")
    plt.title("Film Thickness vs ALD Cycle")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "thickness_vs_cycle.png",
        ),
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        cycles,
        conformality,
        marker="o",
    )
    plt.xlabel("ALD cycle")
    plt.ylabel("Conformality")
    plt.title("Conformality vs ALD Cycle")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "conformality_vs_cycle.png",
        ),
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        cycles,
        coverage,
        marker="o",
    )
    plt.xlabel("ALD cycle")
    plt.ylabel("Surface coverage")
    plt.title("Surface Coverage vs ALD Cycle")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "coverage_vs_cycle.png",
        ),
        dpi=200,
    )
    plt.close()

    print("\nValidation complete.")
    print("Data saved to: data/cycle_validation.npz")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
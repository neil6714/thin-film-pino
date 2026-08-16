import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.simulator import ThinFilmDepositionSimulator


OUTPUT_DIR = "results/figures/pulse_sweep"


def run():
    pulse_times = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 50.0])

    conformality = []
    gpc = []
    top_thickness = []
    bottom_thickness = []

    for pulse_time in pulse_times:
        simulator = ThinFilmDepositionSimulator(
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

        simulator.run()

        conformality.append(
            simulator.history["conformality"][-1]
        )
        gpc.append(
            simulator.history["gpc"][-1]
        )
        top_thickness.append(
            simulator.history["top_thickness"][-1]
        )
        bottom_thickness.append(
            simulator.history["bottom_thickness"][-1]
        )

        print(
            f"Pulse={pulse_time:6.1f} | "
            f"GPC={gpc[-1]:.6e} | "
            f"Top={top_thickness[-1]:.6e} | "
            f"Bottom={bottom_thickness[-1]:.6e} | "
            f"Conformality={conformality[-1]:.6f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez(
        "data/pulse_sweep.npz",
        pulse_times=pulse_times,
        conformality=np.asarray(conformality),
        gpc=np.asarray(gpc),
        top_thickness=np.asarray(top_thickness),
        bottom_thickness=np.asarray(bottom_thickness),
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        pulse_times,
        conformality,
        marker="o",
    )
    plt.xlabel("Pulse time")
    plt.ylabel("Conformality")
    plt.title("Pulse Time vs Conformality")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pulse_vs_conformality.png",
        ),
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        pulse_times,
        gpc,
        marker="o",
    )
    plt.xlabel("Pulse time")
    plt.ylabel("Growth per cycle")
    plt.title("Pulse Time vs GPC")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pulse_vs_gpc.png",
        ),
        dpi=200,
    )
    plt.close()

    print("\nPulse sweep complete.")
    print("Saved to: data/pulse_sweep.npz")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
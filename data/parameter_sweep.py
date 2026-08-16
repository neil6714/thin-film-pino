import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.simulator import ThinFilmDepositionSimulator


OUTPUT_DIR = "results/figures/parameter_sweep"


def run_sweep():
    diffusivities = np.linspace(1.0e-3, 8.0e-3, 8)
    reaction_rates = np.linspace(0.1, 0.8, 8)

    conformality = np.zeros(
        (len(reaction_rates), len(diffusivities))
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, k_rxn in enumerate(reaction_rates):
        for j, diffusivity in enumerate(diffusivities):
            simulator = ThinFilmDepositionSimulator(
                diffusivity=diffusivity,
                k_rxn=k_rxn,
                k_ads=1.0,
                k_des=0.05,
                k_growth=0.02,
            )

            simulator.run()

            top = simulator.history["top_thickness"][-1]
            bottom = simulator.history["bottom_thickness"][-1]

            if top > 0.0:
                conformality[i, j] = bottom / top

            print(
                f"D={diffusivity:.4e}, "
                f"k_rxn={k_rxn:.3f}, "
                f"conformality={conformality[i, j]:.4f}"
            )

    np.savez(
        "data/parameter_sweep.npz",
        diffusivities=diffusivities,
        reaction_rates=reaction_rates,
        conformality=conformality,
    )

    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        conformality,
        origin="lower",
        aspect="auto",
        extent=[
            diffusivities.min(),
            diffusivities.max(),
            reaction_rates.min(),
            reaction_rates.max(),
        ],
    )

    plt.colorbar(image, label="Conformality")
    plt.xlabel("Diffusivity")
    plt.ylabel("Reaction rate")
    plt.title("Conformality Parameter Map")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "conformality_parameter_map.png",
        ),
        dpi=200,
    )

    plt.close()

    best_index = np.unravel_index(
        np.argmax(conformality),
        conformality.shape,
    )

    worst_index = np.unravel_index(
        np.argmin(conformality),
        conformality.shape,
    )

    print("\nSweep complete.")
    print(
        f"Best conformality: "
        f"{conformality[best_index]:.4f}"
    )
    print(
        f"Best D: "
        f"{diffusivities[best_index[1]]:.4e}"
    )
    print(
        f"Best k_rxn: "
        f"{reaction_rates[best_index[0]]:.3f}"
    )

    print(
        f"Worst conformality: "
        f"{conformality[worst_index]:.4f}"
    )
    print(
        f"Worst D: "
        f"{diffusivities[worst_index[1]]:.4e}"
    )
    print(
        f"Worst k_rxn: "
        f"{reaction_rates[worst_index[0]]:.3f}"
    )


if __name__ == "__main__":
    run_sweep()
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.simulator import ThinFilmDepositionSimulator


OUTPUT_DIR = "results/figures/kinetic_sweep"


def run_simulation(diffusivity, k_rxn, k_ads, k_des):
    simulator = ThinFilmDepositionSimulator(
        diffusivity=diffusivity,
        k_rxn=k_rxn,
        k_ads=k_ads,
        k_des=k_des,
        k_growth=0.02,
    )

    simulator.run()

    top = simulator.history["top_thickness"][-1]
    bottom = simulator.history["bottom_thickness"][-1]

    conformality = bottom / top if top > 0.0 else 0.0
    mean_thickness = simulator.h[simulator.surface_mask].mean()

    return top, bottom, mean_thickness, conformality


def run_sweep_d_rxn():
    diffusivities = np.linspace(1.0e-3, 8.0e-3, 8)
    reaction_rates = np.linspace(0.1, 0.8, 8)

    conformality = np.zeros((8, 8))
    top_thickness = np.zeros((8, 8))
    bottom_thickness = np.zeros((8, 8))
    mean_thickness = np.zeros((8, 8))

    for i, k_rxn in enumerate(reaction_rates):
        for j, diffusivity in enumerate(diffusivities):
            top, bottom, mean_h, conf = run_simulation(
                diffusivity=diffusivity,
                k_rxn=k_rxn,
                k_ads=1.0,
                k_des=0.05,
            )

            conformality[i, j] = conf
            top_thickness[i, j] = top
            bottom_thickness[i, j] = bottom
            mean_thickness[i, j] = mean_h

            print(
                f"[D-k_rxn] D={diffusivity:.4e}, "
                f"k_rxn={k_rxn:.3f}, "
                f"conf={conf:.4f}"
            )

    return (
        diffusivities,
        reaction_rates,
        conformality,
        top_thickness,
        bottom_thickness,
        mean_thickness,
    )


def run_sweep_ads_des():
    adsorption_rates = np.linspace(0.5, 1.5, 8)
    desorption_rates = np.linspace(0.02, 0.10, 8)

    conformality = np.zeros((8, 8))
    top_thickness = np.zeros((8, 8))
    bottom_thickness = np.zeros((8, 8))
    mean_thickness = np.zeros((8, 8))

    for i, k_des in enumerate(desorption_rates):
        for j, k_ads in enumerate(adsorption_rates):
            top, bottom, mean_h, conf = run_simulation(
                diffusivity=5.0e-3,
                k_rxn=0.4,
                k_ads=k_ads,
                k_des=k_des,
            )

            conformality[i, j] = conf
            top_thickness[i, j] = top
            bottom_thickness[i, j] = bottom
            mean_thickness[i, j] = mean_h

            print(
                f"[k_ads-k_des] k_ads={k_ads:.3f}, "
                f"k_des={k_des:.3f}, "
                f"conf={conf:.4f}"
            )

    return (
        adsorption_rates,
        desorption_rates,
        conformality,
        top_thickness,
        bottom_thickness,
        mean_thickness,
    )


def plot_map(x, y, values, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 6))

    image = plt.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max(),
        ],
    )

    plt.colorbar(image, label="Conformality")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=200,
    )

    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    (
        diffusivities,
        reaction_rates,
        d_rxn_conf,
        d_rxn_top,
        d_rxn_bottom,
        d_rxn_mean,
    ) = run_sweep_d_rxn()

    (
        adsorption_rates,
        desorption_rates,
        ads_des_conf,
        ads_des_top,
        ads_des_bottom,
        ads_des_mean,
    ) = run_sweep_ads_des()

    np.savez(
        "data/kinetic_sweep.npz",
        diffusivities=diffusivities,
        reaction_rates=reaction_rates,
        adsorption_rates=adsorption_rates,
        desorption_rates=desorption_rates,
        d_rxn_conformality=d_rxn_conf,
        d_rxn_top_thickness=d_rxn_top,
        d_rxn_bottom_thickness=d_rxn_bottom,
        d_rxn_mean_thickness=d_rxn_mean,
        ads_des_conformality=ads_des_conf,
        ads_des_top_thickness=ads_des_top,
        ads_des_bottom_thickness=ads_des_bottom,
        ads_des_mean_thickness=ads_des_mean,
    )

    plot_map(
        diffusivities,
        reaction_rates,
        d_rxn_conf,
        "Diffusivity",
        "Reaction rate",
        "Conformality: Diffusivity vs Reaction Rate",
        "diffusivity_reaction_conformality.png",
    )

    plot_map(
        adsorption_rates,
        desorption_rates,
        ads_des_conf,
        "Adsorption rate",
        "Desorption rate",
        "Conformality: Adsorption vs Desorption",
        "adsorption_desorption_conformality.png",
    )

    print("\nKinetic sweep complete.")
    print("Saved to: data/kinetic_sweep.npz")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
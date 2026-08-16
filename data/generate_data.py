import os
import numpy as np

from physics.simulator import ThinFilmDepositionSimulator


def sample_parameters(rng):
    return {
        "diffusivity": rng.uniform(1.0e-3, 4.0e-3),
        "k_ads": rng.uniform(0.5, 1.5),
        "k_des": rng.uniform(0.02, 0.10),
        "k_rxn": rng.uniform(0.2, 0.8),
        "k_growth": rng.uniform(0.01, 0.04),
    }


def run_simulation(parameters):
    simulator = ThinFilmDepositionSimulator(
        diffusivity=parameters["diffusivity"],
        k_ads=parameters["k_ads"],
        k_des=parameters["k_des"],
        k_rxn=parameters["k_rxn"],
        k_growth=parameters["k_growth"],
    )

    simulator.run()

    top_thickness = simulator.history["top_thickness"][-1]
    bottom_thickness = simulator.history["bottom_thickness"][-1]

    if top_thickness > 0.0:
        conformality = bottom_thickness / top_thickness
    else:
        conformality = 0.0

    return {
        "concentration": simulator.C.copy(),
        "surface_coverage": simulator.theta.copy(),
        "film_thickness": simulator.h.copy(),
        "top_thickness": top_thickness,
        "bottom_thickness": bottom_thickness,
        "conformality": conformality,
    }


def generate_dataset(num_samples=100, seed=42):
    rng = np.random.default_rng(seed)

    parameters = []
    concentrations = []
    surface_coverages = []
    film_thicknesses = []
    top_thicknesses = []
    bottom_thicknesses = []
    conformalities = []

    for index in range(num_samples):
        parameter_set = sample_parameters(rng)
        result = run_simulation(parameter_set)

        parameters.append([
            parameter_set["diffusivity"],
            parameter_set["k_ads"],
            parameter_set["k_des"],
            parameter_set["k_rxn"],
            parameter_set["k_growth"],
        ])

        concentrations.append(result["concentration"])
        surface_coverages.append(result["surface_coverage"])
        film_thicknesses.append(result["film_thickness"])
        top_thicknesses.append(result["top_thickness"])
        bottom_thicknesses.append(result["bottom_thickness"])
        conformalities.append(result["conformality"])

        print(
            f"[{index + 1:03d}/{num_samples}] "
            f"conformality={result['conformality']:.4f}"
        )

    os.makedirs("data", exist_ok=True)

    np.savez_compressed(
        "data/deposition_dataset.npz",
        parameters=np.asarray(parameters, dtype=np.float64),
        concentration=np.asarray(concentrations, dtype=np.float64),
        surface_coverage=np.asarray(surface_coverages, dtype=np.float64),
        film_thickness=np.asarray(film_thicknesses, dtype=np.float64),
        top_thickness=np.asarray(top_thicknesses, dtype=np.float64),
        bottom_thickness=np.asarray(bottom_thicknesses, dtype=np.float64),
        conformality=np.asarray(conformalities, dtype=np.float64),
    )

    print("\nDataset generation complete.")
    print("Saved to: data/deposition_dataset.npz")


if __name__ == "__main__":
    generate_dataset()
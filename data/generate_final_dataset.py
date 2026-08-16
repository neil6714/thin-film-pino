import os

import numpy as np

from physics.simulator import ThinFilmDepositionSimulator


SEED = 42
NUM_SAMPLES = 2000
NUM_CYCLES = 10
OUTPUT_FILE = "data/final_dataset.npz"

PARAMETER_RANGES = {
    "diffusivity": (1.0e-3, 8.0e-3),
    "k_ads": (0.5, 1.5),
    "k_des": (0.02, 0.10),
    "k_rxn": (0.1, 0.8),
    "k_growth": (0.02, 0.02),
    "pulse_time": (10.0, 100.0),
    "purge_time": (1.0, 1.0),
    "reaction_time": (1.0, 1.0),
}


def sample_pulse_time(rng):
    if rng.random() < 0.85:
        return rng.uniform(10.0, 60.0)

    return rng.uniform(60.0, 100.0)


def sample_parameters(rng):
    return np.array([
        rng.uniform(*PARAMETER_RANGES["diffusivity"]),
        rng.uniform(*PARAMETER_RANGES["k_ads"]),
        rng.uniform(*PARAMETER_RANGES["k_des"]),
        rng.uniform(*PARAMETER_RANGES["k_rxn"]),
        PARAMETER_RANGES["k_growth"][0],
        sample_pulse_time(rng),
        PARAMETER_RANGES["purge_time"][0],
        PARAMETER_RANGES["reaction_time"][0],
    ])


def run_sample(parameters):
    (
        diffusivity,
        k_ads,
        k_des,
        k_rxn,
        k_growth,
        pulse_time,
        purge_time,
        reaction_time,
    ) = parameters

    simulator = ThinFilmDepositionSimulator(
        diffusivity=diffusivity,
        k_ads=k_ads,
        k_des=k_des,
        k_rxn=k_rxn,
        k_growth=k_growth,
        pulse_time=pulse_time,
        purge_time=purge_time,
        reaction_time=reaction_time,
        num_cycles=NUM_CYCLES,
    )

    simulator.run()

    concentration = simulator.C.copy()
    surface_coverage = simulator.theta.copy()
    film_thickness = simulator.h.copy()

    gpc = simulator.history["gpc"][-1]
    top_thickness = simulator.history["top_thickness"][-1]
    bottom_thickness = simulator.history["bottom_thickness"][-1]
    conformality = simulator.history["conformality"][-1]

    valid = (
        np.all(np.isfinite(concentration))
        and np.all(np.isfinite(surface_coverage))
        and np.all(np.isfinite(film_thickness))
        and np.isfinite(gpc)
        and np.isfinite(top_thickness)
        and np.isfinite(bottom_thickness)
        and np.isfinite(conformality)
        and np.all(concentration >= 0.0)
        and np.all(surface_coverage >= 0.0)
        and np.all(surface_coverage <= 1.0)
        and np.all(film_thickness >= 0.0)
        and gpc >= 0.0
        and top_thickness >= 0.0
        and bottom_thickness >= 0.0
        and conformality >= 0.0
    )

    if not valid:
        return None

    return (
        concentration,
        surface_coverage,
        film_thickness,
        gpc,
        top_thickness,
        bottom_thickness,
        conformality,
    )


def main():
    rng = np.random.default_rng(SEED)

    parameters = []
    concentrations = []
    surface_coverages = []
    film_thicknesses = []

    gpc = []
    top_thickness = []
    bottom_thickness = []
    conformality = []

    accepted = 0
    attempts = 0

    print("Generating final ALD dataset...")
    print(f"Target samples: {NUM_SAMPLES}")
    print(f"ALD cycles: {NUM_CYCLES}")

    while accepted < NUM_SAMPLES:
        attempts += 1

        sample = sample_parameters(rng)
        result = run_sample(sample)

        if result is None:
            print(
                f"Rejected sample at attempt {attempts}"
            )
            continue

        (
            concentration,
            surface_coverage,
            film_thickness,
            sample_gpc,
            sample_top,
            sample_bottom,
            sample_conformality,
        ) = result

        parameters.append(sample_parameters)
        concentrations.append(concentration)
        surface_coverages.append(surface_coverage)
        film_thicknesses.append(film_thickness)

        gpc.append(sample_gpc)
        top_thickness.append(sample_top)
        bottom_thickness.append(sample_bottom)
        conformality.append(sample_conformality)

        accepted += 1

        if accepted % 100 == 0:
            print(
                f"Generated {accepted}/{NUM_SAMPLES} "
                f"samples"
            )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        parameters=np.asarray(parameters),
        concentration=np.asarray(concentrations),
        surface_coverage=np.asarray(
            surface_coverages
        ),
        film_thickness=np.asarray(
            film_thicknesses
        ),
        gpc=np.asarray(gpc),
        top_thickness=np.asarray(top_thickness),
        bottom_thickness=np.asarray(
            bottom_thickness
        ),
        conformality=np.asarray(
            conformality
        ),
        seed=SEED,
        num_cycles=NUM_CYCLES,
    )

    print("\nDataset generation complete.")
    print(f"Samples generated: {accepted}")
    print(f"Total attempts: {attempts}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
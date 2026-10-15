"""Synthetic parameter-sweep dataset generator for steady-state ALD deposition profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator


def sample_parameters(rng: np.random.Generator) -> Dict[str, float]:
    """Sample kinetic and transport parameters from uniform prior intervals."""
    return {
        "diffusivity": float(rng.uniform(1.0e-3, 4.0e-3)),
        "k_ads": float(rng.uniform(0.5, 1.5)),
        "k_des": float(rng.uniform(0.02, 0.10)),
        "k_rxn": float(rng.uniform(0.2, 0.8)),
        "k_growth": float(rng.uniform(0.01, 0.04)),
    }


def run_simulation(parameters: Dict[str, float]) -> Dict[str, Any]:
    """Instantiate and execute a single continuum simulation instance."""
    simulator = ThinFilmDepositionSimulator(
        diffusivity=parameters["diffusivity"],
        k_ads=parameters["k_ads"],
        k_des=parameters["k_des"],
        k_rxn=parameters["k_rxn"],
        k_growth=parameters["k_growth"],
    )

    simulator.run()

    top_thickness = float(simulator.history["top_thickness"][-1])
    bottom_thickness = float(simulator.history["bottom_thickness"][-1])
    conformality = bottom_thickness / top_thickness if top_thickness > 0.0 else 0.0

    return {
        "concentration": simulator.C.copy(),
        "surface_coverage": simulator.theta.copy(),
        "film_thickness": simulator.h.copy(),
        "top_thickness": top_thickness,
        "bottom_thickness": bottom_thickness,
        "conformality": conformality,
    }


def generate_dataset(
    num_samples: int = 100, seed: int = 42, output_path: str = "data/deposition_dataset.npz"
) -> Path:
    """Generate and serialize an ensemble of steady-state deposition profiles."""
    rng = np.random.default_rng(seed)

    parameters = []
    concentrations = []
    surface_coverages = []
    film_thicknesses = []
    top_thicknesses = []
    bottom_thicknesses = []
    conformalities = []

    print(f"Generating deposition dataset: {num_samples} samples, seed={seed}")

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

        if (index + 1) % 10 == 0 or index == num_samples - 1:
            print(f"[{index + 1:03d}/{num_samples}] conformality={result['conformality']:.4f}")

    target_file = Path(output_path).resolve()
    target_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        target_file,
        parameters=np.asarray(parameters, dtype=np.float64),
        concentration=np.asarray(concentrations, dtype=np.float64),
        surface_coverage=np.asarray(surface_coverages, dtype=np.float64),
        film_thickness=np.asarray(film_thicknesses, dtype=np.float64),
        top_thickness=np.asarray(top_thicknesses, dtype=np.float64),
        bottom_thickness=np.asarray(bottom_thicknesses, dtype=np.float64),
        conformality=np.asarray(conformalities, dtype=np.float64),
    )

    print(f"\nDataset generation completed successfully.")
    print(f"Archived dataset saved to: {target_file}")
    return target_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ALD steady-state deposition dataset.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of simulation samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed.")
    parser.add_argument("--output", default="data/deposition_dataset.npz", help="Output .npz archive path.")
    args = parser.parse_args()

    generate_dataset(num_samples=args.num_samples, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
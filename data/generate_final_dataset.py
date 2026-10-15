"""Dataset generator for steady-state ALD deposition profiles and conformality metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator

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


def sample_pulse_time(rng: np.random.Generator) -> float:
    """Sample precursor pulse duration with bimodal distribution."""
    if rng.random() < 0.85:
        return float(rng.uniform(10.0, 60.0))
    return float(rng.uniform(60.0, 100.0))


def sample_parameters(rng: np.random.Generator) -> np.ndarray:
    """Sample a single physical parameter vector from prescribed parameter bounds."""
    return np.array(
        [
            rng.uniform(*PARAMETER_RANGES["diffusivity"]),
            rng.uniform(*PARAMETER_RANGES["k_ads"]),
            rng.uniform(*PARAMETER_RANGES["k_des"]),
            rng.uniform(*PARAMETER_RANGES["k_rxn"]),
            PARAMETER_RANGES["k_growth"][0],
            sample_pulse_time(rng),
            PARAMETER_RANGES["purge_time"][0],
            PARAMETER_RANGES["reaction_time"][0],
        ],
        dtype=np.float64,
    )


def run_sample(
    parameters: Sequence[float], num_cycles: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]]:
    """Execute a single simulation instance and validate physical invariants."""
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
        num_cycles=num_cycles,
    )

    simulator.run()

    concentration = simulator.C.copy()
    surface_coverage = simulator.theta.copy()
    film_thickness = simulator.h.copy()

    gpc = float(simulator.history["gpc"][-1])
    top_thickness = float(simulator.history["top_thickness"][-1])
    bottom_thickness = float(simulator.history["bottom_thickness"][-1])
    conformality = float(simulator.history["conformality"][-1])

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate steady-state ALD deposition dataset across physical parameter space."
    )
    parser.add_argument("--num-samples", type=int, default=2000, help="Target number of accepted simulation samples.")
    parser.add_argument("--num-cycles", type=int, default=10, help="Number of ALD cycles per sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed.")
    parser.add_argument("--output", default="data/final_dataset.npz", help="Path to destination .npz archive.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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

    print("Executing final ALD dataset generation...")
    print(f"Target sample count: {args.num_samples}")
    print(f"Cycles per sample: {args.num_cycles}")

    while accepted < args.num_samples:
        attempts += 1
        sample = sample_parameters(rng)
        result = run_sample(sample, num_cycles=args.num_cycles)

        if result is None:
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

        parameters.append(sample)
        concentrations.append(concentration)
        surface_coverages.append(surface_coverage)
        film_thicknesses.append(film_thickness)
        gpc.append(sample_gpc)
        top_thickness.append(sample_top)
        bottom_thickness.append(sample_bottom)
        conformality.append(sample_conformality)

        accepted += 1
        if accepted % 100 == 0:
            print(f"Completed {accepted}/{args.num_samples} samples.")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        parameters=np.asarray(parameters, dtype=np.float32),
        concentration=np.asarray(concentrations, dtype=np.float32),
        surface_coverage=np.asarray(surface_coverages, dtype=np.float32),
        film_thickness=np.asarray(film_thicknesses, dtype=np.float32),
        gpc=np.asarray(gpc, dtype=np.float32),
        top_thickness=np.asarray(top_thickness, dtype=np.float32),
        bottom_thickness=np.asarray(bottom_thickness, dtype=np.float32),
        conformality=np.asarray(conformality, dtype=np.float32),
        seed=args.seed,
        num_cycles=args.num_cycles,
    )

    print("\nDataset generation completed successfully.")
    print(f"Accepted samples: {accepted} (total attempts: {attempts})")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
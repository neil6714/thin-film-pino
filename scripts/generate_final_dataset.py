"""Dataset generator for steady-state ALD deposition profiles and conformality metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from thin_film_pino.data.parameters import (
    sample_parameters,
)
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def build_simulator(
    parameters: Sequence[float], num_cycles: int = 10
) -> ThinFilmDepositionSimulator:
    """Instantiate reference CPU simulator configured with designated parameters."""
    return ThinFilmDepositionSimulator(
        diffusivity=float(parameters[0]),
        k_ads=float(parameters[1]),
        k_des=float(parameters[2]),
        k_rxn=float(parameters[3]),
        k_growth=float(parameters[4]),
        pulse_time=float(parameters[5]),
        purge_time=float(parameters[6]),
        reaction_time=float(parameters[7]),
        num_cycles=num_cycles,
    )


def run_sample(
    parameters: Sequence[float], num_cycles: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]]:
    """Execute simulation and return final field snapshots and scalar metrics."""
    try:
        simulator = build_simulator(parameters, num_cycles=num_cycles)
        history = simulator.run()
    except Exception as e:
        print(f"Simulation failed for parameters {parameters}: {e}")
        return None

    c = simulator.C.copy()
    theta = simulator.theta.copy()
    h = simulator.h.copy()

    gpc = float(history["gpc"][-1])
    top_thickness = float(history["top_thickness"][-1])
    bottom_thickness = float(history["bottom_thickness"][-1])
    conformality = float(history["conformality"][-1])

    fields = (c, theta, h)
    scalars = (gpc, top_thickness, bottom_thickness, conformality)

    if not all(np.all(np.isfinite(f)) for f in fields):
        return None
    if not all(np.isfinite(s) for s in scalars):
        return None
    if np.any(c < 0.0) or np.any(theta < 0.0) or np.any(theta > 1.0) or np.any(h < 0.0):
        return None
    if gpc <= 0.0 or top_thickness <= 0.0 or bottom_thickness < 0.0 or conformality < 0.0:
        return None

    return c, theta, h, gpc, top_thickness, bottom_thickness, conformality


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate steady-state ALD deposition profiles and conformality dataset."
    )
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of verified samples.")
    parser.add_argument("--num-cycles", type=int, default=10, help="ALD cycles per trajectory.")
    parser.add_argument("--seed", type=int, default=42, help="Random number generator seed.")
    parser.add_argument(
        "--output", default="data/final_dataset.npz", help="Path to output .npz archive."
    )
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

    attempts = 0
    accepted = 0

    print(f"Generating verified ALD dataset ({args.num_samples} samples, seed={args.seed})...")
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

    print("\nDataset generation completed.")
    print(f"Accepted samples: {accepted} (total attempts: {attempts})")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

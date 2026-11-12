"""Physical parameter definitions and sampling distributions for ALD simulations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

PARAMETER_NAMES: Tuple[str, ...] = (
    "diffusivity",
    "k_ads",
    "k_des",
    "k_rxn",
    "k_growth",
    "pulse_time",
    "purge_time",
    "reaction_time",
)

PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    "diffusivity": (1.0e-3, 8.0e-3),
    "k_ads": (0.5, 1.5),
    "k_des": (0.02, 0.10),
    "k_rxn": (0.1, 0.8),
    "k_growth": (0.02, 0.02),
    "pulse_time": (10.0, 100.0),
    "purge_time": (1.0, 1.0),
    "reaction_time": (1.0, 1.0),
}

PHASE_NAMES: Tuple[str, ...] = (
    "initial",
    "pulse",
    "purge",
    "reaction",
    "post_reaction_purge",
)


def sample_pulse_time(rng: np.random.Generator) -> float:
    """Sample precursor dose time with bimodal weighting favoring short pulses."""
    if rng.random() < 0.85:
        return float(rng.uniform(10.0, 60.0))
    return float(rng.uniform(60.0, 100.0))


def sample_parameters(rng: np.random.Generator) -> np.ndarray:
    """Draw a single 8-dimensional parameter sample from bounded priors."""
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

"""Physical parameter definitions, priors loading, and reproducible sampling distributions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

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

# Varying parameters that carry information to the surrogate network.
# Constant parameters (k_growth, purge_time, reaction_time) are dropped from inputs.
ACTIVE_PARAMETER_NAMES: Tuple[str, ...] = (
    "diffusivity",
    "k_ads",
    "k_des",
    "k_rxn",
    "pulse_time",
)

PHASE_NAMES: Tuple[str, ...] = (
    "initial",
    "pulse",
    "purge",
    "reaction",
    "post_reaction_purge",
)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "priors.yaml"


def load_priors(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load parameter prior distributions from YAML configuration file."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        # Fallback to local relative path if repository root differs
        alt_path = Path("configs/priors.yaml")
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Priors configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def sample_pulse_time(rng: np.random.Generator, pulse_prior: Dict[str, Any]) -> float:
    """Sample precursor dose exposure time from bimodal distribution favoring short pulses."""
    short_fraction = float(pulse_prior.get("short_fraction", 0.85))
    if rng.random() < short_fraction:
        return float(rng.uniform(pulse_prior["short_low"], pulse_prior["short_high"]))
    return float(rng.uniform(pulse_prior["long_low"], pulse_prior["long_high"]))


def sample_parameter_value(rng: np.random.Generator, spec: Dict[str, Any]) -> float:
    """Sample scalar value based on prior specification type."""
    prior_type = spec.get("type", "uniform")
    if prior_type == "constant":
        return float(spec["value"])
    elif prior_type == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    elif prior_type == "bimodal":
        return sample_pulse_time(rng, spec)
    else:
        raise ValueError(f"Unsupported prior distribution type: {prior_type}")


def sample_parameters(
    rng: np.random.Generator,
    priors: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw a single 8-dimensional parameter vector from bounded priors.

    Parameters
    ----------
    rng : np.random.Generator
        Isolated pseudorandom number generator for this sample.
    priors : dict, optional
        Loaded prior specification dictionary. If None, loaded from configs/priors.yaml.

    Returns
    -------
    np.ndarray
        8-element array of shape (8,) with dtype float64 ordered by PARAMETER_NAMES.
    """
    if priors is None:
        priors = load_priors()

    param_specs = priors["parameters"]
    values = [sample_parameter_value(rng, param_specs[name]) for name in PARAMETER_NAMES]
    return np.asarray(values, dtype=np.float64)


def sample_parameters_dict(
    rng: np.random.Generator,
    priors: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Draw a single parameter sample as a dictionary mapping parameter names to floats."""
    arr = sample_parameters(rng, priors=priors)
    return {name: float(arr[i]) for i, name in enumerate(PARAMETER_NAMES)}


def sample_parameter_ensemble(
    num_samples: int,
    seed: int = 42,
    priors: Optional[Dict[str, Any]] = None,
    max_rejection_attempts: int = 1000,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Draw an ensemble of parameter samples using SeedSequence spawning for exact reproducibility.

    Parameters
    ----------
    num_samples : int
        Number of valid parameter samples to draw.
    seed : int, default=42
        Master random seed.
    priors : dict, optional
        Loaded prior specification dictionary.
    max_rejection_attempts : int, default=1000
        Maximum retry attempts before raising RuntimeError.

    Returns
    -------
    parameters : np.ndarray
        Array of shape (num_samples, 8) containing drawn parameters.
    rejection_counts : dict
        Logged rejection statistics by reason.
    """
    if priors is None:
        priors = load_priors()

    seq = np.random.SeedSequence(seed)
    child_seeds = seq.spawn(num_samples)

    samples: List[np.ndarray] = []
    rejection_counts: Dict[str, int] = {}

    for idx, child_seed in enumerate(child_seeds):
        sample_rng = np.random.default_rng(child_seed)
        attempts = 0
        accepted = False

        while not accepted:
            attempts += 1
            if attempts > max_rejection_attempts:
                raise RuntimeError(
                    f"Exceeded max rejection attempts ({max_rejection_attempts}) while sampling sample {idx}."
                )

            param_vec = sample_parameters(sample_rng, priors=priors)

            # Verification of physical admissibility
            # Diffusivity > 0, rates > 0, times > 0
            if np.any(param_vec < 0.0):
                rejection_counts["negative_parameter"] = (
                    rejection_counts.get("negative_parameter", 0) + 1
                )
                continue

            samples.append(param_vec)
            accepted = True

    return np.stack(samples, axis=0), rejection_counts

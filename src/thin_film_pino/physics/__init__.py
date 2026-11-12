"""Physics simulators and stencil operators for continuum atomic layer deposition."""

from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.config import SimulationConfig
from thin_film_pino.physics.geometry import (
    GridGeometry,
    build_geometry,
    compute_stable_dt,
    laplacian_numpy,
    laplacian_torch,
    validate_dt,
)
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator
from thin_film_pino.physics.transient_cuda import TransientALDCudaSimulator

__all__ = [
    "BatchedTransientALDCudaSimulator",
    "GridGeometry",
    "SimulationConfig",
    "ThinFilmDepositionSimulator",
    "TransientALDCudaSimulator",
    "build_geometry",
    "compute_stable_dt",
    "laplacian_numpy",
    "laplacian_torch",
    "validate_dt",
]

"""Batched continuum ALD PyTorch simulator for high-throughput parameter sweeps on GPU/CPU.

Simulates ensembles of 2D transient atomic layer deposition trajectories across
varying kinetic, transport, and pulse timing parameters concurrently using vectorized
finite-difference operators.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np
import torch


class BatchedTransientALDCudaSimulator:
    """Vectorized ensemble ALD simulator implemented in PyTorch with CUDA and CPU support.

    Parameters
    ----------
    parameters : array_like or torch.Tensor
        Batch parameter tensor of shape (batch_size, 8) or (batch_size, 5+) containing:
        - index 0: diffusivity D in m^2/s
        - index 1: adsorption rate constant k_ads in s^-1
        - index 2: desorption rate constant k_des in s^-1
        - index 3: reaction rate constant k_rxn in s^-1
        - index 4: growth coefficient k_growth in m
        - index 5: pulse time in seconds
        - index 6: purge time in seconds
        - index 7: reaction time in seconds
    nx : int, default=80
        Horizontal grid resolution.
    ny : int, default=60
        Vertical grid resolution.
    width : float, default=1.0
        Normalized domain horizontal dimension Lx in meters.
    height : float, default=1.0
        Normalized domain vertical dimension Ly in meters.
    trench_left : float, default=0.375
        Normalized horizontal coordinate of trench left sidewall.
    trench_right : float, default=0.625
        Normalized horizontal coordinate of trench right sidewall.
    trench_top : float, default=0.50
        Normalized vertical coordinate of trench opening mouth.
    trench_bottom : float, default=0.95
        Normalized vertical coordinate of trench floor.
    device : str or torch.device, default="cuda"
        Target execution device ("cuda" or "cpu"). Automatically falls back
        to CPU if CUDA runtime is unavailable.
    """

    def __init__(
        self,
        parameters: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        nx: int = 80,
        ny: int = 60,
        width: float = 1.0,
        height: float = 1.0,
        trench_left: float = 0.375,
        trench_right: float = 0.625,
        trench_top: float = 0.50,
        trench_bottom: float = 0.95,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        target_device = torch.device(device) if isinstance(device, str) else device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = target_device

        self.parameters = torch.as_tensor(parameters, dtype=torch.float32, device=self.device)
        if self.parameters.ndim == 1:
            self.parameters = self.parameters.unsqueeze(0)

        self.batch_size = int(self.parameters.shape[0])
        self.nx = int(nx)
        self.ny = int(ny)
        self.width = float(width)
        self.height = float(height)
        self.trench_left = float(trench_left)
        self.trench_right = float(trench_right)
        self.trench_top = float(trench_top)
        self.trench_bottom = float(trench_bottom)

        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)

        self.diffusivity = self.parameters[:, 0, None, None]
        self.k_ads = self.parameters[:, 1, None, None]
        self.k_des = self.parameters[:, 2, None, None]
        self.k_rxn = self.parameters[:, 3, None, None]
        self.k_growth = self.parameters[:, 4, None, None]

        if self.parameters.shape[1] >= 8:
            self.pulse_time = self.parameters[:, 5]
            self.purge_time = float(self.parameters[0, 6].item())
            self.reaction_time = float(self.parameters[0, 7].item())
        else:
            self.pulse_time = torch.full((self.batch_size,), 1.0, dtype=torch.float32, device=self.device)
            self.purge_time = 1.0
            self.reaction_time = 1.0

        max_diffusivity = float(self.parameters[:, 0].max().item())
        max_rate = float(self.parameters[:, 1:4].max().item())

        diffusion_dt = 0.20 * min(self.dx, self.dy) ** 2 / max(max_diffusivity, 1e-12)
        reaction_dt = 0.20 / max(max_rate, 1e-12)
        self.dt = min(diffusion_dt, reaction_dt)

        diffusion_stability_limit = min(self.dx, self.dy) ** 2 / (4.0 * max(max_diffusivity, 1e-12))
        if self.dt >= diffusion_stability_limit:
            raise ValueError(
                f"Time step dt={self.dt:.4e} exceeds batched 2D diffusive CFL stability "
                f"limit of {diffusion_stability_limit:.4e}."
            )

        x = torch.linspace(0.0, self.width, self.nx, device=self.device)
        y = torch.linspace(0.0, self.height, self.ny, device=self.device)
        self.x = x
        self.y = y
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")

        self.solid_mask = self._build_geometry()
        self.gas_mask = ~self.solid_mask
        self.surface_mask = self._find_surface_cells()

        self.C = torch.zeros((self.batch_size, self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.theta = torch.zeros_like(self.C)
        self.h = torch.zeros_like(self.C)

    def _build_geometry(self) -> torch.Tensor:
        substrate = self.Y >= self.trench_bottom
        outside_trench = (self.Y >= self.trench_top) & (
            (self.X < self.trench_left) | (self.X > self.trench_right)
        )
        return substrate | outside_trench

    def _find_surface_cells(self) -> torch.Tensor:
        neighbors = torch.zeros_like(self.solid_mask)
        neighbors[1:] |= self.solid_mask[:-1]
        neighbors[:-1] |= self.solid_mask[1:]
        neighbors[:, 1:] |= self.solid_mask[:, :-1]
        neighbors[:, :-1] |= self.solid_mask[:, 1:]
        return self.gas_mask & neighbors

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        up = field.clone()
        down = field.clone()
        left = field.clone()
        right = field.clone()

        up[:, 1:] = torch.where(self.gas_mask[:-1], field[:, :-1], field[:, 1:])
        down[:, :-1] = torch.where(self.gas_mask[1:], field[:, 1:], field[:, :-1])
        left[:, :, 1:] = torch.where(self.gas_mask[:, :-1], field[:, :, :-1], field[:, :, 1:])
        right[:, :, :-1] = torch.where(self.gas_mask[:, 1:], field[:, :, 1:], field[:, :, :-1])

        laplacian = (up + down - 2.0 * field) / self.dy**2 + (left + right - 2.0 * field) / self.dx**2
        return torch.where(self.gas_mask, laplacian, torch.zeros_like(laplacian))

    def transport_step(self, active: torch.Tensor, pulse: bool) -> None:
        """Advance batch diffusion and surface Langmuir kinetics by dt.

        Parameters
        ----------
        active : torch.Tensor
            1D boolean mask of shape (batch_size,) indicating trajectories currently active.
        pulse : bool
            True if precursor dose is applied at boundary y = 0; False for purge stage.
        """
        active_t = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        diffusion = self.diffusivity * self._laplacian(self.C)
        adsorption = torch.zeros_like(self.C)
        desorption = torch.zeros_like(self.C)

        if pulse:
            adsorption[:, self.surface_mask] = (
                self.k_ads[:, 0, 0, None]
                * self.C[:, self.surface_mask]
                * (1.0 - self.theta[:, self.surface_mask])
            )
            desorption[:, self.surface_mask] = self.k_des[:, 0, 0, None] * self.theta[:, self.surface_mask]

        new_c = torch.clamp(self.C + self.dt * (diffusion - adsorption), min=0.0)
        new_c = torch.where(self.gas_mask, new_c, torch.zeros_like(new_c))
        active_3d = active_t[:, None, None]
        self.C = torch.where(active_3d, new_c, self.C)

        if pulse:
            new_theta = torch.clamp(self.theta + self.dt * (adsorption - desorption), 0.0, 1.0)
            self.theta = torch.where(active_3d, new_theta, self.theta)
            self.C[active_t, 0, :] = 1.0
        else:
            self.C[active_t, 0, :] = 0.0

    def reaction_step(self) -> None:
        """Advance self-limiting co-reactant surface reaction and film growth by dt across batch."""
        reaction = torch.zeros_like(self.theta)
        reaction[:, self.surface_mask] = self.k_rxn[:, 0, 0, None] * self.theta[:, self.surface_mask]
        self.h = torch.clamp(self.h + self.dt * self.k_growth * reaction, min=0.0)
        self.theta = torch.clamp(self.theta - self.dt * reaction, 0.0, 1.0)

    def phase_steps(self, duration: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute the number of discrete simulation time steps for a given duration.

        Parameters
        ----------
        duration : float or torch.Tensor
            Duration in seconds.

        Returns
        -------
        torch.Tensor
            Integer step counts of shape (batch_size,).
        """
        value = torch.as_tensor(duration, dtype=torch.float32, device=self.device)
        return torch.clamp(torch.ceil(value / self.dt).to(torch.int64), min=1)

    def pulse_steps(self) -> torch.Tensor:
        """Return integer time step counts for precursor dose stage."""
        return self.phase_steps(self.pulse_time)

    def fixed_steps(self, duration: float) -> int:
        """Compute fixed integer step count for uniform duration stages."""
        return int(max(1, math.ceil(duration / self.dt)))

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract host NumPy arrays of concentration, coverage, and thickness across batch.

        Returns
        -------
        tuple of np.ndarray
            (concentration, surface_coverage, film_thickness) on host memory.
        """
        return tuple(field.detach().cpu().numpy().astype(np.float32) for field in (self.C, self.theta, self.h))

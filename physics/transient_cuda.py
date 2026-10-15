"""Transient atomic layer deposition (ALD) PyTorch continuum simulator with GPU/CPU support.

Discretizes 2D continuum transport and surface reaction kinetics in high-aspect-ratio
features using an explicit finite-difference time-domain formulation. Adheres strictly
to Courant-Friedrichs-Lewy (CFL) diffusive stability criteria and Langmuir adsorption
kinetics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch


class TransientALDCudaSimulator:
    """Continuum ALD simulator implemented in PyTorch for GPU and CPU execution.

    Simulates transient precursor mass transport, Langmuir self-limiting surface
    adsorption, purge desorption, and co-reactant surface reaction across 2D
    microelectronic feature geometries.

    Parameters
    ----------
    diffusivity : float
        Knudsen/binary precursor gas diffusion coefficient D in m^2/s.
    k_ads : float
        First-order adsorption rate constant k_ads in s^-1.
    k_des : float
        First-order thermal desorption rate constant k_des in s^-1.
    k_rxn : float
        Co-reactant surface reaction rate constant k_rxn in s^-1.
    k_growth : float
        Growth-per-cycle thickness conversion coefficient k_growth in m.
    pulse_time : float
        Precursor dose exposure time in seconds.
    purge_time : float
        Purge duration in seconds.
    reaction_time : float
        Co-reactant pulse/reaction duration in seconds.
    num_cycles : int
        Total number of cyclic ALD exposures.
    nx : int, default=160
        Spatial grid resolution along the horizontal x-axis.
    ny : int, default=120
        Spatial grid resolution along the vertical y-axis.
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
    inlet_concentration : float, default=1.0
        Normalized precursor concentration C_0 at reactor boundary (y = 0).
    device : str or torch.device, default="cuda"
        Target execution device ("cuda" or "cpu"). Automatically falls back
        to CPU if CUDA runtime is unavailable.
    """

    def __init__(
        self,
        diffusivity: float,
        k_ads: float,
        k_des: float,
        k_rxn: float,
        k_growth: float,
        pulse_time: float,
        purge_time: float,
        reaction_time: float,
        num_cycles: int,
        nx: int = 160,
        ny: int = 120,
        width: float = 1.0,
        height: float = 1.0,
        trench_left: float = 0.375,
        trench_right: float = 0.625,
        trench_top: float = 0.50,
        trench_bottom: float = 0.95,
        inlet_concentration: float = 1.0,
        device: str | torch.device = "cuda",
    ) -> None:
        target_device = torch.device(device) if isinstance(device, str) else device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = target_device

        self.nx = int(nx)
        self.ny = int(ny)
        self.width = float(width)
        self.height = float(height)
        self.trench_left = float(trench_left)
        self.trench_right = float(trench_right)
        self.trench_top = float(trench_top)
        self.trench_bottom = float(trench_bottom)
        self.diffusivity = float(diffusivity)
        self.k_ads = float(k_ads)
        self.k_des = float(k_des)
        self.k_rxn = float(k_rxn)
        self.k_growth = float(k_growth)
        self.inlet_concentration = float(inlet_concentration)
        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.num_cycles = int(num_cycles)

        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)

        diffusion_dt = 0.20 * min(self.dx, self.dy) ** 2 / self.diffusivity
        reaction_dt = 0.20 / max(self.k_ads, self.k_des, self.k_rxn, 1e-12)
        self.dt = min(diffusion_dt, reaction_dt)

        diffusion_stability_limit = min(self.dx, self.dy) ** 2 / (4.0 * self.diffusivity)
        if self.dt >= diffusion_stability_limit:
            raise ValueError(
                f"Time step dt={self.dt:.4e} violates explicit 2D diffusive CFL stability "
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
        self.top_region = self.surface_mask & (
            self.Y < self.trench_top + 2.5 * self.dy
        )
        self.bottom_region = self.surface_mask & (
            self.Y > self.trench_bottom - 2.5 * self.dy
        )

        self.C = torch.zeros((self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.theta = torch.zeros((self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.h = torch.zeros((self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.history: Dict[str, List[float]] = self._empty_history()

    def _empty_history(self) -> Dict[str, List[float]]:
        return {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "surface_coverage": [],
        }

    def _build_geometry(self) -> torch.Tensor:
        substrate = self.Y >= self.trench_bottom
        outside_trench = (self.Y >= self.trench_top) & (
            (self.X < self.trench_left) | (self.X > self.trench_right)
        )
        return substrate | outside_trench

    def _find_surface_cells(self) -> torch.Tensor:
        neighbors = torch.zeros_like(self.solid_mask)
        neighbors[1:, :] |= self.solid_mask[:-1, :]
        neighbors[:-1, :] |= self.solid_mask[1:, :]
        neighbors[:, 1:] |= self.solid_mask[:, :-1]
        neighbors[:, :-1] |= self.solid_mask[:, 1:]
        return self.gas_mask & neighbors

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        up = field.clone()
        down = field.clone()
        left = field.clone()
        right = field.clone()

        up[1:, :] = torch.where(
            self.gas_mask[:-1, :],
            field[:-1, :],
            field[1:, :],
        )
        down[:-1, :] = torch.where(
            self.gas_mask[1:, :],
            field[1:, :],
            field[:-1, :],
        )
        left[:, 1:] = torch.where(
            self.gas_mask[:, :-1],
            field[:, :-1],
            field[:, 1:],
        )
        right[:, :-1] = torch.where(
            self.gas_mask[:, 1:],
            field[:, 1:],
            field[:, :-1],
        )

        laplacian = (
            (up + down - 2.0 * field) / self.dy**2
            + (left + right - 2.0 * field) / self.dx**2
        )
        return torch.where(self.gas_mask, laplacian, torch.zeros_like(laplacian))

    def _num_steps(self, duration: float) -> int:
        return max(1, math.ceil(duration / self.dt))

    def transport_step(self, pulse: bool) -> None:
        """Advance transient diffusion and surface adsorption/desorption by dt.

        Parameters
        ----------
        pulse : bool
            True if precursor dose is active at inlet boundary; False during purge.
        """
        diffusion = self.diffusivity * self._laplacian(self.C)
        adsorption = torch.zeros_like(self.C)
        desorption = torch.zeros_like(self.theta)

        if pulse:
            adsorption[self.surface_mask] = (
                self.k_ads
                * self.C[self.surface_mask]
                * (1.0 - self.theta[self.surface_mask])
            )
            desorption[self.surface_mask] = self.k_des * self.theta[self.surface_mask]

        self.C = torch.clamp(self.C + self.dt * (diffusion - adsorption), min=0.0)
        self.C = torch.where(self.gas_mask, self.C, torch.zeros_like(self.C))

        if pulse:
            self.theta = torch.clamp(
                self.theta + self.dt * (adsorption - desorption),
                min=0.0,
                max=1.0,
            )
            self.C[0, :] = self.inlet_concentration
        else:
            self.C[0, :] = 0.0

    def reaction_step(self) -> None:
        """Advance self-limiting surface co-reactant reaction and film growth by dt."""
        reaction = torch.zeros_like(self.theta)
        reaction[self.surface_mask] = self.k_rxn * self.theta[self.surface_mask]
        self.h = torch.clamp(self.h + self.dt * self.k_growth * reaction, min=0.0)
        self.theta = torch.clamp(self.theta - self.dt * reaction, min=0.0, max=1.0)

    def record_cycle_metrics(self, cycle: int, previous_thickness: float) -> float:
        """Compute and log cycle-level summary metrics.

        Parameters
        ----------
        cycle : int
            Current ALD cycle index.
        previous_thickness : float
            Mean thickness at conclusion of prior cycle.

        Returns
        -------
        float
            Updated mean surface thickness.
        """
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]
        top_thickness = float(self.h[self.top_region].mean().item()) if torch.any(self.top_region) else 0.0
        bottom_thickness = float(self.h[self.bottom_region].mean().item()) if torch.any(self.bottom_region) else 0.0
        mean_thickness = float(surface_h.mean().item()) if surface_h.numel() > 0 else 0.0
        mean_coverage = float(surface_theta.mean().item()) if surface_theta.numel() > 0 else 0.0
        conformality = bottom_thickness / top_thickness if top_thickness > 0.0 else 0.0
        gpc = mean_thickness - previous_thickness

        self.history["cycle"].append(float(cycle))
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(mean_thickness)
        self.history["top_thickness"].append(top_thickness)
        self.history["bottom_thickness"].append(bottom_thickness)
        self.history["conformality"].append(conformality)
        self.history["surface_coverage"].append(mean_coverage)
        return mean_thickness

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract host NumPy arrays of current concentration, coverage, and thickness fields.

        Returns
        -------
        tuple of np.ndarray
            (concentration, surface_coverage, film_thickness) on host memory.
        """
        return (
            self.C.detach().cpu().numpy().astype(np.float32),
            self.theta.detach().cpu().numpy().astype(np.float32),
            self.h.detach().cpu().numpy().astype(np.float32),
        )

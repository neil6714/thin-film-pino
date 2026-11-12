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
    c0 : float, default=1.0
        Inlet precursor molar concentration in mol/m^3.
    post_purge_time : float, default=1.0
        Post-reaction purge duration in seconds.
    device : str or torch.device, default="cuda"
        Target execution device ("cuda" or "cpu"). Falls back to CPU if unavailable.
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
        c0: float = 1.0,
        post_purge_time: float = 1.0,
        device: str | torch.device = "cuda",
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.width = float(width)
        self.height = float(height)

        self.diffusivity = float(diffusivity)
        self.k_ads = float(k_ads)
        self.k_des = float(k_des)
        self.k_rxn = float(k_rxn)
        self.k_growth = float(k_growth)
        self.c0 = float(c0)

        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.post_purge_time = float(post_purge_time)
        self.num_cycles = int(num_cycles)

        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)
        self.inv_dx2 = 1.0 / (self.dx * self.dx)
        self.inv_dy2 = 1.0 / (self.dy * self.dy)

        self.x = torch.linspace(0.0, self.width, self.nx, device=self.device, dtype=torch.float32)
        self.y = torch.linspace(0.0, self.height, self.ny, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(self.y, self.x, indexing="ij")

        left = (grid_x < trench_left) & (grid_y >= trench_top)
        right = (grid_x > trench_right) & (grid_y >= trench_top)
        bottom = grid_y > trench_bottom
        self.solid_mask = left | right | bottom

        padded_solid = torch.nn.functional.pad(
            self.solid_mask.to(dtype=torch.float32), (1, 1, 1, 1), mode="constant", value=0.0
        )
        has_solid_neighbor = (
            padded_solid[1:-1, :-2]
            + padded_solid[1:-1, 2:]
            + padded_solid[:-2, 1:-1]
            + padded_solid[2:, 1:-1]
        ) > 0.0
        self.surface_mask = (~self.solid_mask) & has_solid_neighbor

        self.top_region = self.surface_mask & (grid_y <= trench_top + 2.0 * self.dy)
        self.bottom_region = self.surface_mask & (grid_y >= trench_bottom - 2.0 * self.dy)

        min_dim = min(self.dx, self.dy)
        dt_diff = 0.20 * (min_dim * min_dim) / max(self.diffusivity, 1e-12)
        dt_kin = 0.20 / max(self.k_ads, self.k_des, self.k_rxn, 1e-12)
        self.dt = min(dt_diff, dt_kin)

        cfl_threshold = (min_dim * min_dim) / (4.0 * max(self.diffusivity, 1e-12))
        if self.dt >= cfl_threshold:
            raise ValueError(
                f"Integration time step dt={self.dt:.3e} violates diffusive CFL limit {cfl_threshold:.3e}"
            )

        self.C = torch.zeros((self.ny, self.nx), device=self.device, dtype=torch.float32)
        self.theta = torch.zeros((self.ny, self.nx), device=self.device, dtype=torch.float32)
        self.h = torch.zeros((self.ny, self.nx), device=self.device, dtype=torch.float32)

        self.history: Dict[str, List[float]] = {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "surface_coverage": [],
        }

    def apply_boundary_conditions(self, c_tensor: torch.Tensor, inlet_c: float) -> None:
        """Enforce inlet Dirichlet and zero-flux Neumann boundary conditions."""
        c_tensor[0, ~self.solid_mask[0, :]] = inlet_c
        c_tensor[self.solid_mask] = 0.0
        c_tensor[:, 0] = c_tensor[:, 1]
        c_tensor[:, -1] = c_tensor[:, -2]
        c_tensor[-1, :] = c_tensor[-2, :]

    def step_diffusion(self, dt: float, inlet_c: float) -> None:
        """Advance transient diffusion using 2D central differences."""
        self.apply_boundary_conditions(self.C, inlet_c)
        c_mid = self.C[1:-1, 1:-1]
        d2x = (self.C[1:-1, 2:] - 2.0 * c_mid + self.C[1:-1, :-2]) * self.inv_dx2
        d2y = (self.C[2:, 1:-1] - 2.0 * c_mid + self.C[:-2, 1:-1]) * self.inv_dy2
        laplacian = d2x + d2y
        self.C[1:-1, 1:-1] = torch.clamp(c_mid + dt * self.diffusivity * laplacian, min=0.0)
        self.apply_boundary_conditions(self.C, inlet_c)

    def advance_pulse_step(self, dt: float) -> None:
        """Advance one integration time step of the precursor pulse stage."""
        self.step_diffusion(dt, self.c0)
        c_surf = self.C[self.surface_mask]
        theta_surf = self.theta[self.surface_mask]

        r_ads = self.k_ads * c_surf * (1.0 - theta_surf)
        r_des = self.k_des * theta_surf
        d_theta = (r_ads - r_des) * dt

        self.theta[self.surface_mask] = torch.clamp(theta_surf + d_theta, 0.0, 1.0)
        self.C[self.surface_mask] = torch.clamp(c_surf - r_ads * dt, min=0.0)

    def advance_purge_step(self, dt: float) -> None:
        """Advance one integration time step of inert gas purge with desorption."""
        self.step_diffusion(dt, 0.0)
        theta_surf = self.theta[self.surface_mask]
        r_des = self.k_des * theta_surf
        self.theta[self.surface_mask] = torch.clamp(theta_surf - r_des * dt, 0.0, 1.0)

    def advance_reaction_step(self, dt: float) -> None:
        """Advance one integration time step of co-reactant exposure and surface growth."""
        theta_surf = self.theta[self.surface_mask]
        d_theta = self.k_rxn * theta_surf * dt
        self.theta[self.surface_mask] = torch.clamp(theta_surf - d_theta, 0.0, 1.0)
        self.h[self.surface_mask] += self.k_growth * d_theta

    def run_pulse_stage(self, duration: float | None = None) -> None:
        """Execute complete precursor exposure stage."""
        stage_time = self.pulse_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        sub_dt = stage_time / steps
        for _ in range(steps):
            self.advance_pulse_step(sub_dt)

    def run_purge_stage(self, duration: float | None = None) -> None:
        """Execute complete inert purge stage."""
        stage_time = self.purge_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        sub_dt = stage_time / steps
        self.C[0, :] = 0.0
        for _ in range(steps):
            self.advance_purge_step(sub_dt)
        self.C.zero_()

    def run_reaction_stage(self, duration: float | None = None) -> None:
        """Execute complete co-reactant exposure stage."""
        stage_time = self.reaction_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        sub_dt = stage_time / steps
        for _ in range(steps):
            self.advance_reaction_step(sub_dt)

    def run_cycle(self) -> None:
        """Execute all four stages of a single ALD cycle."""
        self.run_pulse_stage()
        self.run_purge_stage()
        self.run_reaction_stage()
        self.run_purge_stage(duration=self.post_purge_time)

    def record_cycle_metrics(self, cycle: int, previous_thickness: float) -> float:
        """Compute and log cycle-level summary metrics."""
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]
        top_thickness = (
            float(self.h[self.top_region].mean().item()) if torch.any(self.top_region) else 0.0
        )
        bottom_thickness = (
            float(self.h[self.bottom_region].mean().item())
            if torch.any(self.bottom_region)
            else 0.0
        )
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
        """Extract host NumPy arrays of current concentration, coverage, and thickness fields."""
        return (
            self.C.detach().cpu().numpy().astype(np.float32),
            self.theta.detach().cpu().numpy().astype(np.float32),
            self.h.detach().cpu().numpy().astype(np.float32),
        )

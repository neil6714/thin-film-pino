"""Batched continuum ALD PyTorch simulator for high-throughput parameter sweeps on GPU/CPU.

Simulates ensembles of 2D transient atomic layer deposition trajectories across
varying kinetic, transport, and pulse timing parameters concurrently using vectorized
finite-difference operators.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from thin_film_pino.physics.geometry import (
    build_geometry,
    compute_stable_dt,
    laplacian_torch,
    validate_dt,
)


class BatchedTransientALDCudaSimulator:
    """Vectorized ensemble ALD simulator implemented in PyTorch with CUDA and CPU support.

    Parameters
    ----------
    parameters : array_like or torch.Tensor
        Batch parameter tensor of shape (batch_size, 8) or (batch_size, 5+) containing:
        - index 0: diffusivity D (nondimensional)
        - index 1: adsorption rate constant k_ads (1/time)
        - index 2: desorption rate constant k_des (1/time)
        - index 3: reaction rate constant k_rxn (1/time)
        - index 4: growth coefficient k_growth (thickness per converted coverage)
        - index 5: pulse time (time)
        - index 6: purge time (time)
        - index 7: reaction time (time)
    nx : int, default=80
        Horizontal grid resolution.
    ny : int, default=60
        Vertical grid resolution.
    lx : float, default=1.0
        Domain horizontal dimension Lx.
    ly : float, default=1.0
        Domain vertical dimension Ly.
    trench_left : float, default=0.375
        Horizontal position of left trench sidewall.
    trench_right : float, default=0.625
        Horizontal position of right trench sidewall.
    trench_top : float, default=0.50
        Vertical position of trench opening mouth.
    trench_bottom : float, default=0.95
        Vertical position of trench floor.
    inlet_concentration : float, default=1.0
        Inlet boundary precursor concentration at y = 0.
    post_purge_time : float, default=1.0
        Post-reaction purge duration.
    dt : float, optional
        Integration time step. If None, computed from stability criteria.
    dtype : torch.dtype or str, default=torch.float32
        Precision for floating-point calculations (float32 or float64).
    device : str or torch.device, default="cuda"
        Execution device ("cuda" or "cpu"). Falls back to CPU if unavailable.
    """

    def __init__(
        self,
        parameters: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        nx: int = 80,
        ny: int = 60,
        lx: float = 1.0,
        ly: float = 1.0,
        trench_left: float = 0.375,
        trench_right: float = 0.625,
        trench_top: float = 0.50,
        trench_bottom: float = 0.95,
        inlet_concentration: float = 1.0,
        post_purge_time: float = 1.0,
        dt: Optional[float] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        target_device = torch.device(device) if isinstance(device, str) else device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = target_device

        if isinstance(dtype, str):
            torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        else:
            torch_dtype = dtype
        self.dtype = torch_dtype

        self.parameters = torch.as_tensor(parameters, dtype=self.dtype, device=self.device)
        if self.parameters.ndim == 1:
            self.parameters = self.parameters.unsqueeze(0)

        self.batch_size = int(self.parameters.shape[0])
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.inlet_concentration = float(inlet_concentration)
        self.post_purge_time = float(post_purge_time)

        # Build shared geometry
        geom = build_geometry(
            nx=self.nx,
            ny=self.ny,
            lx=self.lx,
            ly=self.ly,
            trench_left=trench_left,
            trench_right=trench_right,
            trench_top=trench_top,
            trench_bottom=trench_bottom,
        )
        self.dx = geom.dx
        self.dy = geom.dy
        self.x = torch.as_tensor(geom.x, dtype=self.dtype, device=self.device)
        self.y = torch.as_tensor(geom.y, dtype=self.dtype, device=self.device)

        # 3D masks of shape (1, ny, nx) for broadcasting across batch
        self.solid_mask = torch.as_tensor(
            geom.solid_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        self.gas_mask = torch.as_tensor(
            geom.gas_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        self.surface_mask = torch.as_tensor(
            geom.surface_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        self.field_mask = torch.as_tensor(
            geom.field_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        self.sidewall_mask = torch.as_tensor(
            geom.sidewall_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        self.floor_mask = torch.as_tensor(
            geom.floor_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        # Kinetic parameters shaped (batch_size, 1, 1) for spatial broadcast
        self.diffusivity = self.parameters[:, 0, None, None]
        self.k_ads = self.parameters[:, 1, None, None]
        self.k_des = self.parameters[:, 2, None, None]
        self.k_rxn = self.parameters[:, 3, None, None]
        self.k_growth = self.parameters[:, 4, None, None]

        if self.parameters.shape[1] >= 8:
            self.pulse_time = self.parameters[:, 5]
            purge_times = self.parameters[:, 6]
            reaction_times = self.parameters[:, 7]

            # Require constant purge and reaction durations across batch
            if not torch.allclose(purge_times, purge_times[0]):
                raise ValueError(
                    "Batched simulator requires identical purge_time across all samples."
                )
            if not torch.allclose(reaction_times, reaction_times[0]):
                raise ValueError(
                    "Batched simulator requires identical reaction_time across all samples."
                )

            self.purge_time = float(purge_times[0].item())
            self.reaction_time = float(reaction_times[0].item())
        else:
            self.pulse_time = torch.full(
                (self.batch_size,), 1.0, dtype=self.dtype, device=self.device
            )
            self.purge_time = 1.0
            self.reaction_time = 1.0

        # Time step determination: compute from single sample or fixed prior bounds, never batch
        if dt is not None:
            max_d = float(self.parameters[:, 0].max().item())
            max_rate = (
                float(self.parameters[:, 1:4].max().item())
                if self.parameters.shape[1] >= 4
                else 1.0
            )
            validate_dt(dt, self.dx, self.dy, max_d, max_rate, max_rate, max_rate)
            self.dt = float(dt)
        else:
            # Fixed upper bounds from prior: D_max = 8.0e-3, k_ads_max = 1.5, k_des_max = 0.10, k_rxn_max = 0.80
            self.dt = compute_stable_dt(
                self.dx,
                self.dy,
                8.0e-3,
                1.5,
                0.10,
                0.80,
            )

        self.C = torch.zeros(
            (self.batch_size, self.ny, self.nx), dtype=self.dtype, device=self.device
        )
        self.theta = torch.zeros_like(self.C)
        self.dose_theta = torch.zeros_like(self.C)
        self.h = torch.zeros_like(self.C)

        self.clamp_counts: Dict[str, int] = {
            "c_negative": 0,
            "theta_negative": 0,
            "theta_exceeded": 0,
        }

        self.history: Dict[str, List[float]] = {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "field_thickness": [],
            "floor_thickness": [],
            "sidewall_thickness": [],
            "step_coverage": [],
            "conformality": [],
            "dose_coverage": [],
            "surface_coverage": [],
        }

    @property
    def clamp_stats(self) -> Dict[str, int]:
        """Return total numerical clamp activations recorded during execution."""
        return dict(self.clamp_counts)

    def apply_boundary_conditions(self, c_tensor: torch.Tensor, inlet_mask: bool = True) -> None:
        """Enforce inlet Dirichlet and zero-flux Neumann boundary conditions."""
        gas_inlet = self.gas_mask[0, 0, :]
        if inlet_mask:
            c_tensor[:, 0, gas_inlet] = self.inlet_concentration
        else:
            c_tensor[:, 0, :] = 0.0

        c_tensor[:, :, 0] = c_tensor[:, :, 1]
        c_tensor[:, :, -1] = c_tensor[:, :, -2]
        c_tensor[:, -1, :] = c_tensor[:, -2, :]
        c_tensor.masked_fill_(self.solid_mask, 0.0)

    def step_diffusion(self, dt: float, inlet_mask: bool = True) -> None:
        """Advance transient diffusion using 2D central differences with zero-flux boundaries."""
        self.apply_boundary_conditions(self.C, inlet_mask=inlet_mask)
        lap = laplacian_torch(self.C, self.dx, self.dy, self.gas_mask)
        new_c = self.C + dt * self.diffusivity * lap
        self.apply_boundary_conditions(new_c, inlet_mask=inlet_mask)

        neg = (new_c.masked_select(self.gas_mask) < 0.0).sum().item()
        if neg > 0:
            self.clamp_counts["c_negative"] += int(neg)

        self.C = torch.clamp(new_c, min=0.0)

    def transport_step(self, active: torch.Tensor, pulse: bool, dt: Optional[float] = None) -> None:
        """Advance batch diffusion and surface Langmuir kinetics by dt."""
        step_dt = self.dt if dt is None else float(dt)
        active_t = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self.apply_boundary_conditions(self.C, inlet_mask=pulse)
        lap = laplacian_torch(self.C, self.dx, self.dy, self.gas_mask)
        diffusion = self.diffusivity * lap

        adsorption = torch.zeros_like(self.C)
        desorption = torch.zeros_like(self.C)

        surf_3d = self.surface_mask.expand_as(self.C)
        if pulse:
            c_surf = self.C.masked_select(surf_3d).reshape(self.batch_size, -1)
            theta_surf = self.theta.masked_select(surf_3d).reshape(self.batch_size, -1)

            ads_surf = self.k_ads[:, 0, :] * c_surf * (1.0 - theta_surf)
            des_surf = self.k_des[:, 0, :] * theta_surf

            adsorption.masked_scatter_(surf_3d, ads_surf)
            desorption.masked_scatter_(surf_3d, des_surf)

        new_c = self.C + step_dt * (diffusion - adsorption)
        neg_c = (new_c.masked_select(self.gas_mask.expand_as(self.C)) < 0.0).sum().item()
        if neg_c > 0:
            self.clamp_counts["c_negative"] += int(neg_c)

        new_c = torch.clamp(new_c, min=0.0)
        new_c = torch.where(self.gas_mask, new_c, torch.zeros_like(new_c))

        active_3d = active_t[:, None, None]
        self.C = torch.where(active_3d, new_c, self.C)

        if pulse:
            new_theta = self.theta + step_dt * (adsorption - desorption)
            neg_theta = (new_theta.masked_select(surf_3d) < 0.0).sum().item()
            exc_theta = (new_theta.masked_select(surf_3d) > 1.0).sum().item()
            if neg_theta > 0:
                self.clamp_counts["theta_negative"] += int(neg_theta)
            if exc_theta > 0:
                self.clamp_counts["theta_exceeded"] += int(exc_theta)

            new_theta = torch.clamp(new_theta, 0.0, 1.0)
            self.theta = torch.where(active_3d, new_theta, self.theta)

        self.apply_boundary_conditions(self.C, inlet_mask=pulse)

    def reaction_step(self, dt: Optional[float] = None) -> None:
        """Advance self-limiting co-reactant surface reaction and film growth by dt across batch."""
        step_dt = self.dt if dt is None else float(dt)
        surf_3d = self.surface_mask.expand_as(self.theta)
        theta_surf = self.theta.masked_select(surf_3d).reshape(self.batch_size, -1)

        d_theta = step_dt * self.k_rxn[:, 0, :] * theta_surf
        new_theta_surf = theta_surf - d_theta

        neg = (new_theta_surf < 0.0).sum().item()
        if neg > 0:
            self.clamp_counts["theta_negative"] += int(neg)

        new_theta_clamped = torch.clamp(new_theta_surf, 0.0, 1.0)
        self.theta.masked_scatter_(surf_3d, new_theta_clamped)

        h_surf = self.h.masked_select(surf_3d).reshape(self.batch_size, -1)
        new_h_surf = h_surf + self.k_growth[:, 0, :] * d_theta
        self.h.masked_scatter_(surf_3d, new_h_surf)

    def phase_steps(self, duration: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute the number of discrete simulation time steps for a given duration."""
        value = torch.as_tensor(duration, dtype=self.dtype, device=self.device)
        return torch.clamp(torch.ceil(value / self.dt).to(torch.int64), min=1)

    def pulse_steps(self) -> torch.Tensor:
        """Return integer time step counts for precursor dose stage."""
        return self.phase_steps(self.pulse_time)

    def fixed_steps(self, duration: float) -> int:
        """Compute fixed integer step count for uniform duration stages."""
        return int(max(1, math.ceil(duration / self.dt)))

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract host NumPy arrays of concentration, coverage, and thickness across batch."""
        return tuple(field.detach().cpu().numpy().copy() for field in (self.C, self.theta, self.h))

    def record_dose_coverage(self) -> np.ndarray:
        """Return mean saturation coverage at dose stage conclusion across batch."""
        self.dose_theta = self.theta.clone()
        surf_3d = self.surface_mask.expand_as(self.theta)
        cov_surf = self.theta.masked_select(surf_3d).reshape(self.batch_size, -1)
        return cov_surf.mean(dim=1).detach().cpu().numpy()

    def record_cycle_metrics(
        self, cycle: int, previous_thickness: np.ndarray, dose_cov: np.ndarray
    ) -> np.ndarray:
        """Compute and record cycle metrics across batch."""
        surf_3d = self.surface_mask.expand_as(self.h)
        field_3d = self.field_mask.expand_as(self.h)
        floor_3d = self.floor_mask.expand_as(self.h)
        sidewall_3d = self.sidewall_mask.expand_as(self.h)

        h_surf = self.h.masked_select(surf_3d).reshape(self.batch_size, -1)
        h_field = self.h.masked_select(field_3d).reshape(self.batch_size, -1)
        h_floor = self.h.masked_select(floor_3d).reshape(self.batch_size, -1)
        h_sidewall = self.h.masked_select(sidewall_3d).reshape(self.batch_size, -1)

        theta_surf = self.theta.masked_select(surf_3d).reshape(self.batch_size, -1)

        mean_h = h_surf.mean(dim=1).detach().cpu().numpy()
        mean_field = h_field.mean(dim=1).detach().cpu().numpy()
        mean_floor = h_floor.mean(dim=1).detach().cpu().numpy()
        mean_sidewall = h_sidewall.mean(dim=1).detach().cpu().numpy()
        mean_cov = theta_surf.mean(dim=1).detach().cpu().numpy()

        step_coverage = np.where(mean_field > 0.0, mean_floor / mean_field, 0.0)
        gpc = mean_h - previous_thickness

        self.history["cycle"].append(float(cycle))
        self.history["gpc"].append(float(np.mean(gpc)))
        self.history["mean_thickness"].append(float(np.mean(mean_h)))
        self.history["field_thickness"].append(float(np.mean(mean_field)))
        self.history["floor_thickness"].append(float(np.mean(mean_floor)))
        self.history["sidewall_thickness"].append(float(np.mean(mean_sidewall)))
        self.history["step_coverage"].append(float(np.mean(step_coverage)))
        self.history["conformality"].append(float(np.mean(step_coverage)))
        self.history["dose_coverage"].append(float(np.mean(dose_cov)))
        self.history["surface_coverage"].append(float(np.mean(mean_cov)))

        return mean_h

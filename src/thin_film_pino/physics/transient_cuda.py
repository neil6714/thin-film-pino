"""Transient atomic layer deposition (ALD) PyTorch simulator for single trajectories.

Thin single-sample wrapper around BatchedTransientALDCudaSimulator to provide
a unified execution path and eliminate duplicate geometry and stencil code.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator


class TransientALDCudaSimulator:
    """Single-trajectory continuum ALD simulator wrapping BatchedTransientALDCudaSimulator.

    Parameters
    ----------
    diffusivity : float
        Precursor gas diffusion coefficient D (nondimensional).
    k_ads : float
        First-order adsorption rate constant (1/time).
    k_des : float
        Thermal desorption rate constant (1/time).
    k_rxn : float
        Co-reactant surface reaction rate constant (1/time).
    k_growth : float
        Thickness conversion coefficient per unit converted coverage (length).
    pulse_time : float
        Precursor dose exposure time.
    purge_time : float
        Inert purge duration.
    reaction_time : float
        Co-reactant reaction duration.
    num_cycles : int
        Total ALD cycles to execute.
    nx : int, default=160
        Horizontal grid resolution.
    ny : int, default=120
        Vertical grid resolution.
    width : float, default=1.0
        Normalized domain horizontal dimension Lx.
    height : float, default=1.0
        Normalized domain vertical dimension Ly.
    trench_left : float, default=0.375
        Normalized horizontal coordinate of left trench sidewall.
    trench_right : float, default=0.625
        Normalized horizontal coordinate of right trench sidewall.
    trench_top : float, default=0.50
        Normalized vertical coordinate of trench opening mouth.
    trench_bottom : float, default=0.95
        Normalized vertical coordinate of trench floor.
    c0 : float, default=1.0
        Inlet precursor concentration.
    post_purge_time : float, default=1.0
        Post-reaction purge duration.
    dt : float, optional
        Integration time step. If None, computed from stability limits.
    dtype : str, default="float32"
        Computation precision ('float32' or 'float64').
    device : str or torch.device, default="cuda"
        Execution device ("cuda" or "cpu"). Falls back to CPU if unavailable.
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
        dt: Optional[float] = None,
        dtype: str = "float32",
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.num_cycles = int(num_cycles)
        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.post_purge_time = float(post_purge_time)
        self.width = float(width)
        self.height = float(height)

        params = np.array(
            [[diffusivity, k_ads, k_des, k_rxn, k_growth, pulse_time, purge_time, reaction_time]],
            dtype=np.float64,
        )

        self._batched = BatchedTransientALDCudaSimulator(
            parameters=params,
            nx=nx,
            ny=ny,
            lx=width,
            ly=height,
            trench_left=trench_left,
            trench_right=trench_right,
            trench_top=trench_top,
            trench_bottom=trench_bottom,
            inlet_concentration=c0,
            post_purge_time=post_purge_time,
            dt=dt,
            dtype=dtype,
            device=device,
        )

        self.device = self._batched.device
        self.dt = self._batched.dt
        self.dx = self._batched.dx
        self.dy = self._batched.dy
        self.nx = self._batched.nx
        self.ny = self._batched.ny
        self.diffusivity = float(diffusivity)
        self.k_ads = float(k_ads)
        self.k_des = float(k_des)
        self.k_rxn = float(k_rxn)
        self.k_growth = float(k_growth)
        self.c0 = float(c0)

        self._active = torch.ones(1, dtype=torch.bool, device=self.device)

    @property
    def C(self) -> torch.Tensor:
        """2D concentration field tensor."""
        return self._batched.C[0]

    @C.setter
    def C(self, val: torch.Tensor) -> None:
        self._batched.C[0] = val

    @property
    def theta(self) -> torch.Tensor:
        """2D fractional surface coverage tensor."""
        return self._batched.theta[0]

    @theta.setter
    def theta(self, val: torch.Tensor) -> None:
        self._batched.theta[0] = val

    @property
    def h(self) -> torch.Tensor:
        """2D deposited film thickness tensor."""
        return self._batched.h[0]

    @h.setter
    def h(self, val: torch.Tensor) -> None:
        self._batched.h[0] = val

    @property
    def solid_mask(self) -> torch.Tensor:
        """2D boolean solid substrate mask."""
        return self._batched.solid_mask[0]

    @property
    def gas_mask(self) -> torch.Tensor:
        """2D boolean gas domain mask."""
        return self._batched.gas_mask[0]

    @property
    def surface_mask(self) -> torch.Tensor:
        """2D boolean reacting surface mask."""
        return self._batched.surface_mask[0]

    @property
    def field_mask(self) -> torch.Tensor:
        """2D boolean planar wafer top field mask."""
        return self._batched.field_mask[0]

    @property
    def floor_mask(self) -> torch.Tensor:
        """2D boolean trench bottom floor mask."""
        return self._batched.floor_mask[0]

    @property
    def sidewall_mask(self) -> torch.Tensor:
        """2D boolean trench sidewall mask."""
        return self._batched.sidewall_mask[0]

    @property
    def top_region(self) -> torch.Tensor:
        """Alias to field_mask for backwards compatibility."""
        return self._batched.field_mask[0]

    @property
    def bottom_region(self) -> torch.Tensor:
        """Alias to floor_mask for backwards compatibility."""
        return self._batched.floor_mask[0]

    @property
    def clamp_stats(self) -> Dict[str, int]:
        """Numerical clamping counts."""
        return self._batched.clamp_stats

    @property
    def history(self) -> Dict[str, List[float]]:
        """Recorded cycle metrics."""
        return self._batched.history

    def advance_pulse_step(self, dt: Optional[float] = None) -> None:
        """Advance one integration time step of precursor dose."""
        self._batched.transport_step(self._active, pulse=True, dt=dt)

    def advance_purge_step(self, dt: Optional[float] = None) -> None:
        """Advance one integration time step of inert gas purge."""
        self._batched.transport_step(self._active, pulse=False, dt=dt)

    def advance_reaction_step(self, dt: Optional[float] = None) -> None:
        """Advance one integration time step of co-reactant exposure and surface growth."""
        self._batched.reaction_step(dt=dt)

    def run_pulse_stage(self, duration: Optional[float] = None) -> None:
        """Execute complete precursor dose stage."""
        stage_time = self.pulse_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        for _ in range(steps):
            self.advance_pulse_step()

    def run_purge_stage(self, duration: Optional[float] = None) -> None:
        """Execute complete inert purge stage."""
        stage_time = self.purge_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        self.C[0, :] = 0.0
        for _ in range(steps):
            self.advance_purge_step()
        self._batched.C.zero_()

    def run_reaction_stage(self, duration: Optional[float] = None) -> None:
        """Execute complete co-reactant exposure stage."""
        stage_time = self.reaction_time if duration is None else duration
        steps = max(1, math.ceil(stage_time / self.dt))
        for _ in range(steps):
            self.advance_reaction_step()

    def run_cycle(self) -> None:
        """Execute single four-phase ALD cycle."""
        self.run_pulse_stage()
        self.run_purge_stage()
        self.run_reaction_stage()
        self.run_purge_stage(duration=self.post_purge_time)

    def record_cycle_metrics(self, cycle: int, previous_thickness: float) -> float:
        """Record metrics for completed cycle and return current mean thickness."""
        dose_cov = self._batched.record_dose_coverage()
        prev = np.array([previous_thickness], dtype=np.float32)
        mean_h = self._batched.record_cycle_metrics(cycle, prev, dose_cov)
        return float(mean_h[0])

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract host NumPy arrays of current 2D concentration, coverage, and thickness."""
        c, theta, h = self._batched.snapshot()
        return c[0], theta[0], h[0]

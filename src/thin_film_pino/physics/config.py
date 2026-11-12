"""Configuration dataclass for continuum ALD simulation parameters and geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuration holding numerical, geometric, kinetic, and scheduling parameters.

    Attributes
    ----------
    nx : int
        Horizontal spatial grid resolution. Default is 160.
    ny : int
        Vertical spatial grid resolution. Default is 120.
    lx : float
        Nondimensional domain width Lx. Default is 1.0.
    ly : float
        Nondimensional domain height Ly. Default is 1.0.
    trench_left : float
        Horizontal position of left trench sidewall. Default is 0.375.
    trench_right : float
        Horizontal position of right trench sidewall. Default is 0.625.
    trench_top : float
        Vertical position of trench opening mouth. Default is 0.50.
    trench_bottom : float
        Vertical position of trench floor. Default is 0.95.
    aspect_ratio : Optional[float]
        Target depth-to-width ratio of trench. If specified, overrides trench width
        to match target depth / width while keeping trench centered.
    diffusivity : float
        Nondimensional diffusion coefficient D. Default is 5.0e-3.
    k_ads : float
        First-order adsorption rate constant (1/time). Default is 1.0.
    k_des : float
        Thermal desorption rate constant (1/time). Default is 0.05.
    k_rxn : float
        Co-reactant surface reaction rate constant (1/time). Default is 0.4.
    k_growth : float
        Thickness conversion factor per unit converted coverage (length). Default is 0.02.
    inlet_concentration : float
        Nondimensional precursor concentration at inlet boundary y = 0. Default is 1.0.
    pulse_time : float
        Precursor dose duration. Default is 1.0.
    purge_time : float
        Inert gas purge duration. Default is 1.0.
    reaction_time : float
        Co-reactant reaction stage duration. Default is 1.0.
    post_purge_time : float
        Post-reaction purge duration. Default is 1.0.
    num_cycles : int
        Total ALD cycles to execute. Default is 10.
    dt : Optional[float]
        Discretized time step. If None, computed from diffusive and kinetic stability limits.
    dtype : str
        Floating-point precision ('float32' or 'float64'). Default is 'float32'.
    device : str
        Execution device ('cpu' or 'cuda'). Default is 'cpu'.
    """

    nx: int = 160
    ny: int = 120
    lx: float = 1.0
    ly: float = 1.0

    trench_left: float = 0.375
    trench_right: float = 0.625
    trench_top: float = 0.50
    trench_bottom: float = 0.95
    aspect_ratio: Optional[float] = None

    diffusivity: float = 5.0e-3
    k_ads: float = 1.0
    k_des: float = 0.05
    k_rxn: float = 0.4
    k_growth: float = 0.02
    inlet_concentration: float = 1.0

    pulse_time: float = 1.0
    purge_time: float = 1.0
    reaction_time: float = 1.0
    post_purge_time: float = 1.0
    num_cycles: int = 10

    dt: Optional[float] = None
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.aspect_ratio is not None and self.aspect_ratio > 0.0:
            depth = self.trench_bottom - self.trench_top
            target_width = depth / self.aspect_ratio
            mid_x = 0.5 * self.lx
            self.trench_left = max(0.0, mid_x - 0.5 * target_width)
            self.trench_right = min(self.lx, mid_x + 0.5 * target_width)

    @classmethod
    def high_aspect_ratio_preset(cls, aspect_ratio: float = 10.0, **kwargs) -> SimulationConfig:
        """Create a configuration preset with a designated high-aspect-ratio trench."""
        depth = 0.45
        width = depth / aspect_ratio
        left = 0.5 - 0.5 * width
        right = 0.5 + 0.5 * width
        return cls(
            aspect_ratio=aspect_ratio,
            trench_left=left,
            trench_right=right,
            trench_top=0.50,
            trench_bottom=0.95,
            **kwargs,
        )

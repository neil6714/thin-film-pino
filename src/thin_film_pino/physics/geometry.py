"""Shared geometry, region partitioning, and stencil operators for ALD simulation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from thin_film_pino.physics.stencil import laplacian_numpy, laplacian_torch


class GridGeometry(NamedTuple):
    """Container for grid coordinates, resolution, and spatial domain masks."""

    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    solid_mask: np.ndarray
    gas_mask: np.ndarray
    surface_mask: np.ndarray
    field_mask: np.ndarray
    sidewall_mask: np.ndarray
    floor_mask: np.ndarray


def build_geometry(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    trench_left: float = 0.375,
    trench_right: float = 0.625,
    trench_top: float = 0.50,
    trench_bottom: float = 0.95,
) -> GridGeometry:
    """Construct discrete 2D grid and explicit region masks for trench simulation.

    Parameters
    ----------
    nx : int
        Number of grid nodes along horizontal x-axis.
    ny : int
        Number of grid nodes along vertical y-axis.
    lx : float, default=1.0
        Total domain horizontal length.
    ly : float, default=1.0
        Total domain vertical length.
    trench_left : float, default=0.375
        Horizontal position of left trench sidewall.
    trench_right : float, default=0.625
        Horizontal position of right trench sidewall.
    trench_top : float, default=0.50
        Vertical position of planar wafer top / trench opening.
    trench_bottom : float, default=0.95
        Vertical position of trench floor.

    Returns
    -------
    GridGeometry
        Container with coordinates, steps, solid, gas, surface, and region masks.
    """
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    grid_x, grid_y = np.meshgrid(x, y)

    # Substrate walls: left bank, right bank, and floor
    left_bank = (grid_x < trench_left) & (grid_y >= trench_top)
    right_bank = (grid_x > trench_right) & (grid_y >= trench_top)
    floor_substrate = grid_y > trench_bottom
    solid_mask = left_bank | right_bank | floor_substrate
    gas_mask = ~solid_mask

    # Find surface cells: gas cells with at least one 4-connected solid neighbor
    padded_solid = np.pad(solid_mask, 1, mode="constant", constant_values=False)
    has_solid_neighbor = (
        padded_solid[1:-1, :-2]
        | padded_solid[1:-1, 2:]
        | padded_solid[:-2, 1:-1]
        | padded_solid[2:, 1:-1]
    )
    surface_mask = gas_mask & has_solid_neighbor

    # Partition surface into disjoint regions: field, floor, sidewall
    field_mask = (
        surface_mask
        & (grid_y <= trench_top + 0.5 * dy)
        & ((grid_x < trench_left) | (grid_x > trench_right))
    )
    floor_mask = (
        surface_mask
        & (grid_y >= trench_bottom - 1.5 * dy)
        & (grid_x >= trench_left - 0.5 * dx)
        & (grid_x <= trench_right + 0.5 * dx)
    )
    sidewall_mask = surface_mask & ~field_mask & ~floor_mask

    return GridGeometry(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        solid_mask=solid_mask,
        gas_mask=gas_mask,
        surface_mask=surface_mask,
        field_mask=field_mask,
        sidewall_mask=sidewall_mask,
        floor_mask=floor_mask,
    )


def compute_stable_dt(
    dx: float,
    dy: float,
    diffusivity: float,
    k_ads: float,
    k_des: float,
    k_rxn: float,
    cfl_factor: float = 0.20,
) -> float:
    """Calculate explicit FTCS stable time step bounded by diffusion and kinetics."""
    min_dim = min(dx, dy)
    dt_diff = cfl_factor * (min_dim**2) / max(diffusivity, 1e-12)
    max_rate = max(k_ads, k_des, k_rxn, 1e-12)
    dt_kin = cfl_factor / max_rate
    return min(dt_diff, dt_kin)


def validate_dt(
    dt: float,
    dx: float,
    dy: float,
    diffusivity: float,
    k_ads: float,
    k_des: float,
    k_rxn: float,
) -> None:
    """Check user-supplied time step against diffusive and reaction stability criteria."""
    min_dim = min(dx, dy)
    cfl_threshold = (min_dim**2) / (4.0 * max(diffusivity, 1e-12))
    if dt >= cfl_threshold:
        raise ValueError(
            f"Supplied dt={dt:.4e} s violates diffusive von Neumann stability limit ({cfl_threshold:.4e} s)."
        )
    max_rate = max(k_ads, k_des, k_rxn, 1e-12)
    kinetic_limit = 1.0 / max_rate
    if dt > kinetic_limit:
        raise ValueError(
            f"Supplied dt={dt:.4e} s exceeds reaction kinetic stability limit ({kinetic_limit:.4e} s)."
        )


__all__ = [
    "GridGeometry",
    "build_geometry",
    "compute_stable_dt",
    "validate_dt",
    "laplacian_numpy",
    "laplacian_torch",
]

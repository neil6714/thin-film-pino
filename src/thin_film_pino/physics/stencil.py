"""Spatial stencil operators with boundary condition reflection for continuum ALD simulation."""

from __future__ import annotations

import numpy as np
import torch


def laplacian_numpy(
    field: np.ndarray,
    dx: float,
    dy: float,
    gas_mask: np.ndarray,
) -> np.ndarray:
    """Compute 2D discrete Laplacian with zero-flux Neumann conditions at solid boundaries.

    Parameters
    ----------
    field : np.ndarray
        2D scalar field array shaped (ny, nx) or (B, ny, nx).
    dx : float
        Grid spacing in horizontal direction.
    dy : float
        Grid spacing in vertical direction.
    gas_mask : np.ndarray
        Boolean indicator array where True represents accessible gas phase.

    Returns
    -------
    np.ndarray
        Discrete Laplacian evaluated on active gas nodes, zeroed in solid regions.
    """
    up = field.copy()
    down = field.copy()
    left = field.copy()
    right = field.copy()

    if field.ndim == 2:
        up[1:, :] = np.where(gas_mask[:-1, :], field[:-1, :], field[1:, :])
        down[:-1, :] = np.where(gas_mask[1:, :], field[1:, :], field[:-1, :])
        left[:, 1:] = np.where(gas_mask[:, :-1], field[:, :-1], field[:, 1:])
        right[:, :-1] = np.where(gas_mask[:, 1:], field[:, 1:], field[:, :-1])
    elif field.ndim == 3:
        g = gas_mask if gas_mask.ndim == 3 else gas_mask[None]
        up[:, 1:, :] = np.where(g[:, :-1, :], field[:, :-1, :], field[:, 1:, :])
        down[:, :-1, :] = np.where(g[:, 1:, :], field[..., 1:, :], field[..., :-1, :])
        left[:, :, 1:] = np.where(g[:, :, :-1], field[:, :, :-1], field[:, :, 1:])
        right[:, :, :-1] = np.where(g[:, :, 1:], field[:, :, 1:], field[:, :, :-1])
    else:
        raise ValueError(f"field must be 2D or 3D, got {field.ndim}D.")

    lap = (up + down - 2.0 * field) / (dy**2) + (left + right - 2.0 * field) / (dx**2)
    return np.where(gas_mask, lap, 0.0)


def laplacian_torch(
    field: torch.Tensor,
    dx: float,
    dy: float,
    gas_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D discrete Laplacian for PyTorch tensors with zero-flux solid reflection.

    Parameters
    ----------
    field : torch.Tensor
        Scalar field tensor shaped (B, ny, nx) or (B, T, ny, nx) or (ny, nx).
    dx : float
        Grid spacing in horizontal direction.
    dy : float
        Grid spacing in vertical direction.
    gas_mask : torch.Tensor
        Boolean indicator tensor where True represents gas phase.

    Returns
    -------
    torch.Tensor
        Discrete Laplacian matching field shape, evaluated on active gas domain.
    """
    orig_ndim = field.ndim
    if orig_ndim == 2:
        f = field.unsqueeze(0)
    elif orig_ndim == 4:
        # Flatten leading dimensions (B, T) -> (B * T, ny, nx)
        b, t, ny, nx = field.shape
        f = field.reshape(b * t, ny, nx)
    else:
        f = field

    g = gas_mask
    if g.ndim == 2:
        g = g.unsqueeze(0).expand(f.shape[0], -1, -1)
    elif g.ndim == 3 and g.shape[0] == 1:
        g = g.expand(f.shape[0], -1, -1)

    up = f.clone()
    down = f.clone()
    left = f.clone()
    right = f.clone()

    up[:, 1:, :] = torch.where(g[:, :-1, :], f[:, :-1, :], f[:, 1:, :])
    down[:, :-1, :] = torch.where(g[:, 1:, :], f[:, 1:, :], f[:, :-1, :])
    left[:, :, 1:] = torch.where(g[:, :, :-1], f[:, :, :-1], f[:, :, 1:])
    right[:, :, :-1] = torch.where(g[:, :, 1:], f[:, :, 1:], f[:, :, :-1])

    lap = (up + down - 2.0 * f) / (dy**2) + (left + right - 2.0 * f) / (dx**2)
    lap = torch.where(g, lap, torch.zeros_like(lap))

    if orig_ndim == 2:
        return lap.squeeze(0)
    if orig_ndim == 4:
        return lap.reshape(b, t, ny, nx)
    return lap

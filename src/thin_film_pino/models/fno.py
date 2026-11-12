"""Fourier Neural Operator architectures for transient field dynamics in trench geometries."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer parameterizing integral kernels in Fourier space.

    Operates on both positive and negative frequencies along the non-Hermitian y-axis
    and non-negative frequencies along the Hermitian-symmetric x-axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        modes: Union[int, Tuple[int, int]] = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        if isinstance(modes, int):
            self.modes_y = modes
            self.modes_x = modes
        else:
            self.modes_y, self.modes_x = modes

        scale = 1.0 / (self.in_channels * self.out_channels)
        # Weight block 1: positive y-frequencies (rows 0 to modes_y - 1)
        self.weight1 = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes_y,
                self.modes_x,
                dtype=torch.cfloat,
            )
        )
        # Weight block 2: negative y-frequencies (rows from -modes_y to -1)
        self.weight2 = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                self.modes_y,
                self.modes_x,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution via 2D real FFT and inverse real FFT."""
        *batch_dims, _, height, width = x.shape
        transformed = torch.fft.rfft2(x)

        output = torch.zeros(
            *batch_dims,
            self.out_channels,
            height,
            transformed.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )

        effective_modes_y = min(self.modes_y, height // 2)
        effective_modes_x = min(self.modes_x, transformed.shape[-1])

        # Multiply positive y-frequencies
        output[..., :effective_modes_y, :effective_modes_x] = torch.einsum(
            "...ihw,iohw->...ohw",
            transformed[..., :effective_modes_y, :effective_modes_x],
            self.weight1[:, :, :effective_modes_y, :effective_modes_x],
        )

        # Multiply negative y-frequencies
        output[..., -effective_modes_y:, :effective_modes_x] = torch.einsum(
            "...ihw,iohw->...ohw",
            transformed[..., -effective_modes_y:, :effective_modes_x],
            self.weight2[:, :, :effective_modes_y, :effective_modes_x],
        )

        return torch.fft.irfft2(output, s=(height, width))


class FourierBlock(nn.Module):
    """Spectral convolution block with pointwise linear bypass and GELU activation."""

    def __init__(self, width: int, modes: Union[int, Tuple[int, int]] = 12) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class TemporalFourierOperator(nn.Module):
    """Temporal Fourier neural operator mapping parameters, time, phase, and geometry to fields.

    Each trajectory snapshot at time t is predicted independently given physical parameters p,
    continuous time t, discrete phase state (one-hot), cycle fraction, spatial coordinates,
    and a domain geometry mask.
    """

    def __init__(
        self,
        parameter_dim: int = 5,
        spatial_channels: int = 3,
        num_phases: int = 5,
        include_cycle: bool = True,
        output_channels: int = 3,
        width: int = 32,
        modes: Union[int, Tuple[int, int]] = 12,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.parameter_dim = parameter_dim
        self.spatial_channels = spatial_channels
        self.num_phases = num_phases
        self.include_cycle = include_cycle
        self.output_channels = output_channels
        self.width = width
        self.modes = modes
        self.num_blocks = num_blocks

        # Input channels: parameters + continuous time + phase one-hot + cycle fraction + spatial channels
        scalar_channels = parameter_dim + 1 + num_phases + (1 if include_cycle else 0)
        in_channels = scalar_channels + spatial_channels

        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([FourierBlock(width, modes) for _ in range(num_blocks)])
        self.projection = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, output_channels, kernel_size=1),
        )

    def forward(
        self,
        parameters: torch.Tensor,
        time: torch.Tensor,
        coordinates: torch.Tensor,
        phase: torch.Tensor | None = None,
        cycle_fraction: torch.Tensor | None = None,
        geometry_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict field snapshots independently across batch and time dimensions.

        Parameters
        ----------
        parameters : torch.Tensor
            Batch physical parameters shaped (B, P).
        time : torch.Tensor
            Time values shaped (B, T) or (B, T, 1).
        coordinates : torch.Tensor
            Normalized grid coordinates shaped (C_coord, H, W) where C_coord >= 2.
        phase : torch.Tensor, optional
            Phase indices shaped (B, T) or one-hot (B, T, num_phases).
        cycle_fraction : torch.Tensor, optional
            Continuous cycle fraction in [0, 1] shaped (B, T) or (B, T, 1).
        geometry_mask : torch.Tensor, optional
            Spatial domain indicator mask shaped (1, H, W) or (H, W).

        Returns
        -------
        torch.Tensor
            Predicted physical fields shaped (B, T, output_channels, H, W).
        """
        batch_size = parameters.shape[0]
        if time.ndim == 1:
            time = time.unsqueeze(0).expand(batch_size, -1)
        time_steps = time.shape[1]

        spatial_grid = coordinates
        if spatial_grid.ndim == 4 and spatial_grid.shape[0] == 1:
            spatial_grid = spatial_grid.squeeze(0)
        height, width = spatial_grid.shape[-2:]

        # Concatenate geometry mask to coordinates if provided
        if geometry_mask is not None and spatial_grid.shape[0] == 2:
            if geometry_mask.ndim == 2:
                g_mask = geometry_mask.unsqueeze(0)
            elif geometry_mask.ndim == 3:
                g_mask = geometry_mask
            else:
                g_mask = geometry_mask.squeeze(0)
            spatial_grid = torch.cat([spatial_grid, g_mask], dim=0)

        # Pad spatial channels if fewer than expected
        if spatial_grid.shape[0] < self.spatial_channels:
            pad_channels = self.spatial_channels - spatial_grid.shape[0]
            pad = torch.zeros(
                pad_channels,
                height,
                width,
                device=spatial_grid.device,
                dtype=spatial_grid.dtype,
            )
            spatial_grid = torch.cat([spatial_grid, pad], dim=0)

        # 1. Physical parameters: (B, P) -> (B, T, P)
        p_expanded = parameters.unsqueeze(1).expand(-1, time_steps, -1)

        # 2. Continuous time: (B, T, 1)
        t_expanded = time.unsqueeze(-1) if time.ndim == 2 else time

        conditioning_list = [p_expanded, t_expanded]

        # 3. Discrete phase one-hot: (B, T, num_phases)
        if self.num_phases > 0:
            if phase is not None:
                if phase.ndim == 1:
                    phase = phase.unsqueeze(0).expand(batch_size, -1)
                if phase.ndim == 3 and phase.shape[-1] == self.num_phases:
                    phase_one_hot = phase.float()
                else:
                    phase_ids = phase.long().clamp(0, self.num_phases - 1)
                    phase_one_hot = F.one_hot(phase_ids, num_classes=self.num_phases).float()
            else:
                phase_one_hot = torch.zeros(
                    batch_size,
                    time_steps,
                    self.num_phases,
                    device=parameters.device,
                    dtype=parameters.dtype,
                )
            conditioning_list.append(phase_one_hot)

        # 4. Cycle fraction: (B, T, 1)
        if self.include_cycle:
            if cycle_fraction is not None:
                if cycle_fraction.ndim == 1:
                    cycle_fraction = cycle_fraction.unsqueeze(0).expand(batch_size, -1)
                c_frac = (
                    cycle_fraction.unsqueeze(-1) if cycle_fraction.ndim == 2 else cycle_fraction
                )
            else:
                c_frac = torch.zeros(
                    batch_size,
                    time_steps,
                    1,
                    device=parameters.device,
                    dtype=parameters.dtype,
                )
            conditioning_list.append(c_frac)

        # Concatenate scalar conditionings: (B, T, scalar_channels)
        scalars = torch.cat(conditioning_list, dim=-1)

        # Spatially broadcast scalar features: (B * T, scalar_channels, H, W)
        scalars = (
            scalars.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, height, width)
            .reshape(batch_size * time_steps, -1, height, width)
        )

        # Spatial grid features: (B * T, spatial_channels, H, W)
        spatial_expanded = spatial_grid.unsqueeze(0).expand(batch_size * time_steps, -1, -1, -1)

        # Full input representation: (B * T, in_channels, H, W)
        x = torch.cat([scalars, spatial_expanded], dim=1)

        # Independent snapshot evaluation through lifting, Fourier blocks, and projection
        h = self.lifting(x)
        for block in self.blocks:
            h = block(h)
        out = self.projection(h)

        return out.reshape(batch_size, time_steps, -1, height, width)

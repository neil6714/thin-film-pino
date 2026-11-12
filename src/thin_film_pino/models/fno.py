"""Fourier Neural Operator architectures for transient field dynamics."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer operating in Fourier space."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        transformed = torch.fft.rfft2(x)
        output = torch.zeros_like(transformed)
        modes_y = min(self.modes, height)
        modes_x = min(self.modes, transformed.shape[-1])
        output[:, :, :modes_y, :modes_x] = torch.einsum(
            "bihw,iohw->bohw",
            transformed[:, :, :modes_y, :modes_x],
            self.weight[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(output, s=(height, width))


class FourierBlock(nn.Module):
    """Spectral convolution block with bypass projection and GELU activation."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class TemporalFourierOperator(nn.Module):
    """Temporal Fourier neural operator mapping parameters and continuous time to field states."""

    def __init__(
        self,
        parameter_dim: int = 8,
        spatial_channels: int = 2,
        output_channels: int = 3,
        width: int = 32,
        modes: int = 12,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        in_channels = parameter_dim + 1 + spatial_channels
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
    ) -> torch.Tensor:
        batch_size, time_steps = time.shape[:2]
        height, width = coordinates.shape[-2:]

        p_expanded = parameters.unsqueeze(1).expand(-1, time_steps, -1)
        t_expanded = time.unsqueeze(-1)
        conditioned = torch.cat([p_expanded, t_expanded], dim=-1)

        conditioned = (
            conditioned.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, height, width)
            .reshape(batch_size * time_steps, -1, height, width)
        )
        coords = coordinates.unsqueeze(0).expand(batch_size * time_steps, -1, -1, -1)
        x = torch.cat([conditioned, coords], dim=1)

        h = self.lifting(x)
        for block in self.blocks:
            h = block(h)
        out = self.projection(h)
        return out.reshape(batch_size, time_steps, -1, height, width)

"""Unit tests for SpectralConv2d, TemporalFourierOperator conditioning, and training sanity checks."""

from __future__ import annotations

import torch

from thin_film_pino.models.fno import SpectralConv2d, TemporalFourierOperator
from thin_film_pino.training.train import loss_and_metrics


def test_spectral_conv2d_properties() -> None:
    """Verify that SpectralConv2d is real-valued, shape-preserving, and translation equivariant."""
    torch.manual_seed(42)
    channels = 4
    height, width = 20, 16
    modes = 6

    conv = SpectralConv2d(in_channels=channels, out_channels=channels, modes=modes)
    x = torch.randn(2, channels, height, width, dtype=torch.float32)

    out = conv(x)

    # 1. Real-valued
    assert torch.is_floating_point(out), "SpectralConv2d output must be real floating-point."
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}."

    # 2. Shape-preserving
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}."

    # 3. Translation equivariant for periodic input (circular shift)
    shift_y, shift_x = 3, 5
    x_shifted = torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))
    out_shifted_input = conv(x_shifted)
    expected_shifted_out = torch.roll(out, shifts=(shift_y, shift_x), dims=(-2, -1))

    diff = (out_shifted_input - expected_shifted_out).abs().max().item()
    assert diff < 1.0e-5, f"Translation equivariance violation: max difference {diff:.3e}."


def test_temporal_fourier_operator_conditioning() -> None:
    """Verify TemporalFourierOperator forward pass with parameter, phase, cycle, and mask inputs."""
    torch.manual_seed(42)
    batch_size = 2
    time_steps = 8
    height, width = 20, 15
    param_dim = 5
    num_phases = 5

    model = TemporalFourierOperator(
        parameter_dim=param_dim,
        spatial_channels=3,
        num_phases=num_phases,
        include_cycle=True,
        output_channels=3,
        width=16,
        modes=6,
        num_blocks=2,
    )

    parameters = torch.randn(batch_size, param_dim)
    time = torch.linspace(0.0, 1.0, time_steps).unsqueeze(0).expand(batch_size, -1)
    coordinates = torch.randn(2, height, width)
    geometry_mask = torch.ones(1, height, width)
    phase = torch.randint(0, num_phases, (batch_size, time_steps))
    cycle_fraction = torch.linspace(0.0, 1.0, time_steps).unsqueeze(0).expand(batch_size, -1)

    out = model(
        parameters=parameters,
        time=time,
        coordinates=coordinates,
        phase=phase,
        cycle_fraction=cycle_fraction,
        geometry_mask=geometry_mask,
    )

    assert out.shape == (batch_size, time_steps, 3, height, width)
    assert not torch.isnan(out).any(), "Output contains NaN values."


def test_independent_snapshot_prediction() -> None:
    """Verify that each snapshot is evaluated independently of other time indices."""
    torch.manual_seed(42)
    model = TemporalFourierOperator(
        parameter_dim=5,
        spatial_channels=3,
        num_phases=5,
        include_cycle=True,
        output_channels=3,
        width=16,
        modes=6,
        num_blocks=2,
    )
    model.eval()

    p = torch.randn(1, 5)
    coords = torch.randn(3, 10, 8)

    # Two snapshots evaluated together
    t_pair = torch.tensor([[0.2, 0.8]])
    phase_pair = torch.tensor([[1, 3]])
    cycle_pair = torch.tensor([[0.1, 0.4]])

    with torch.no_grad():
        out_pair = model(p, t_pair, coords, phase=phase_pair, cycle_fraction=cycle_pair)

        # Single snapshot 0 evaluated alone
        out_single_0 = model(
            p,
            t_pair[:, :1],
            coords,
            phase=phase_pair[:, :1],
            cycle_fraction=cycle_pair[:, :1],
        )

    # Snapshot 0 must be identical whether evaluated alone or alongside snapshot 1
    diff = (out_pair[:, 0] - out_single_0[:, 0]).abs().max().item()
    assert diff < 1.0e-5, f"Independent snapshot prediction violation: max diff {diff:.3e}."


def test_overfit_one_batch() -> None:
    """Sanity test verifying that the operator network can overfit a single batch to low loss."""
    torch.manual_seed(42)
    batch_size = 2
    time_steps = 4
    height, width = 12, 10
    param_dim = 5

    model = TemporalFourierOperator(
        parameter_dim=param_dim,
        spatial_channels=3,
        num_phases=5,
        include_cycle=True,
        output_channels=3,
        width=24,
        modes=4,
        num_blocks=2,
    )

    parameters = torch.randn(batch_size, param_dim)
    time = torch.linspace(0.0, 1.0, time_steps).unsqueeze(0).expand(batch_size, -1)
    coords = torch.randn(3, height, width)
    phase = torch.randint(0, 5, (batch_size, time_steps))
    cycle_frac = torch.rand(batch_size, time_steps)
    target = torch.randn(batch_size, time_steps, 3, height, width)
    mask = torch.ones(3, height, width)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=1.0e-5)

    initial_loss = None
    final_loss = None

    model.train()
    for step in range(60):
        optimizer.zero_grad()
        pred = model(parameters, time, coords, phase=phase, cycle_fraction=cycle_frac)
        loss, _, _ = loss_and_metrics(pred, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < 0.20 * initial_loss, (
        f"Overfit test failed: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}"
    )


def test_train_mean_predictor_loss() -> None:
    """Verify that predicting zero on standardized fields yields approximately 1.0 MSE."""
    torch.manual_seed(42)
    # Generate standardized target field with mean 0, variance 1
    target = torch.randn(4, 8, 3, 20, 15)
    zero_pred = torch.zeros_like(target)
    mask = torch.ones(3, 20, 15)

    mse, mae, _ = loss_and_metrics(zero_pred, target, mask)

    # Standard normal variance is 1.0, expected MSE is ~1.0
    assert 0.85 <= mse.item() <= 1.15, f"Expected MSE near 1.0, got {mse.item():.4f}"
    # Standard normal expected absolute value is sqrt(2/pi) ~ 0.798
    assert 0.70 <= mae.item() <= 0.90, f"Expected MAE near 0.80, got {mae.item():.4f}"

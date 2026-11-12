"""Unit tests for PINO continuous autograd, collocation sampling, domain masks, and golden simulator residual convergence."""

from __future__ import annotations

import numpy as np
import torch

from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.stencil import laplacian_torch
from thin_film_pino.training.pino import compute_physics_loss, sample_collocation_times


def test_collocation_time_sampling_bounds() -> None:
    """Verify that collocation times fall strictly inside phase bounds across multi-cycle spans."""
    pulse_times = torch.tensor([15.0, 45.0, 80.0])
    num_cycles = 5
    num_collocation = 4

    t_dose, phase_dose, cycle_dose, t_rxn, phase_rxn, cycle_rxn = sample_collocation_times(
        pulse_times=pulse_times,
        num_collocation_per_phase=num_collocation,
        num_cycles=num_cycles,
    )

    assert t_dose.shape == (3, num_collocation)
    assert t_rxn.shape == (3, num_collocation)
    assert (phase_dose == 1).all()
    assert (phase_rxn == 3).all()
    assert (0.0 < cycle_dose).all() and (cycle_dose <= 1.0).all()
    assert (0.0 < cycle_rxn).all() and (cycle_rxn <= 1.0).all()

    # Verify physical time bounds
    cycle_durations = pulse_times + 3.0
    for b in range(3):
        for k in range(num_collocation):
            c_d = int(torch.round(cycle_dose[b, k] * num_cycles).item())
            c_start_d = (c_d - 1) * cycle_durations[b].item()
            assert c_start_d <= t_dose[b, k].item() <= c_start_d + pulse_times[b].item()

            c_r = int(torch.round(cycle_rxn[b, k] * num_cycles).item())
            c_start_r = (c_r - 1) * cycle_durations[b].item()
            rxn_start = c_start_r + pulse_times[b].item() + 1.0
            assert rxn_start <= t_rxn[b, k].item() <= rxn_start + 1.0


def test_pino_continuous_autograd_and_scaling() -> None:
    """Verify that compute_physics_loss evaluates without NaNs, respects masks, and scales residuals."""
    torch.manual_seed(42)
    batch_size = 2
    height, width = 20, 15
    param_dim = 5

    model = TemporalFourierOperator(
        parameter_dim=param_dim,
        spatial_channels=3,
        num_phases=5,
        include_cycle=True,
        output_channels=3,
        width=16,
        modes=4,
        num_blocks=2,
    )

    params_norm = torch.randn(batch_size, param_dim)
    params_phys = torch.tensor(
        [
            [3.0e-3, 1.0, 0.05, 0.4, 40.0],
            [5.0e-3, 0.8, 0.04, 0.3, 30.0],
        ]
    )

    coords = torch.randn(3, height, width)
    gas_mask = torch.ones(height, width, dtype=torch.bool)
    gas_mask[10:, :4] = False
    gas_mask[10:, 11:] = False

    surface_mask = torch.zeros(height, width, dtype=torch.bool)
    surface_mask[10, :4] = True
    surface_mask[10, 11:] = True

    stats = (
        torch.tensor([0.2, 0.5, 0.01]),  # field means
        torch.tensor([0.3, 0.3, 0.01]),  # field stds
        torch.tensor(50.0),  # time mean
        torch.tensor(30.0),  # time std
    )

    loss, metrics = compute_physics_loss(
        model=model,
        parameters_norm=params_norm,
        parameters_phys=params_phys,
        spatial_coords=coords,
        gas_mask=gas_mask,
        surface_mask=surface_mask,
        stats=stats,
        dx=1.0 / (width - 1),
        dy=1.0 / (height - 1),
        num_collocation_per_phase=2,
        num_cycles=5,
    )

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0

    # Verify individual metrics are recorded and finite
    assert "transport_mse" in metrics
    assert "coverage_mse" in metrics
    assert "growth_mse" in metrics
    for v in metrics.values():
        assert np.isfinite(v) and v >= 0.0

    # Verify backpropagation through model weights
    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_golden_simulator_residual_convergence() -> None:
    """Golden test: prove that feeding dense simulator trajectories yields small residuals that shrink with dt."""
    params = np.array([[3.0e-3, 1.0, 0.05, 0.4, 0.02, 5.0, 1.0, 1.0]])
    sim = BatchedTransientALDCudaSimulator(params, nx=20, ny=15, device="cpu", dtype=torch.float64)
    dt1 = float(sim.dt)
    active = torch.tensor([True])

    interior_gas = sim.gas_mask.clone()
    interior_gas[:, 0, :] = False
    surf_3d = sim.surface_mask.expand_as(sim.C)

    # 1. Forward difference vs. simulator RHS (matches to machine precision)
    sim.apply_boundary_conditions(sim.C, inlet_mask=True)
    c0 = sim.C.clone()
    th0 = sim.theta.clone()

    sim.transport_step(active=active, pulse=True, dt=dt1)
    c1 = sim.C.clone()
    th1 = sim.theta.clone()

    dc_dt_fd = (c1 - c0) / dt1
    lap0 = laplacian_torch(c0, sim.dx, sim.dy, sim.gas_mask)
    r_ads0 = torch.zeros_like(c0)
    c_surf0 = c0.masked_select(surf_3d).reshape(1, -1)
    th_surf0 = th0.masked_select(surf_3d).reshape(1, -1)
    r_ads0.masked_scatter_(surf_3d, sim.k_ads[:, 0, :] * c_surf0 * (1.0 - th_surf0))
    rhs0 = sim.diffusivity * lap0 - r_ads0

    forward_diff_err = ((dc_dt_fd - rhs0).abs() * interior_gas).max().item()
    assert forward_diff_err < 1.0e-12, (
        f"Expected machine precision forward difference, got {forward_diff_err}"
    )

    # Surface coverage forward difference
    dth_dt_fd = (th1 - th0) / dt1
    r_des0 = torch.zeros_like(th0)
    r_des0.masked_scatter_(surf_3d, sim.k_des[:, 0, :] * th_surf0)
    rhs_th0 = r_ads0 - r_des0
    theta_fd_err = ((dth_dt_fd - rhs_th0).abs() * surf_3d).max().item()
    assert theta_fd_err < 1.0e-12, (
        f"Expected machine precision coverage difference, got {theta_fd_err}"
    )

    # 2. Central difference truncation error convergence as dt shrinks
    # Run with dt1:
    sim_a = BatchedTransientALDCudaSimulator(
        params, nx=20, ny=15, device="cpu", dtype=torch.float64
    )
    sim_a.apply_boundary_conditions(sim_a.C, inlet_mask=True)
    c_history1 = [sim_a.C.clone()]
    th_history1 = [sim_a.theta.clone()]
    for _ in range(4):
        sim_a.transport_step(active=active, pulse=True, dt=dt1)
        sim_a.apply_boundary_conditions(sim_a.C, inlet_mask=True)
        c_history1.append(sim_a.C.clone())
        th_history1.append(sim_a.theta.clone())

    dc_dt_cd1 = (c_history1[3] - c_history1[1]) / (2.0 * dt1)
    lap1 = laplacian_torch(c_history1[2], sim_a.dx, sim_a.dy, sim_a.gas_mask)
    r_ads1 = torch.zeros_like(c_history1[2])
    c_surf1 = c_history1[2].masked_select(surf_3d).reshape(1, -1)
    th_surf1 = th_history1[2].masked_select(surf_3d).reshape(1, -1)
    r_ads1.masked_scatter_(surf_3d, sim_a.k_ads[:, 0, :] * c_surf1 * (1.0 - th_surf1))
    rhs1 = sim_a.diffusivity * lap1 - r_ads1
    res1 = ((dc_dt_cd1 - rhs1).square() * interior_gas).sum().sqrt().item()

    # Run with dt2 = dt1 / 2:
    sim_b = BatchedTransientALDCudaSimulator(
        params, nx=20, ny=15, dt=dt1 / 2.0, device="cpu", dtype=torch.float64
    )
    dt2 = float(sim_b.dt)
    sim_b.apply_boundary_conditions(sim_b.C, inlet_mask=True)
    c_history2 = [sim_b.C.clone()]
    th_history2 = [sim_b.theta.clone()]
    for _ in range(8):
        sim_b.transport_step(active=active, pulse=True, dt=dt2)
        sim_b.apply_boundary_conditions(sim_b.C, inlet_mask=True)
        c_history2.append(sim_b.C.clone())
        th_history2.append(sim_b.theta.clone())

    dc_dt_cd2 = (c_history2[5] - c_history2[3]) / (2.0 * dt2)
    lap2 = laplacian_torch(c_history2[4], sim_b.dx, sim_b.dy, sim_b.gas_mask)
    r_ads2 = torch.zeros_like(c_history2[4])
    c_surf2 = c_history2[4].masked_select(surf_3d).reshape(1, -1)
    th_surf2 = th_history2[4].masked_select(surf_3d).reshape(1, -1)
    r_ads2.masked_scatter_(surf_3d, sim_b.k_ads[:, 0, :] * c_surf2 * (1.0 - th_surf2))
    rhs2 = sim_b.diffusivity * lap2 - r_ads2
    res2 = ((dc_dt_cd2 - rhs2).square() * interior_gas).sum().sqrt().item()

    assert res1 < 1.0, f"Expected small residual at dt1, got {res1}"
    assert res2 < res1, f"Expected residual to shrink with dt (res2={res2} < res1={res1})"

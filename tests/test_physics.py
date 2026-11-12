"""Unit tests for ALD physics, geometry partitioning, clamping, and solver parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.config import SimulationConfig
from thin_film_pino.physics.geometry import (
    build_geometry,
    compute_stable_dt,
    validate_dt,
)
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator
from thin_film_pino.physics.transient_cuda import TransientALDCudaSimulator


def test_geometry_masks_partition() -> None:
    """Verify surface regions form a mutually disjoint partition of the surface mask."""
    geom = build_geometry(nx=20, ny=15, lx=1.0, ly=1.0)

    # Disjointness checks
    assert not np.any(geom.field_mask & geom.sidewall_mask)
    assert not np.any(geom.field_mask & geom.floor_mask)
    assert not np.any(geom.sidewall_mask & geom.floor_mask)

    # Partition check: union of regions must equal surface_mask
    union_regions = geom.field_mask | geom.sidewall_mask | geom.floor_mask
    np.testing.assert_array_equal(union_regions, geom.surface_mask)

    # Gas / solid domain separation
    assert not np.any(geom.surface_mask & geom.solid_mask)
    assert np.all(geom.surface_mask <= geom.gas_mask)
    assert np.all((geom.solid_mask | geom.gas_mask) == True)  # noqa: E712
    assert not np.any(geom.solid_mask & geom.gas_mask)


def test_aspect_ratio_preset() -> None:
    """Verify aspect ratio preset configures correct trench dimensions."""
    cfg = SimulationConfig.high_aspect_ratio_preset(aspect_ratio=10.0, nx=20, ny=15)
    assert cfg.aspect_ratio == 10.0
    depth = cfg.trench_bottom - cfg.trench_top
    width = cfg.trench_right - cfg.trench_left
    computed_ar = depth / width
    assert pytest.approx(computed_ar, rel=1e-5) == 10.0


def test_default_simulation_zero_clamps() -> None:
    """Verify standard simulation on small CPU grid completes with zero clamp activations."""
    config = SimulationConfig(
        nx=20,
        ny=15,
        num_cycles=2,
        dtype="float32",
        device="cpu",
    )
    sim = ThinFilmDepositionSimulator(config=config)
    sim.run()

    # Assert zero clamp activations for standard parameters
    stats = sim.clamp_stats
    assert stats["c_negative"] == 0
    assert stats["theta_negative"] == 0
    assert stats["theta_exceeded"] == 0

    # Physical validity of state fields
    assert np.all(sim.C >= 0.0)
    assert np.all(sim.theta >= 0.0)
    assert np.all(sim.theta <= 1.0)
    assert np.all(sim.h >= 0.0)

    # Saturation coverage recorded in history
    assert "dose_coverage" in sim.history
    assert len(sim.history["dose_coverage"]) == 2
    assert sim.history["dose_coverage"][-1] > 0.0

    # Step coverage is floor / field
    field_h = float(np.mean(sim.h[sim.field_mask]))
    floor_h = float(np.mean(sim.h[sim.floor_mask]))
    expected_sc = floor_h / field_h if field_h > 0 else 0.0
    assert pytest.approx(sim.history["step_coverage"][-1], rel=1e-5) == expected_sc


def test_time_step_validation() -> None:
    """Verify stability checker rejects time steps that violate von Neumann or reaction limits."""
    dx = 1.0 / 19.0
    dy = 1.0 / 14.0
    diffusivity = 5.0e-3
    k_ads, k_des, k_rxn = 1.0, 0.05, 0.4

    # Stable dt computed automatically
    dt_stable = compute_stable_dt(dx, dy, diffusivity, k_ads, k_des, k_rxn)
    validate_dt(dt_stable, dx, dy, diffusivity, k_ads, k_des, k_rxn)

    # Violate diffusive limit
    unstable_diff_dt = 1.0
    with pytest.raises(ValueError, match="diffusive von Neumann"):
        validate_dt(unstable_diff_dt, dx, dy, diffusivity, k_ads, k_des, k_rxn)

    # Violate reaction kinetic limit
    with pytest.raises(ValueError, match="reaction kinetic"):
        validate_dt(2.0, 10.0, 10.0, 1e-6, 1.0, 0.1, 0.5)


def test_numpy_and_pytorch_solver_parity() -> None:
    """Verify exact numerical parity between reference NumPy and PyTorch simulators."""
    nx, ny = 20, 15
    num_cycles = 2
    params = [5.0e-3, 1.0, 0.05, 0.4, 0.02, 1.0, 1.0, 1.0]

    ref_sim = ThinFilmDepositionSimulator(
        diffusivity=params[0],
        k_ads=params[1],
        k_des=params[2],
        k_rxn=params[3],
        k_growth=params[4],
        pulse_time=params[5],
        purge_time=params[6],
        reaction_time=params[7],
        num_cycles=num_cycles,
        nx=nx,
        ny=ny,
        dtype="float64",
    )
    ref_sim.run()

    fast_sim = BatchedTransientALDCudaSimulator(
        parameters=np.asarray([params]),
        nx=nx,
        ny=ny,
        dtype="float64",
        device="cpu",
        dt=ref_sim.dt,
    )

    pulse_steps = fast_sim.pulse_steps()
    purge_steps = fast_sim.fixed_steps(fast_sim.purge_time)
    reaction_steps = fast_sim.fixed_steps(fast_sim.reaction_time)
    active_mask = torch.ones(1, dtype=torch.bool, device="cpu")

    max_pulse = int(pulse_steps.max().item())
    for _ in range(num_cycles):
        for s in range(1, max_pulse + 1):
            fast_sim.transport_step(pulse_steps >= s, pulse=True)
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()
        for _ in range(reaction_steps):
            fast_sim.reaction_step()
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()

    c_batch, theta_batch, h_batch = fast_sim.snapshot()

    # Exact parity on identical discretization
    np.testing.assert_allclose(theta_batch[0], ref_sim.theta, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(h_batch[0], ref_sim.h, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(c_batch[0], ref_sim.C, atol=1e-10, rtol=1e-7)


def test_transient_wrapper_delegation() -> None:
    """Verify TransientALDCudaSimulator delegates properly to BatchedTransientALDCudaSimulator."""
    wrapper = TransientALDCudaSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=1.0,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=2,
        nx=20,
        ny=15,
        device="cpu",
    )
    wrapper.run_cycle()
    c, theta, h = wrapper.snapshot()

    assert c.shape == (15, 20)
    assert theta.shape == (15, 20)
    assert h.shape == (15, 20)
    assert wrapper.field_mask.shape == (15, 20)
    assert wrapper.floor_mask.shape == (15, 20)
    assert wrapper.clamp_stats["c_negative"] == 0


def test_golden_regression_matches() -> None:
    """Verify that current simulators exactly reproduce the committed golden regression run."""
    golden_path = Path(__file__).resolve().parent / "data" / "golden_regression.npz"
    assert golden_path.exists(), f"Golden regression file not found at: {golden_path}"

    golden = np.load(golden_path)
    p = golden["parameters"]
    grid = golden["grid"]
    ny, nx = int(grid[0]), int(grid[1])
    num_cycles = int(golden["num_cycles"])

    # 1. Run reference NumPy solver on golden setup
    sim = ThinFilmDepositionSimulator(
        diffusivity=float(p[0]),
        k_ads=float(p[1]),
        k_des=float(p[2]),
        k_rxn=float(p[3]),
        k_growth=float(p[4]),
        pulse_time=float(p[5]),
        purge_time=float(p[6]),
        reaction_time=float(p[7]),
        num_cycles=num_cycles,
        nx=nx,
        ny=ny,
        dtype="float64",
    )
    sim.run()

    np.testing.assert_allclose(sim.C, golden["concentration"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(sim.theta, golden["surface_coverage"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(sim.h, golden["film_thickness"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(sim.history["gpc"], golden["gpc"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(
        sim.history["step_coverage"], golden["step_coverage"], atol=1e-12, rtol=1e-10
    )

    # 2. Run batched PyTorch solver with matching dt and verify exact parity
    fast_sim = BatchedTransientALDCudaSimulator(
        parameters=np.asarray([p]),
        nx=nx,
        ny=ny,
        device="cpu",
        dtype="float64",
        dt=sim.dt,
    )
    pulse_steps = int(fast_sim.pulse_steps().item())
    purge_steps = int(fast_sim.fixed_steps(fast_sim.purge_time))
    reaction_steps = int(fast_sim.fixed_steps(fast_sim.reaction_time))
    active_mask = torch.ones(1, dtype=torch.bool, device="cpu")

    for _ in range(num_cycles):
        for _ in range(pulse_steps):
            fast_sim.transport_step(active_mask, pulse=True)
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()
        for _ in range(reaction_steps):
            fast_sim.reaction_step()
        for _ in range(purge_steps):
            fast_sim.transport_step(active_mask, pulse=False)
        fast_sim.C.zero_()

    c_f, th_f, h_f = fast_sim.snapshot()
    np.testing.assert_allclose(c_f[0], golden["concentration"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(th_f[0], golden["surface_coverage"], atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(h_f[0], golden["film_thickness"], atol=1e-12, rtol=1e-10)


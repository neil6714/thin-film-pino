"""Physics-informed residual formulation and continuous collocation autograd for ALD surrogates."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from thin_film_pino.physics.stencil import laplacian_torch


def sample_collocation_times(
    pulse_times: torch.Tensor,
    num_collocation_per_phase: int = 2,
    num_cycles: int = 10,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample continuous collocation times uniformly inside dose and reaction phase windows.

    Parameters
    ----------
    pulse_times : torch.Tensor
        Batch precursor dose durations shaped (B,).
    num_collocation_per_phase : int, default=2
        Number of collocation evaluation points sampled per phase.
    num_cycles : int, default=10
        Total multi-cycle count for trajectory span.
    device : torch.device, optional
        Target PyTorch compute device.

    Returns
    -------
    t_dose_phys : torch.Tensor
        Dose stage physical times shaped (B, N_coll).
    phase_dose : torch.Tensor
        Dose stage phase indicators (value 1) shaped (B, N_coll).
    cycle_frac_dose : torch.Tensor
        Normalized cycle progress for dose samples shaped (B, N_coll).
    t_rxn_phys : torch.Tensor
        Reaction stage physical times shaped (B, N_coll).
    phase_rxn : torch.Tensor
        Reaction stage phase indicators (value 3) shaped (B, N_coll).
    cycle_frac_rxn : torch.Tensor
        Normalized cycle progress for reaction samples shaped (B, N_coll).
    """
    batch_size = pulse_times.shape[0]
    dev = pulse_times.device if device is None else device

    # Cycle length per sample: pulse_time + purge(1.0) + rxn(1.0) + purge(1.0)
    cycle_duration = pulse_times + 3.0

    # Sample random cycle index in [1, num_cycles] for each collocation point
    cycles_dose = torch.randint(
        1, num_cycles + 1, (batch_size, num_collocation_per_phase), device=dev
    )
    cycles_rxn = torch.randint(
        1, num_cycles + 1, (batch_size, num_collocation_per_phase), device=dev
    )

    # Uniform interior sampling in [0.05, 0.95] to avoid boundary singularity
    u_dose = 0.05 + 0.90 * torch.rand(batch_size, num_collocation_per_phase, device=dev)
    u_rxn = 0.05 + 0.90 * torch.rand(batch_size, num_collocation_per_phase, device=dev)

    c_start_dose = (cycles_dose - 1).float() * cycle_duration.unsqueeze(1)
    c_start_rxn = (cycles_rxn - 1).float() * cycle_duration.unsqueeze(1)

    # 1. Dose stage times: t in [c_start, c_start + pulse_time]
    t_dose_phys = c_start_dose + u_dose * pulse_times.unsqueeze(1)
    phase_dose = torch.full(
        (batch_size, num_collocation_per_phase), 1, dtype=torch.long, device=dev
    )
    cycle_frac_dose = cycles_dose.float() / float(num_cycles)

    # 2. Reaction stage times: t in [c_start + pulse_time + 1.0, c_start + pulse_time + 2.0]
    t_rxn_phys = c_start_rxn + pulse_times.unsqueeze(1) + 1.0 + u_rxn * 1.0
    phase_rxn = torch.full((batch_size, num_collocation_per_phase), 3, dtype=torch.long, device=dev)
    cycle_frac_rxn = cycles_rxn.float() / float(num_cycles)

    return t_dose_phys, phase_dose, cycle_frac_dose, t_rxn_phys, phase_rxn, cycle_frac_rxn


def compute_physics_loss(
    model: nn.Module,
    parameters_norm: torch.Tensor,
    parameters_phys: torch.Tensor,
    spatial_coords: torch.Tensor,
    gas_mask: torch.Tensor,
    surface_mask: torch.Tensor,
    stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    dx: float,
    dy: float,
    num_collocation_per_phase: int = 2,
    num_cycles: int = 10,
    k_growth: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute continuum PDE residuals via continuous autograd and masked boundary stencils.

    Parameters
    ----------
    model : nn.Module
        Neural operator model taking (parameters, time, coords, phase, cycle_fraction).
    parameters_norm : torch.Tensor
        Standardized network input parameters shaped (B, P).
    parameters_phys : torch.Tensor
        Physical unnormalized parameters shaped (B, P) with pulse_time at index 4 or last.
    spatial_coords : torch.Tensor
        Grid coordinates and geometry mask shaped (3, ny, nx).
    gas_mask : torch.Tensor
        Boolean mask of gas phase domain shaped (ny, nx).
    surface_mask : torch.Tensor
        Boolean mask of reactive substrate surface shaped (ny, nx).
    stats : tuple
        (field_mean, field_std, time_mean, time_std) tensors.
    dx : float
        Grid spacing along horizontal direction.
    dy : float
        Grid spacing along vertical direction.
    num_collocation_per_phase : int, default=2
        Number of unlabeled continuous collocation times sampled inside each phase.
    num_cycles : int, default=10
        Total multi-cycle count for trajectory span.
    k_growth : float, default=0.02
        Physical film thickness conversion factor per converted coverage.

    Returns
    -------
    total_loss : torch.Tensor
        Scalar regularized PDE residual objective.
    metrics : dict
        Individual MSE components for transport, coverage, and thickness residuals.
    """
    field_mean, field_std, time_mean, time_std = stats
    mu_c, mu_theta, mu_h = field_mean[0], field_mean[1], field_mean[2]
    sigma_c, sigma_theta, sigma_h = field_std[0], field_std[1], field_std[2]

    batch_size = parameters_norm.shape[0]
    height, width = gas_mask.shape[-2:]

    # Physical kinetic parameters: [diffusivity, k_ads, k_des, k_rxn, pulse_time]
    diffusivity = parameters_phys[:, 0, None, None, None]
    k_ads = parameters_phys[:, 1, None, None, None]
    k_des = parameters_phys[:, 2, None, None, None]
    k_rxn = parameters_phys[:, 3, None, None, None]
    pulse_time = parameters_phys[:, 4] if parameters_phys.shape[1] > 4 else parameters_phys[:, -1]

    # Sample continuous collocation points
    t_dose_phys, phase_dose, cycle_frac_dose, t_rxn_phys, phase_rxn, cycle_frac_rxn = (
        sample_collocation_times(
            pulse_times=pulse_time,
            num_collocation_per_phase=num_collocation_per_phase,
            num_cycles=num_cycles,
            device=parameters_norm.device,
        )
    )

    surface_float = surface_mask.float().unsqueeze(0).unsqueeze(0)

    # -------------------------------------------------------------
    # 1. Dose Phase Residuals (Transport, Adsorption, Desorption)
    # -------------------------------------------------------------
    t_norm_dose = (t_dose_phys - time_mean) / time_std
    t_spatial_dose = (
        t_norm_dose.unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 1, height, width)
        .contiguous()
        .requires_grad_(True)
    )

    pred_dose = model(
        parameters_norm,
        t_spatial_dose,
        spatial_coords,
        phase=phase_dose,
        cycle_fraction=cycle_frac_dose,
    )

    # Physical unnormalized fields
    c_dose = pred_dose[:, :, 0] * sigma_c + mu_c
    theta_dose = pred_dose[:, :, 1] * sigma_theta + mu_theta
    h_dose = pred_dose[:, :, 2] * sigma_h + mu_h

    # Enforce exact physical masking outside domain
    c_dose = torch.where(gas_mask, c_dose, torch.zeros_like(c_dose))
    theta_dose = torch.where(surface_mask, theta_dose, torch.zeros_like(theta_dose))
    h_dose = torch.where(surface_mask, h_dose, torch.zeros_like(h_dose))

    # Continuous autograd time derivatives
    dc_dtnorm = torch.autograd.grad(
        outputs=c_dose,
        inputs=t_spatial_dose,
        grad_outputs=torch.ones_like(c_dose),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze(2)

    dtheta_dtnorm_dose = torch.autograd.grad(
        outputs=theta_dose,
        inputs=t_spatial_dose,
        grad_outputs=torch.ones_like(theta_dose),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze(2)

    dh_dtnorm_dose = torch.autograd.grad(
        outputs=h_dose,
        inputs=t_spatial_dose,
        grad_outputs=torch.ones_like(h_dose),
        create_graph=True,
    )[0].squeeze(2)

    dc_dt_dose = dc_dtnorm / time_std
    dtheta_dt_dose = dtheta_dtnorm_dose / time_std
    dh_dt_dose = dh_dtnorm_dose / time_std

    # Spatial Laplacian with zero-flux solid reflection
    lap_c = laplacian_torch(c_dose, dx, dy, gas_mask)

    # Langmuir surface reactions localized strictly to boundary cells
    r_ads = surface_float * k_ads * c_dose * (1.0 - theta_dose)
    r_des = surface_float * k_des * theta_dose

    # Physical residuals
    res_c_dose = dc_dt_dose - diffusivity * lap_c + r_ads
    res_theta_dose = dtheta_dt_dose - (r_ads - r_des)
    res_h_dose = dh_dt_dose

    # -------------------------------------------------------------
    # 2. Reaction Phase Residuals (Co-Reactant Conversion & Growth)
    # -------------------------------------------------------------
    t_norm_rxn = (t_rxn_phys - time_mean) / time_std
    t_spatial_rxn = (
        t_norm_rxn.unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 1, height, width)
        .contiguous()
        .requires_grad_(True)
    )

    pred_rxn = model(
        parameters_norm,
        t_spatial_rxn,
        spatial_coords,
        phase=phase_rxn,
        cycle_fraction=cycle_frac_rxn,
    )

    theta_rxn = pred_rxn[:, :, 1] * sigma_theta + mu_theta
    h_rxn = pred_rxn[:, :, 2] * sigma_h + mu_h

    theta_rxn = torch.where(surface_mask, theta_rxn, torch.zeros_like(theta_rxn))
    h_rxn = torch.where(surface_mask, h_rxn, torch.zeros_like(h_rxn))

    dtheta_dtnorm_rxn = torch.autograd.grad(
        outputs=theta_rxn,
        inputs=t_spatial_rxn,
        grad_outputs=torch.ones_like(theta_rxn),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze(2)

    dh_dtnorm_rxn = torch.autograd.grad(
        outputs=h_rxn,
        inputs=t_spatial_rxn,
        grad_outputs=torch.ones_like(h_rxn),
        create_graph=True,
    )[0].squeeze(2)

    dtheta_dt_rxn = dtheta_dtnorm_rxn / time_std
    dh_dt_rxn = dh_dtnorm_rxn / time_std

    r_rxn = surface_float * k_rxn * theta_rxn

    res_theta_rxn = dtheta_dt_rxn + r_rxn
    res_h_rxn = dh_dt_rxn - k_growth * r_rxn

    # -------------------------------------------------------------
    # 3. Normalization, Scaling, and Domain Mask Reductions
    # -------------------------------------------------------------
    scaled_res_c = res_c_dose / sigma_c
    scaled_res_theta_dose = res_theta_dose / sigma_theta
    scaled_res_theta_rxn = res_theta_rxn / sigma_theta
    scaled_res_h_dose = res_h_dose / sigma_h
    scaled_res_h_rxn = res_h_rxn / sigma_h

    n_gas = gas_mask.float().sum().clamp_min(1.0) * float(batch_size * num_collocation_per_phase)
    n_surf = surface_mask.float().sum().clamp_min(1.0) * float(
        batch_size * num_collocation_per_phase
    )

    mse_transport = (scaled_res_c.square() * gas_mask.float()).sum() / n_gas
    mse_coverage = (
        (scaled_res_theta_dose.square() * surface_mask.float()).sum()
        + (scaled_res_theta_rxn.square() * surface_mask.float()).sum()
    ) / (2.0 * n_surf)
    mse_growth = (
        (scaled_res_h_dose.square() * surface_mask.float()).sum()
        + (scaled_res_h_rxn.square() * surface_mask.float()).sum()
    ) / (2.0 * n_surf)

    # Check for NaN / numerical failure without nan_to_num masking
    if (
        torch.isnan(mse_transport)
        or torch.isnan(mse_coverage)
        or torch.isnan(mse_growth)
        or torch.isinf(mse_transport)
        or torch.isinf(mse_coverage)
        or torch.isinf(mse_growth)
    ):
        raise FloatingPointError(
            "Numerical failure (NaN/Inf) encountered during physics residual computation."
        )

    total_pde_loss = (mse_transport + mse_coverage + mse_growth) / 3.0

    metrics = {
        "transport_mse": float(mse_transport.item()),
        "coverage_mse": float(mse_coverage.item()),
        "growth_mse": float(mse_growth.item()),
    }

    return total_pde_loss, metrics

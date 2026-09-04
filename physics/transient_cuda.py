import math

import torch


class TransientALDCudaSimulator:
    """CUDA trajectory recorder matching the validated ALD update equations."""

    def __init__(
        self,
        diffusivity,
        k_ads,
        k_des,
        k_rxn,
        k_growth,
        pulse_time,
        purge_time,
        reaction_time,
        num_cycles,
        nx=160,
        ny=120,
        width=1.0,
        height=1.0,
        trench_left=0.375,
        trench_right=0.625,
        trench_top=0.50,
        trench_bottom=0.95,
        inlet_concentration=1.0,
        device="cuda",
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Stage 4A requires an available CUDA device.")

        self.nx = nx
        self.ny = ny
        self.width = width
        self.height = height
        self.trench_left = trench_left
        self.trench_right = trench_right
        self.trench_top = trench_top
        self.trench_bottom = trench_bottom
        self.diffusivity = float(diffusivity)
        self.k_ads = float(k_ads)
        self.k_des = float(k_des)
        self.k_rxn = float(k_rxn)
        self.k_growth = float(k_growth)
        self.inlet_concentration = float(inlet_concentration)
        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.num_cycles = int(num_cycles)

        self.dx = width / (nx - 1)
        self.dy = height / (ny - 1)
        diffusion_dt = 0.20 * min(self.dx, self.dy) ** 2 / self.diffusivity
        reaction_dt = 0.20 / max(self.k_ads, self.k_des, self.k_rxn, 1e-12)
        self.dt = min(diffusion_dt, reaction_dt)

        x = torch.linspace(0.0, width, nx, device=self.device)
        y = torch.linspace(0.0, height, ny, device=self.device)
        self.x = x
        self.y = y
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")
        self.solid_mask = self._build_geometry()
        self.gas_mask = ~self.solid_mask
        self.surface_mask = self._find_surface_cells()
        self.top_region = self.surface_mask & (
            self.Y < self.trench_top + 2.5 * self.dy
        )
        self.bottom_region = self.surface_mask & (
            self.Y > self.trench_bottom - 2.5 * self.dy
        )

        self.C = torch.zeros((ny, nx), device=self.device)
        self.theta = torch.zeros((ny, nx), device=self.device)
        self.h = torch.zeros((ny, nx), device=self.device)
        self.history = self._empty_history()

    def _empty_history(self):
        return {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "surface_coverage": [],
        }

    def _build_geometry(self):
        substrate = self.Y >= self.trench_bottom
        outside_trench = (self.Y >= self.trench_top) & (
            (self.X < self.trench_left) | (self.X > self.trench_right)
        )
        return substrate | outside_trench

    def _find_surface_cells(self):
        neighbors = torch.zeros_like(self.solid_mask)
        neighbors[1:, :] |= self.solid_mask[:-1, :]
        neighbors[:-1, :] |= self.solid_mask[1:, :]
        neighbors[:, 1:] |= self.solid_mask[:, :-1]
        neighbors[:, :-1] |= self.solid_mask[:, 1:]
        return self.gas_mask & neighbors

    def _laplacian(self, field):
        up = field.clone()
        down = field.clone()
        left = field.clone()
        right = field.clone()

        up[1:, :] = torch.where(
            self.gas_mask[:-1, :],
            field[:-1, :],
            field[1:, :],
        )
        down[:-1, :] = torch.where(
            self.gas_mask[1:, :],
            field[1:, :],
            field[:-1, :],
        )
        left[:, 1:] = torch.where(
            self.gas_mask[:, :-1],
            field[:, :-1],
            field[:, 1:],
        )
        right[:, :-1] = torch.where(
            self.gas_mask[:, 1:],
            field[:, 1:],
            field[:, :-1],
        )

        laplacian = (
            (up + down - 2.0 * field) / self.dy**2
            + (left + right - 2.0 * field) / self.dx**2
        )
        return torch.where(self.gas_mask, laplacian, torch.zeros_like(laplacian))

    def _num_steps(self, duration):
        return max(1, math.ceil(duration / self.dt))

    def transport_step(self, pulse):
        diffusion = self.diffusivity * self._laplacian(self.C)
        adsorption = torch.zeros_like(self.C)
        desorption = torch.zeros_like(self.theta)

        if pulse:
            adsorption[self.surface_mask] = (
                self.k_ads
                * self.C[self.surface_mask]
                * (1.0 - self.theta[self.surface_mask])
            )
            desorption[self.surface_mask] = self.k_des * self.theta[self.surface_mask]

        self.C = torch.clamp(self.C + self.dt * (diffusion - adsorption), min=0.0)
        self.C = torch.where(self.gas_mask, self.C, torch.zeros_like(self.C))

        if pulse:
            self.theta = torch.clamp(
                self.theta + self.dt * (adsorption - desorption),
                min=0.0,
                max=1.0,
            )
            self.C[0, :] = self.inlet_concentration
        else:
            self.C[0, :] = 0.0

    def reaction_step(self):
        reaction = torch.zeros_like(self.theta)
        reaction[self.surface_mask] = self.k_rxn * self.theta[self.surface_mask]
        self.h = self.h + self.dt * self.k_growth * reaction
        self.theta = torch.clamp(self.theta - self.dt * reaction, min=0.0, max=1.0)

    def record_cycle_metrics(self, cycle, previous_thickness):
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]
        top_thickness = self.h[self.top_region].mean().item()
        bottom_thickness = self.h[self.bottom_region].mean().item()
        mean_thickness = surface_h.mean().item()
        mean_coverage = surface_theta.mean().item()
        conformality = bottom_thickness / top_thickness if top_thickness > 0.0 else 0.0
        gpc = mean_thickness - previous_thickness

        self.history["cycle"].append(cycle)
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(mean_thickness)
        self.history["top_thickness"].append(top_thickness)
        self.history["bottom_thickness"].append(bottom_thickness)
        self.history["conformality"].append(conformality)
        self.history["surface_coverage"].append(mean_coverage)
        return mean_thickness

    def snapshot(self):
        return (
            self.C.detach().cpu().numpy().astype("float32"),
            self.theta.detach().cpu().numpy().astype("float32"),
            self.h.detach().cpu().numpy().astype("float32"),
        )

import math

import torch


class BatchedTransientALDCudaSimulator:
    def __init__(self, parameters, nx=80, ny=60, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The fast transient generator requires CUDA.")

        self.parameters = torch.as_tensor(parameters, dtype=torch.float32, device=self.device)
        self.batch_size = self.parameters.shape[0]
        self.nx, self.ny = nx, ny
        self.dx, self.dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
        self.diffusivity = self.parameters[:, 0, None, None]
        self.k_ads = self.parameters[:, 1, None, None]
        self.k_des = self.parameters[:, 2, None, None]
        self.k_rxn = self.parameters[:, 3, None, None]
        self.k_growth = self.parameters[:, 4, None, None]
        self.pulse_time = self.parameters[:, 5]
        self.purge_time = float(self.parameters[0, 6].item())
        self.reaction_time = float(self.parameters[0, 7].item())
        max_diffusivity = float(self.parameters[:, 0].max().item())
        max_rate = float(self.parameters[:, 1:4].max().item())
        self.dt = min(
            0.20 * min(self.dx, self.dy) ** 2 / max_diffusivity,
            0.20 / max(max_rate, 1e-12),
        )

        x = torch.linspace(0.0, 1.0, nx, device=self.device)
        y = torch.linspace(0.0, 1.0, ny, device=self.device)
        self.x, self.y = x, y
        self.X, self.Y = torch.meshgrid(x, y, indexing="xy")
        self.solid_mask = self._build_geometry()
        self.gas_mask = ~self.solid_mask
        self.surface_mask = self._find_surface_cells()
        self.C = torch.zeros((self.batch_size, ny, nx), device=self.device)
        self.theta = torch.zeros_like(self.C)
        self.h = torch.zeros_like(self.C)

    def _build_geometry(self):
        substrate = self.Y >= 0.95
        outside_trench = (self.Y >= 0.50) & ((self.X < 0.375) | (self.X > 0.625))
        return substrate | outside_trench

    def _find_surface_cells(self):
        neighbors = torch.zeros_like(self.solid_mask)
        neighbors[1:] |= self.solid_mask[:-1]
        neighbors[:-1] |= self.solid_mask[1:]
        neighbors[:, 1:] |= self.solid_mask[:, :-1]
        neighbors[:, :-1] |= self.solid_mask[:, 1:]
        return self.gas_mask & neighbors

    def _laplacian(self, field):
        up, down, left, right = field.clone(), field.clone(), field.clone(), field.clone()
        up[:, 1:] = torch.where(self.gas_mask[:-1], field[:, :-1], field[:, 1:])
        down[:, :-1] = torch.where(self.gas_mask[1:], field[:, 1:], field[:, :-1])
        left[:, :, 1:] = torch.where(self.gas_mask[:, :-1], field[:, :, :-1], field[:, :, 1:])
        right[:, :, :-1] = torch.where(self.gas_mask[:, 1:], field[:, :, 1:], field[:, :, :-1])
        laplacian = (up + down - 2.0 * field) / self.dy**2
        laplacian += (left + right - 2.0 * field) / self.dx**2
        return torch.where(self.gas_mask, laplacian, torch.zeros_like(laplacian))

    def transport_step(self, active, pulse):
        diffusion = self.diffusivity * self._laplacian(self.C)
        adsorption = torch.zeros_like(self.C)
        desorption = torch.zeros_like(self.C)
        if pulse:
            adsorption[:, self.surface_mask] = (
                self.k_ads[:, 0, 0, 0, None]
                * self.C[:, self.surface_mask]
                * (1.0 - self.theta[:, self.surface_mask])
            )
            desorption[:, self.surface_mask] = self.k_des[:, 0, 0, 0, None] * self.theta[:, self.surface_mask]
        new_c = torch.clamp(self.C + self.dt * (diffusion - adsorption), min=0.0)
        new_c = torch.where(self.gas_mask, new_c, torch.zeros_like(new_c))
        active_3d = active[:, None, None]
        self.C = torch.where(active_3d, new_c, self.C)
        if pulse:
            new_theta = torch.clamp(self.theta + self.dt * (adsorption - desorption), 0.0, 1.0)
            self.theta = torch.where(active_3d, new_theta, self.theta)
            self.C[active, 0, :] = 1.0
        else:
            self.C[active, 0, :] = 0.0

    def reaction_step(self):
        reaction = torch.zeros_like(self.theta)
        reaction[:, self.surface_mask] = self.k_rxn[:, 0, 0, 0, None] * self.theta[:, self.surface_mask]
        self.h += self.dt * self.k_growth * reaction
        self.theta = torch.clamp(self.theta - self.dt * reaction, 0.0, 1.0)

    def phase_steps(self, duration):
        value = torch.as_tensor(duration, device=self.device)
        return torch.clamp(torch.ceil(value / self.dt).to(torch.int64), min=1)

    def pulse_steps(self):
        return self.phase_steps(self.pulse_time)

    def fixed_steps(self, duration):
        return int(math.ceil(duration / self.dt))

    def snapshot(self):
        return tuple(field.detach().cpu().numpy().astype("float32") for field in (self.C, self.theta, self.h))

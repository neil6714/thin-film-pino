"""2D continuum atomic layer deposition (ALD) transport and surface reaction simulator.

Discretizes transient precursor vapor diffusion, self-limiting Langmuir surface
adsorption, purge-stage desorption, and co-reactant surface reaction in deep trench
microelectronic structures. Uses an explicit finite-difference time-domain (FDTD) scheme
governed by the 2D diffusive Courant-Friedrichs-Lewy (CFL) stability criterion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ThinFilmDepositionSimulator:
    """Continuum finite-difference simulator for cyclic ALD in 2D trench geometries.

    Solves the coupled diffusion-reaction system:
        dC/dt = D grad^2C - R_ads(x, y, t)
        dtheta/dt = R_ads - R_des - R_rxn
        dh/dt = k_growth * R_rxn

    with Langmuir kinetics:
        R_ads = k_ads * C * (1 - theta)
        R_des = k_des * theta
        R_rxn = k_rxn * theta

    subject to Dirichlet/Robin boundary conditions at the reactor inlet (y = 0)
    and zero-flux Neumann boundary conditions along feature surfaces and domain boundaries.

    Parameters
    ----------
    nx : int, default=160
        Horizontal spatial grid resolution.
    ny : int, default=120
        Vertical spatial grid resolution.
    width : float, default=1.0
        Normalized simulation domain width Lx in meters.
    height : float, default=1.0
        Normalized simulation domain height Ly in meters.
    trench_left : float, default=0.375
        Horizontal position of left trench sidewall in meters.
    trench_right : float, default=0.625
        Horizontal position of right trench sidewall in meters.
    trench_top : float, default=0.50
        Vertical position of trench mouth in meters.
    trench_bottom : float, default=0.95
        Vertical position of trench bottom floor in meters.
    diffusivity : float, default=5.0e-3
        Precursor gas diffusion coefficient D in m^2/s.
    k_ads : float, default=1.0
        Adsorption rate constant in s^-1.
    k_des : float, default=0.05
        Thermal desorption rate constant in s^-1.
    k_rxn : float, default=0.4
        Co-reactant reaction rate constant in s^-1.
    k_growth : float, default=0.02
        Growth factor per reacted monolayer in meters.
    c0 : float, default=1.0
        Inlet boundary precursor molar concentration in mol/m^3.
    pulse_time : float, default=1.0
        Duration of precursor pulse stage in seconds.
    purge_time : float, default=1.0
        Duration of inert gas purge stage in seconds.
    reaction_time : float, default=1.0
        Duration of co-reactant reaction stage in seconds.
    post_purge_time : float, default=1.0
        Duration of post-reaction purge stage in seconds.
    num_cycles : int, default=10
        Total number of complete ALD reaction-purge cycles to execute.
    """

    def __init__(
        self,
        nx: int = 160,
        ny: int = 120,
        width: float = 1.0,
        height: float = 1.0,
        trench_left: float = 0.375,
        trench_right: float = 0.625,
        trench_top: float = 0.50,
        trench_bottom: float = 0.95,
        diffusivity: float = 5.0e-3,
        k_ads: float = 1.0,
        k_des: float = 0.05,
        k_rxn: float = 0.4,
        k_growth: float = 0.02,
        c0: float = 1.0,
        pulse_time: float = 1.0,
        purge_time: float = 1.0,
        reaction_time: float = 1.0,
        post_purge_time: float = 1.0,
        num_cycles: int = 10,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.width = float(width)
        self.height = float(height)

        self.trench_left = float(trench_left)
        self.trench_right = float(trench_right)
        self.trench_top = float(trench_top)
        self.trench_bottom = float(trench_bottom)

        self.diffusivity = float(diffusivity)
        self.k_ads = float(k_ads)
        self.k_des = float(k_des)
        self.k_rxn = float(k_rxn)
        self.k_growth = float(k_growth)
        self.c0 = float(c0)

        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.post_purge_time = float(post_purge_time)
        self.num_cycles = int(num_cycles)

        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)
        self.x = np.linspace(0.0, self.width, self.nx)
        self.y = np.linspace(0.0, self.height, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.solid_mask, self.surface_mask = self._build_masks()
        self.dt = self._calculate_stable_dt()

        self.C = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.theta = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.h = np.zeros((self.ny, self.nx), dtype=np.float64)

        self.history: Dict[str, List[float]] = {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "surface_coverage": [],
        }

    def _build_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        solid = np.zeros((self.ny, self.nx), dtype=bool)

        left_mask = (self.X < self.trench_left) & (self.Y >= self.trench_top)
        right_mask = (self.X > self.trench_right) & (self.Y >= self.trench_top)
        bottom_mask = self.Y > self.trench_bottom
        solid = left_mask | right_mask | bottom_mask

        surface = np.zeros((self.ny, self.nx), dtype=bool)
        for i in range(self.ny):
            for j in range(self.nx):
                if not solid[i, j]:
                    neighbors = []
                    if i > 0:
                        neighbors.append(solid[i - 1, j])
                    if i < self.ny - 1:
                        neighbors.append(solid[i + 1, j])
                    if j > 0:
                        neighbors.append(solid[i, j - 1])
                    if j < self.nx - 1:
                        neighbors.append(solid[i, j + 1])
                    if any(neighbors):
                        surface[i, j] = True

        return solid, surface

    def _calculate_stable_dt(self) -> float:
        min_cell_dim = min(self.dx, self.dy)
        dt_diffusion = 0.20 * (min_cell_dim**2) / max(self.diffusivity, 1e-12)
        max_rate = max(self.k_ads, self.k_des, self.k_rxn, 1e-12)
        dt_kinetics = 0.20 / max_rate
        dt = min(dt_diffusion, dt_kinetics)
        cfl_threshold = (min_cell_dim**2) / (4.0 * max(self.diffusivity, 1e-12))
        if dt >= cfl_threshold:
            raise ValueError(
                f"Calculated dt ({dt:.2e}) violates explicit diffusive CFL stability ({cfl_threshold:.2e})."
            )
        return dt

    def _apply_boundary_conditions(self, c_array: np.ndarray, inlet_c: float) -> np.ndarray:
        c_array[0, ~self.solid_mask[0, :]] = inlet_c
        c_array[self.solid_mask] = 0.0
        c_array[:, 0] = c_array[:, 1]
        c_array[:, -1] = c_array[:, -2]
        c_array[-1, :] = c_array[-2, :]
        return c_array

    def step_diffusion(self, c_array: np.ndarray, dt_sub: float, inlet_c: float) -> np.ndarray:
        c_current = self._apply_boundary_conditions(c_array, inlet_c)
        laplacian = np.zeros_like(c_current)

        d2x = (c_current[1:-1, 2:] - 2.0 * c_current[1:-1, 1:-1] + c_current[1:-1, :-2]) / (
            self.dx**2
        )
        d2y = (c_current[2:, 1:-1] - 2.0 * c_current[1:-1, 1:-1] + c_current[:-2, 1:-1]) / (
            self.dy**2
        )
        laplacian[1:-1, 1:-1] = d2x + d2y

        c_next = c_current + dt_sub * self.diffusivity * laplacian
        c_next = self._apply_boundary_conditions(c_next, inlet_c)
        return np.maximum(c_next, 0.0)

    def run_pulse_stage(self, duration: Optional[float] = None) -> None:
        stage_time = self.pulse_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))
        dt_sub = stage_time / steps

        for _ in range(steps):
            self.C = self.step_diffusion(self.C, dt_sub, self.c0)
            r_ads = self.k_ads * self.C * (1.0 - self.theta)
            r_des = self.k_des * self.theta

            d_theta = (r_ads - r_des) * dt_sub
            self.theta[self.surface_mask] += d_theta[self.surface_mask]
            self.theta = np.clip(self.theta, 0.0, 1.0)

            sink_amount = r_ads * dt_sub
            self.C[self.surface_mask] = np.maximum(
                self.C[self.surface_mask] - sink_amount[self.surface_mask], 0.0
            )

    def run_purge_stage(self, duration: Optional[float] = None) -> None:
        stage_time = self.purge_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))
        dt_sub = stage_time / steps

        self.C[0, :] = 0.0
        for _ in range(steps):
            self.C = self.step_diffusion(self.C, dt_sub, 0.0)
            r_des = self.k_des * self.theta
            self.theta[self.surface_mask] -= (r_des * dt_sub)[self.surface_mask]
            self.theta = np.clip(self.theta, 0.0, 1.0)

        self.C.fill(0.0)

    def run_reaction_stage(self, duration: Optional[float] = None) -> None:
        stage_time = self.reaction_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))
        dt_sub = stage_time / steps

        for _ in range(steps):
            r_rxn = self.k_rxn * self.theta
            d_theta = r_rxn * dt_sub
            self.theta[self.surface_mask] -= d_theta[self.surface_mask]
            self.theta = np.clip(self.theta, 0.0, 1.0)
            self.h[self.surface_mask] += (self.k_growth * d_theta)[self.surface_mask]

    def run_cycle(self) -> None:
        self.run_pulse_stage()
        self.run_purge_stage()
        self.run_reaction_stage()
        self.run_purge_stage(duration=self.post_purge_time)

    def compute_metrics(
        self, current_cycle: int, previous_thickness: float
    ) -> Tuple[float, float, float]:
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]

        top_region = self.surface_mask & (self.Y <= self.trench_top + 2.0 * self.dy)
        bottom_region = self.surface_mask & (self.Y >= self.trench_bottom - 2.0 * self.dy)

        top_thickness = float(np.mean(self.h[top_region])) if np.any(top_region) else 0.0
        bottom_thickness = float(np.mean(self.h[bottom_region])) if np.any(bottom_region) else 0.0
        mean_thickness = float(np.mean(surface_h)) if surface_h.size > 0 else 0.0
        mean_coverage = float(np.mean(surface_theta)) if surface_theta.size > 0 else 0.0

        conformality = float(bottom_thickness / top_thickness) if top_thickness > 0.0 else 0.0
        gpc = float(mean_thickness - previous_thickness)

        self.history["cycle"].append(current_cycle)
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(mean_thickness)
        self.history["top_thickness"].append(top_thickness)
        self.history["bottom_thickness"].append(bottom_thickness)
        self.history["conformality"].append(conformality)
        self.history["surface_coverage"].append(mean_coverage)

        return gpc, conformality, mean_thickness

    def run(self) -> Dict[str, List[float]]:
        running_thickness = 0.0
        for cycle in range(1, self.num_cycles + 1):
            self.run_cycle()
            _, _, running_thickness = self.compute_metrics(cycle, running_thickness)
        return self.history

    def save_results(self, output_dir: Union[str, Path] = "results") -> Path:
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / "simulation.npz"

        np.savez_compressed(
            archive_path,
            x=self.x,
            y=self.y,
            concentration=self.C,
            surface_coverage=self.theta,
            film_thickness=self.h,
            surface_mask=self.surface_mask,
            solid_mask=self.solid_mask,
            cycle=np.asarray(self.history["cycle"]),
            gpc=np.asarray(self.history["gpc"]),
            mean_thickness=np.asarray(self.history["mean_thickness"]),
            top_thickness=np.asarray(self.history["top_thickness"]),
            bottom_thickness=np.asarray(self.history["bottom_thickness"]),
            conformality=np.asarray(self.history["conformality"]),
            cycle_coverage=np.asarray(self.history["surface_coverage"]),
        )
        return archive_path

    def plot_results(self, output_dir: Union[str, Path] = "results") -> None:
        figure_dir = Path(output_dir).resolve() / "figures" / "simulation"
        figure_dir.mkdir(parents=True, exist_ok=True)

        extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        concentration = self.C.copy()
        concentration[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(concentration, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        plt.colorbar(label="Precursor Concentration C (mol/m^3)")
        plt.xlabel("Horizontal Position x (m)")
        plt.ylabel("Vertical Position y (m)")
        plt.title("Precursor Concentration Profile")
        plt.tight_layout()
        plt.savefig(figure_dir / "precursor_concentration.png", dpi=200)
        plt.close()

        coverage = self.theta.copy()
        coverage[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(
            coverage,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="plasma",
        )
        plt.colorbar(label="Surface Coverage theta")
        plt.xlabel("Horizontal Position x (m)")
        plt.ylabel("Vertical Position y (m)")
        plt.title("Fractional Surface Coverage Profile")
        plt.tight_layout()
        plt.savefig(figure_dir / "surface_coverage.png", dpi=200)
        plt.close()

        thickness = self.h.copy()
        thickness[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(thickness, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        plt.colorbar(label="Deposited Film Thickness h (m)")
        plt.xlabel("Horizontal Position x (m)")
        plt.ylabel("Vertical Position y (m)")
        plt.title("Deposited Film Thickness Distribution")
        plt.tight_layout()
        plt.savefig(figure_dir / "film_thickness.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(
            self.history["cycle"],
            self.history["mean_thickness"],
            marker="o",
            label="Mean Surface Thickness",
        )
        plt.plot(
            self.history["cycle"],
            self.history["top_thickness"],
            marker="s",
            label="Top Region Thickness",
        )
        plt.plot(
            self.history["cycle"],
            self.history["bottom_thickness"],
            marker="^",
            label="Bottom Region Thickness",
        )
        plt.xlabel("Cycle Index")
        plt.ylabel("Film Thickness (m)")
        plt.title("Film Thickness Evolution per ALD Cycle")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "thickness_evolution.png", dpi=200)
        plt.close()


def main() -> None:
    simulator = ThinFilmDepositionSimulator(
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        pulse_time=1.0,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=10,
    )

    print("Initiating continuum ALD simulation...")
    print(f"Spatial Discretization: {simulator.nx} x {simulator.ny} nodes")
    print(f"Discretized Time Step dt: {simulator.dt:.6e} s")
    print(f"Total Cycles: {simulator.num_cycles}")

    simulator.run()

    print("\nCycle Summary Metrics")
    print("-" * 55)
    for i, cycle in enumerate(simulator.history["cycle"]):
        print(
            f"Cycle {int(cycle):02d} | "
            f"GPC = {simulator.history['gpc'][i]:.6e} m/cycle | "
            f"Conformality = {simulator.history['conformality'][i]:.4f}"
        )

    out_file = simulator.save_results()
    simulator.plot_results()

    print("\nSimulation completed.")
    print(f"Archived dataset written to: {out_file}")


if __name__ == "__main__":
    main()

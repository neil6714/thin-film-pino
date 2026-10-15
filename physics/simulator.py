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
        Vertical position of trench floor in meters.
    diffusivity : float, default=5.0e-3
        Precursor gas diffusion coefficient D in m^2/s.
    k_ads : float, default=1.0
        First-order adsorption rate constant k_ads in s^-1.
    k_des : float, default=0.05
        First-order desorption rate constant k_des in s^-1.
    k_rxn : float, default=0.4
        Co-reactant reaction rate constant k_rxn in s^-1.
    k_growth : float, default=0.02
        Film thickness growth coefficient k_growth in meters/cycle.
    inlet_concentration : float, default=1.0
        Normalized precursor concentration C_0 at inlet boundary (y = 0).
    pulse_time : float, default=1.0
        Precursor exposure pulse duration in seconds.
    purge_time : float, default=1.0
        Purge duration in seconds.
    reaction_time : float, default=1.0
        Co-reactant reaction exposure duration in seconds.
    num_cycles : int, default=10
        Total cyclic ALD iterations.
    dt : float, optional
        Explicit time step in seconds. If None, computed from CFL and reaction stability limits.
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
        inlet_concentration: float = 1.0,
        pulse_time: float = 1.0,
        purge_time: float = 1.0,
        reaction_time: float = 1.0,
        num_cycles: int = 10,
        dt: Optional[float] = None,
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

        self.inlet_concentration = float(inlet_concentration)
        self.pulse_time = float(pulse_time)
        self.purge_time = float(purge_time)
        self.reaction_time = float(reaction_time)
        self.num_cycles = int(num_cycles)

        self.dx = self.width / (self.nx - 1)
        self.dy = self.height / (self.ny - 1)

        diffusion_dt = 0.20 * min(self.dx, self.dy) ** 2 / self.diffusivity
        reaction_dt = 0.20 / max(self.k_ads, self.k_des, self.k_rxn, 1.0e-12)
        stable_dt = min(diffusion_dt, reaction_dt)
        self.dt = float(stable_dt if dt is None else dt)

        diffusion_limit = min(self.dx, self.dy) ** 2 / (4.0 * self.diffusivity)
        if self.dt >= diffusion_limit:
            raise ValueError(
                f"Specified time step dt={self.dt:.4e} s violates explicit 2D diffusive CFL "
                f"stability limit of {diffusion_limit:.4e} s."
            )

        self.x = np.linspace(0.0, self.width, self.nx)
        self.y = np.linspace(0.0, self.height, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.solid_mask = self._build_geometry()
        self.gas_mask = ~self.solid_mask
        self.surface_mask = self._find_surface_cells()

        self.C = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.theta = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.h = np.zeros((self.ny, self.nx), dtype=np.float64)

        self.history: Dict[str, List[float]] = self._empty_history()

    def _empty_history(self) -> Dict[str, List[float]]:
        return {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "surface_coverage": [],
        }

    def _build_geometry(self) -> np.ndarray:
        solid = np.zeros((self.ny, self.nx), dtype=bool)
        substrate = self.Y >= self.trench_bottom
        outside_trench = (self.Y >= self.trench_top) & (
            (self.X < self.trench_left) | (self.X > self.trench_right)
        )
        solid[substrate] = True
        solid[outside_trench] = True
        return solid

    def _find_surface_cells(self) -> np.ndarray:
        neighboring_solid = np.zeros_like(self.solid_mask, dtype=bool)
        neighboring_solid[1:, :] |= self.solid_mask[:-1, :]
        neighboring_solid[:-1, :] |= self.solid_mask[1:, :]
        neighboring_solid[:, 1:] |= self.solid_mask[:, :-1]
        neighboring_solid[:, :-1] |= self.solid_mask[:, 1:]
        return self.gas_mask & neighboring_solid

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        up = field.copy()
        down = field.copy()
        left = field.copy()
        right = field.copy()

        up[1:, :] = np.where(self.gas_mask[:-1, :], field[:-1, :], field[1:, :])
        down[:-1, :] = np.where(self.gas_mask[1:, :], field[1:, :], field[:-1, :])
        left[:, 1:] = np.where(self.gas_mask[:, :-1], field[:, :-1], field[:, 1:])
        right[:, :-1] = np.where(self.gas_mask[:, 1:], field[:, 1:], field[:, :-1])

        laplacian = (up + down - 2.0 * field) / self.dy**2 + (left + right - 2.0 * field) / self.dx**2
        return np.where(self.gas_mask, laplacian, 0.0)

    def _num_steps(self, duration: float) -> int:
        return max(1, int(np.ceil(duration / self.dt)))

    def _transport_step(self, pulse: bool) -> None:
        diffusion = self.diffusivity * self._laplacian(self.C)
        adsorption = np.zeros_like(self.C)
        desorption = np.zeros_like(self.theta)

        if pulse:
            adsorption[self.surface_mask] = (
                self.k_ads * self.C[self.surface_mask] * (1.0 - self.theta[self.surface_mask])
            )
            desorption[self.surface_mask] = self.k_des * self.theta[self.surface_mask]

        self.C += self.dt * (diffusion - adsorption)
        self.C = np.maximum(self.C, 0.0)
        self.C[~self.gas_mask] = 0.0

        if pulse:
            self.theta += self.dt * (adsorption - desorption)
            self.theta = np.clip(self.theta, 0.0, 1.0)
            self.C[0, :] = self.inlet_concentration
        else:
            self.C[0, :] = 0.0

    def _pulse(self) -> None:
        self.C[0, :] = self.inlet_concentration
        for _ in range(self._num_steps(self.pulse_time)):
            self._transport_step(pulse=True)

    def _purge(self) -> None:
        self.C[0, :] = 0.0
        for _ in range(self._num_steps(self.purge_time)):
            self._transport_step(pulse=False)
        self.C.fill(0.0)

    def _reaction(self) -> None:
        for _ in range(self._num_steps(self.reaction_time)):
            reaction = np.zeros_like(self.theta)
            reaction[self.surface_mask] = self.k_rxn * self.theta[self.surface_mask]
            self.h = np.maximum(self.h + self.dt * self.k_growth * reaction, 0.0)
            self.theta = np.clip(self.theta - self.dt * reaction, 0.0, 1.0)

    def _surface_metrics(self) -> Tuple[float, float, float, float, float]:
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]

        top_region = self.surface_mask & (self.Y < self.trench_top + 2.5 * self.dy)
        bottom_region = self.surface_mask & (self.Y > self.trench_bottom - 2.5 * self.dy)

        top_thickness = float(np.mean(self.h[top_region])) if np.any(top_region) else 0.0
        bottom_thickness = float(np.mean(self.h[bottom_region])) if np.any(bottom_region) else 0.0
        mean_thickness = float(np.mean(surface_h)) if surface_h.size else 0.0
        mean_coverage = float(np.mean(surface_theta)) if surface_theta.size else 0.0
        conformality = bottom_thickness / top_thickness if top_thickness > 0.0 else 0.0

        return mean_thickness, mean_coverage, top_thickness, bottom_thickness, conformality

    def _record_cycle(self, cycle: int, previous_thickness: float) -> float:
        (
            mean_thickness,
            mean_coverage,
            top_thickness,
            bottom_thickness,
            conformality,
        ) = self._surface_metrics()

        gpc = mean_thickness - previous_thickness

        self.history["cycle"].append(float(cycle))
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(mean_thickness)
        self.history["top_thickness"].append(top_thickness)
        self.history["bottom_thickness"].append(bottom_thickness)
        self.history["conformality"].append(conformality)
        self.history["surface_coverage"].append(mean_coverage)

        return mean_thickness

    def run_cycle(self) -> None:
        """Execute a complete 4-step ALD cycle: pulse, purge, reaction, purge."""
        self._pulse()
        self._purge()
        self._reaction()
        self._purge()

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full sequence of ALD cycles from initial clean substrate state.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (concentration, surface_coverage, film_thickness) fields at final time.
        """
        self.C.fill(0.0)
        self.theta.fill(0.0)
        self.h.fill(0.0)

        self.history = self._empty_history()
        previous_thickness = 0.0

        for cycle in range(1, self.num_cycles + 1):
            self.run_cycle()
            previous_thickness = self._record_cycle(cycle, previous_thickness)

        return self.C, self.theta, self.h

    def save_results(self, output_dir: Union[str, Path] = "results") -> Path:
        """Serialize simulation fields and cycle histories to compressed NPZ archive.

        Parameters
        ----------
        output_dir : str or Path, default="results"
            Destination directory for the output file.

        Returns
        -------
        Path
            Path to the saved NPZ archive.
        """
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / "stage1f_simulation.npz"

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
        """Render diagnostic contour plots of final physical fields and cycle trends.

        Parameters
        ----------
        output_dir : str or Path, default="results"
            Parent directory under which figure artifacts are generated.
        """
        figure_dir = Path(output_dir).resolve() / "figures" / "stage1f"
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
        plt.imshow(coverage, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=1.0, cmap="plasma")
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
        plt.imshow(thickness, origin="lower", extent=extent, aspect="auto", cmap="magma")
        plt.colorbar(label="Film Thickness h (m)")
        plt.xlabel("Horizontal Position x (m)")
        plt.ylabel("Vertical Position y (m)")
        plt.title("Deposited Film Thickness Profile")
        plt.tight_layout()
        plt.savefig(figure_dir / "film_thickness.png", dpi=200)
        plt.close()

        cycles = self.history["cycle"]
        plt.figure(figsize=(8, 5))
        plt.plot(cycles, self.history["mean_thickness"], label="Mean Surface Thickness", marker="o")
        plt.plot(cycles, self.history["top_thickness"], label="Top Trench Thickness", marker="s")
        plt.plot(cycles, self.history["bottom_thickness"], label="Bottom Trench Thickness", marker="^")
        plt.xlabel("Cycle Number")
        plt.ylabel("Film Thickness (m)")
        plt.title("Film Thickness Evolution Across ALD Cycles")
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

    print("\nSimulation completed successfully.")
    print(f"Archived dataset written to: {out_file}")


if __name__ == "__main__":
    main()
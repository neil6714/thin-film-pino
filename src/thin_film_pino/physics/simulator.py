"""2D continuum atomic layer deposition (ALD) transport and surface reaction simulator.

Discretizes transient precursor vapor diffusion, self-limiting Langmuir surface
adsorption, purge desorption, and co-reactant surface reaction in 2D trench geometries.
Uses an explicit forward-time central-space (FTCS) finite-difference scheme governed by
diffusive von Neumann and reaction kinetic stability limits.

Modelling Assumptions
---------------------
1. Operator Splitting: Each cycle splits into discrete stages: precursor dose, inert purge,
   co-reactant reaction, and post-reaction purge.
2. Perfect Purge: Gas-phase precursor concentration is cleared (C = 0) after each purge.
   The nominal purge time of 1.0 s is much shorter than the diffusion timescale L^2/D
   (25 to 200 s); clearance is an idealized operational assumption rather than a diffusive result.
3. Reaction Isolation: Adsorption occurs only during precursor dose. Co-reactant surface
   reaction occurs only during the reaction phase. Desorption is inactive during purge.
4. Mass Non-Conservation: Mass is not conserved. The precursor surface sink
   does not depend on absolute surface site density, and desorption does not return vapor
   into the gas domain.
5. Incomplete Reaction Carryover: With k_rxn in [0.1, 0.8] s^-1 and reaction_time = 1.0 s,
   at most 55 percent of chemisorbed precursor reacts per cycle, and remaining unreacted
   species carry over into the subsequent dose.
6. Numerical Bounds: Non-negativity C >= 0 and fractional coverage bounds theta in [0, 1]
   are enforced via numerical clamping and tracked with activation counters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.config import SimulationConfig
from thin_film_pino.physics.geometry import (
    build_geometry,
    compute_stable_dt,
    laplacian_numpy,
    validate_dt,
)


class ThinFilmDepositionSimulator:
    """Continuum finite-difference simulator for cyclic ALD in 2D trench geometries.

    Solves the nondimensional transport-reaction system:
        dC/dt = D grad^2 C - R_ads(x, y, t)
        dtheta/dt = R_ads - R_des - R_rxn
        dh/dt = k_growth * R_rxn

    with Langmuir kinetics:
        R_ads = k_ads * C * (1 - theta)
        R_des = k_des * theta
        R_rxn = k_rxn * theta

    subject to Dirichlet boundary conditions at the inlet (y = 0) and zero-flux
    homogeneous Neumann conditions along solid surfaces and walls.

    Parameters
    ----------
    config : SimulationConfig, optional
        Pre-configured simulation parameter dataclass. If provided, overrides kwargs.
    **kwargs
        Individual parameter overrides matching SimulationConfig fields.
    """

    def __init__(self, config: Optional[SimulationConfig] = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SimulationConfig(**kwargs)

        self.nx = self.config.nx
        self.ny = self.config.ny
        self.lx = self.config.lx
        self.ly = self.config.ly

        self.trench_left = self.config.trench_left
        self.trench_right = self.config.trench_right
        self.trench_top = self.config.trench_top
        self.trench_bottom = self.config.trench_bottom

        self.diffusivity = self.config.diffusivity
        self.k_ads = self.config.k_ads
        self.k_des = self.config.k_des
        self.k_rxn = self.config.k_rxn
        self.k_growth = self.config.k_growth
        self.c0 = self.config.inlet_concentration

        self.pulse_time = self.config.pulse_time
        self.purge_time = self.config.purge_time
        self.reaction_time = self.config.reaction_time
        self.post_purge_time = self.config.post_purge_time
        self.num_cycles = self.config.num_cycles

        # Build grid and region masks
        geom = build_geometry(
            nx=self.nx,
            ny=self.ny,
            lx=self.lx,
            ly=self.ly,
            trench_left=self.trench_left,
            trench_right=self.trench_right,
            trench_top=self.trench_top,
            trench_bottom=self.trench_bottom,
        )
        self.x = geom.x
        self.y = geom.y
        self.dx = geom.dx
        self.dy = geom.dy
        self.solid_mask = geom.solid_mask
        self.gas_mask = geom.gas_mask
        self.surface_mask = geom.surface_mask
        self.field_mask = geom.field_mask
        self.sidewall_mask = geom.sidewall_mask
        self.floor_mask = geom.floor_mask

        # Time step determination
        if self.config.dt is not None:
            validate_dt(
                self.config.dt,
                self.dx,
                self.dy,
                self.diffusivity,
                self.k_ads,
                self.k_des,
                self.k_rxn,
            )
            self.dt = float(self.config.dt)
        else:
            self.dt = compute_stable_dt(
                self.dx,
                self.dy,
                self.diffusivity,
                self.k_ads,
                self.k_des,
                self.k_rxn,
            )

        dtype = np.float64 if self.config.dtype == "float64" else np.float32
        self.C = np.zeros((self.ny, self.nx), dtype=dtype)
        self.theta = np.zeros((self.ny, self.nx), dtype=dtype)
        self.dose_theta = np.zeros((self.ny, self.nx), dtype=dtype)
        self.h = np.zeros((self.ny, self.nx), dtype=dtype)

        # Numerical clamping activation counters
        self.clamp_counts: Dict[str, int] = {
            "c_negative": 0,
            "theta_negative": 0,
            "theta_exceeded": 0,
        }

        self.history: Dict[str, List[float]] = {
            "cycle": [],
            "gpc": [],
            "mean_thickness": [],
            "field_thickness": [],
            "sidewall_thickness": [],
            "floor_thickness": [],
            "top_thickness": [],
            "bottom_thickness": [],
            "conformality": [],
            "step_coverage": [],
            "dose_coverage": [],
            "surface_coverage": [],
        }

    @property
    def clamp_stats(self) -> Dict[str, int]:
        """Return total numerical clamp activations recorded during execution."""
        return dict(self.clamp_counts)

    def _apply_boundary_conditions(self, c_array: np.ndarray, inlet_c: float) -> np.ndarray:
        c_array[0, self.gas_mask[0, :]] = inlet_c
        c_array[self.solid_mask] = 0.0
        c_array[:, 0] = c_array[:, 1]
        c_array[:, -1] = c_array[:, -2]
        c_array[-1, :] = c_array[-2, :]
        return c_array

    def step_diffusion(self, c_array: np.ndarray, dt_sub: float, inlet_c: float) -> np.ndarray:
        c_current = self._apply_boundary_conditions(c_array, inlet_c)
        lap = laplacian_numpy(c_current, self.dx, self.dy, self.gas_mask)
        c_next = c_current + dt_sub * self.diffusivity * lap
        c_next = self._apply_boundary_conditions(c_next, inlet_c)

        neg = (c_next[self.gas_mask] < 0.0).sum()
        if neg > 0:
            self.clamp_counts["c_negative"] += int(neg)
        return np.maximum(c_next, 0.0)

    def run_pulse_stage(self, duration: Optional[float] = None) -> float:
        """Execute precursor dose stage and return mean saturation coverage at dose conclusion."""
        stage_time = self.pulse_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))

        for _ in range(steps):
            self.C = self._apply_boundary_conditions(self.C, self.c0)
            lap = laplacian_numpy(self.C, self.dx, self.dy, self.gas_mask)
            diff = self.diffusivity * lap

            ads = np.zeros_like(self.C)
            des = np.zeros_like(self.C)
            c_surf = self.C[self.surface_mask]
            th_surf = self.theta[self.surface_mask]

            ads[self.surface_mask] = self.k_ads * c_surf * (1.0 - th_surf)
            des[self.surface_mask] = self.k_des * th_surf

            new_c = self.C + self.dt * (diff - ads)
            neg_c = (new_c[self.gas_mask] < 0.0).sum()
            if neg_c > 0:
                self.clamp_counts["c_negative"] += int(neg_c)

            new_c = np.maximum(new_c, 0.0)
            new_c = np.where(self.gas_mask, new_c, 0.0)
            self.C = self._apply_boundary_conditions(new_c, self.c0)

            new_theta = self.theta + self.dt * (ads - des)
            neg_th = (new_theta[self.surface_mask] < 0.0).sum()
            exc_th = (new_theta[self.surface_mask] > 1.0).sum()
            if neg_th > 0:
                self.clamp_counts["theta_negative"] += int(neg_th)
            if exc_th > 0:
                self.clamp_counts["theta_exceeded"] += int(exc_th)

            self.theta = np.clip(new_theta, 0.0, 1.0)
            self.theta = np.where(self.surface_mask, self.theta, 0.0)

        # Record saturation coverage at end of dose
        self.dose_theta = self.theta.copy()
        surf_theta = self.theta[self.surface_mask]
        return float(np.mean(surf_theta)) if surf_theta.size > 0 else 0.0

    def run_purge_stage(self, duration: Optional[float] = None) -> None:
        """Execute inert purge stage with idealized clearance of gas precursor."""
        stage_time = self.purge_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))

        for _ in range(steps):
            self.C = self._apply_boundary_conditions(self.C, 0.0)
            lap = laplacian_numpy(self.C, self.dx, self.dy, self.gas_mask)
            new_c = self.C + self.dt * self.diffusivity * lap
            neg = (new_c[self.gas_mask] < 0.0).sum()
            if neg > 0:
                self.clamp_counts["c_negative"] += int(neg)
            new_c = np.maximum(new_c, 0.0)
            new_c = np.where(self.gas_mask, new_c, 0.0)
            self.C = self._apply_boundary_conditions(new_c, 0.0)

        # Perfect purge operational assumption: reset vapor concentration
        self.C.fill(0.0)

    def run_reaction_stage(self, duration: Optional[float] = None) -> None:
        """Execute co-reactant exposure and surface film growth stage."""
        stage_time = self.reaction_time if duration is None else duration
        steps = max(1, int(np.ceil(stage_time / self.dt)))

        for _ in range(steps):
            th_surf = self.theta[self.surface_mask]
            d_theta = self.dt * self.k_rxn * th_surf
            new_th = th_surf - d_theta

            neg = (new_th < 0.0).sum()
            if neg > 0:
                self.clamp_counts["theta_negative"] += int(neg)

            self.theta[self.surface_mask] = np.clip(new_th, 0.0, 1.0)
            self.h[self.surface_mask] += self.k_growth * d_theta

    def run_cycle(self) -> float:
        """Execute all four stages of a single ALD cycle and return dose saturation coverage."""
        dose_cov = self.run_pulse_stage()
        self.run_purge_stage()
        self.run_reaction_stage()
        self.run_purge_stage(duration=self.post_purge_time)
        return dose_cov

    def compute_metrics(
        self, current_cycle: int, previous_thickness: float, dose_cov: float
    ) -> Tuple[float, float, float]:
        """Compute cycle metrics: regional thicknesses, step coverage, and GPC."""
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]

        field_thickness = (
            float(np.mean(self.h[self.field_mask])) if np.any(self.field_mask) else 0.0
        )
        floor_thickness = (
            float(np.mean(self.h[self.floor_mask])) if np.any(self.floor_mask) else 0.0
        )
        sidewall_thickness = (
            float(np.mean(self.h[self.sidewall_mask])) if np.any(self.sidewall_mask) else 0.0
        )

        mean_thickness = float(np.mean(surface_h)) if surface_h.size > 0 else 0.0
        mean_cycle_coverage = float(np.mean(surface_theta)) if surface_theta.size > 0 else 0.0

        step_coverage = float(floor_thickness / field_thickness) if field_thickness > 0.0 else 0.0
        gpc = float(mean_thickness - previous_thickness)

        self.history["cycle"].append(float(current_cycle))
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(mean_thickness)
        self.history["field_thickness"].append(field_thickness)
        self.history["sidewall_thickness"].append(sidewall_thickness)
        self.history["floor_thickness"].append(floor_thickness)
        # Compatibility aliases for top and bottom
        self.history["top_thickness"].append(field_thickness)
        self.history["bottom_thickness"].append(floor_thickness)
        self.history["step_coverage"].append(step_coverage)
        self.history["conformality"].append(step_coverage)
        self.history["dose_coverage"].append(dose_cov)
        self.history["surface_coverage"].append(mean_cycle_coverage)

        return gpc, step_coverage, mean_thickness

    def run(self) -> Dict[str, List[float]]:
        """Run complete cyclic ALD sequence and return summary histories."""
        running_thickness = 0.0
        for cycle in range(1, self.num_cycles + 1):
            dose_cov = self.run_cycle()
            _, _, running_thickness = self.compute_metrics(cycle, running_thickness, dose_cov)
        return self.history

    def save_results(
        self, output_dir: Union[str, Path] = "results", run_name: str = "simulation"
    ) -> Path:
        """Save archived arrays and cycle metrics to disk."""
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / f"{run_name}.npz"

        np.savez_compressed(
            archive_path,
            x=self.x,
            y=self.y,
            concentration=self.C,
            surface_coverage=self.theta,
            film_thickness=self.h,
            surface_mask=self.surface_mask,
            solid_mask=self.solid_mask,
            field_mask=self.field_mask,
            sidewall_mask=self.sidewall_mask,
            floor_mask=self.floor_mask,
            cycle=np.asarray(self.history["cycle"]),
            gpc=np.asarray(self.history["gpc"]),
            mean_thickness=np.asarray(self.history["mean_thickness"]),
            field_thickness=np.asarray(self.history["field_thickness"]),
            floor_thickness=np.asarray(self.history["floor_thickness"]),
            sidewall_thickness=np.asarray(self.history["sidewall_thickness"]),
            top_thickness=np.asarray(self.history["top_thickness"]),
            bottom_thickness=np.asarray(self.history["bottom_thickness"]),
            step_coverage=np.asarray(self.history["step_coverage"]),
            conformality=np.asarray(self.history["conformality"]),
            dose_coverage=np.asarray(self.history["dose_coverage"]),
            cycle_coverage=np.asarray(self.history["surface_coverage"]),
            clamp_stats=self.clamp_stats,
        )
        return archive_path

    def plot_results(
        self, output_dir: Union[str, Path] = "results", run_name: str = "simulation"
    ) -> None:
        """Render diagnostic contour plots with masked substrate regions."""
        figure_dir = Path(output_dir).resolve() / "figures" / run_name
        figure_dir.mkdir(parents=True, exist_ok=True)

        extent = [
            float(self.x.min()),
            float(self.x.max()),
            float(self.y.min()),
            float(self.y.max()),
        ]

        cmap_c = matplotlib.colormaps["viridis"].copy()
        cmap_c.set_bad(color="lightgray")
        cmap_theta = matplotlib.colormaps["plasma"].copy()
        cmap_theta.set_bad(color="lightgray")
        cmap_h = matplotlib.colormaps["inferno"].copy()
        cmap_h.set_bad(color="lightgray")

        # Nondimensional Concentration C / C0
        c_plot = self.C.copy()
        c_plot[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(
            c_plot / max(self.c0, 1e-12), origin="lower", extent=extent, aspect="auto", cmap=cmap_c
        )
        plt.colorbar(label="Precursor Concentration C / C_0")
        plt.xlabel("Horizontal Coordinate x / L_x")
        plt.ylabel("Vertical Coordinate y / L_y")
        plt.title("Precursor Concentration Profile (Nondimensional)")
        plt.tight_layout()
        plt.savefig(figure_dir / "precursor_concentration.png", dpi=200)
        plt.close()

        # Surface Coverage theta
        cov_plot = self.theta.copy()
        cov_plot[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(
            cov_plot,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap_theta,
        )
        plt.colorbar(label="Fractional Surface Coverage theta")
        plt.xlabel("Horizontal Coordinate x / L_x")
        plt.ylabel("Vertical Coordinate y / L_y")
        plt.title("Surface Coverage Profile")
        plt.tight_layout()
        plt.savefig(figure_dir / "surface_coverage.png", dpi=200)
        plt.close()

        # Film Thickness h
        h_plot = self.h.copy()
        h_plot[self.solid_mask] = np.nan
        plt.figure(figsize=(8, 5))
        plt.imshow(h_plot, origin="lower", extent=extent, aspect="auto", cmap=cmap_h)
        plt.colorbar(label="Film Thickness h (length units)")
        plt.xlabel("Horizontal Coordinate x / L_x")
        plt.ylabel("Vertical Coordinate y / L_y")
        plt.title("Deposited Film Thickness Distribution")
        plt.tight_layout()
        plt.savefig(figure_dir / "film_thickness.png", dpi=200)
        plt.close()

        # Thickness evolution per cycle
        plt.figure(figsize=(8, 4))
        plt.plot(
            self.history["cycle"], self.history["mean_thickness"], marker="o", label="Mean Surface"
        )
        plt.plot(
            self.history["cycle"], self.history["field_thickness"], marker="s", label="Planar Field"
        )
        plt.plot(
            self.history["cycle"], self.history["floor_thickness"], marker="^", label="Trench Floor"
        )
        plt.plot(
            self.history["cycle"], self.history["sidewall_thickness"], marker="d", label="Sidewalls"
        )
        plt.xlabel("ALD Cycle Index")
        plt.ylabel("Film Thickness (length units)")
        plt.title("Film Thickness Evolution per ALD Cycle")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "thickness_evolution.png", dpi=200)
        plt.close()

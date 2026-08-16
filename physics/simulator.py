import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ThinFilmDepositionSimulator:
    def __init__(
        self,
        nx=160,
        ny=120,
        width=1.0,
        height=1.0,
        trench_left=0.375,
        trench_right=0.625,
        trench_top=0.50,
        trench_bottom=0.95,
        diffusivity=5.0e-3,
        k_ads=1.0,
        k_des=0.05,
        k_rxn=0.4,
        k_growth=0.02,
        inlet_concentration=1.0,
        pulse_time=1.0,
        purge_time=1.0,
        reaction_time=1.0,
        num_cycles=10,
        dt=None,
    ):
        self.nx = nx
        self.ny = ny

        self.width = width
        self.height = height

        self.trench_left = trench_left
        self.trench_right = trench_right
        self.trench_top = trench_top
        self.trench_bottom = trench_bottom

        self.diffusivity = diffusivity
        self.k_ads = k_ads
        self.k_des = k_des
        self.k_rxn = k_rxn
        self.k_growth = k_growth

        self.inlet_concentration = inlet_concentration
        self.pulse_time = pulse_time
        self.purge_time = purge_time
        self.reaction_time = reaction_time
        self.num_cycles = num_cycles

        self.dx = width / (nx - 1)
        self.dy = height / (ny - 1)

        diffusion_dt = (
            0.20
            * min(self.dx, self.dy) ** 2
            / diffusivity
        )

        reaction_dt = 0.20 / max(
            k_ads,
            k_des,
            k_rxn,
            1e-12,
        )

        stable_dt = min(diffusion_dt, reaction_dt)
        self.dt = stable_dt if dt is None else dt

        diffusion_limit = (
            min(self.dx, self.dy) ** 2
            / (4.0 * diffusivity)
        )

        if self.dt >= diffusion_limit:
            raise ValueError(
                "dt is too large for the explicit diffusion scheme."
            )

        self.x = np.linspace(0.0, width, nx)
        self.y = np.linspace(0.0, height, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.solid_mask = self._build_geometry()
        self.gas_mask = ~self.solid_mask
        self.surface_mask = self._find_surface_cells()

        self.C = np.zeros((ny, nx), dtype=np.float64)
        self.theta = np.zeros((ny, nx), dtype=np.float64)
        self.h = np.zeros((ny, nx), dtype=np.float64)

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
        solid = np.zeros((self.ny, self.nx), dtype=bool)

        substrate = self.Y >= self.trench_bottom

        outside_trench = (
            (self.Y >= self.trench_top)
            & (
                (self.X < self.trench_left)
                | (self.X > self.trench_right)
            )
        )

        solid[substrate] = True
        solid[outside_trench] = True

        return solid

    def _find_surface_cells(self):
        neighboring_solid = np.zeros_like(
            self.solid_mask,
            dtype=bool,
        )

        neighboring_solid[1:, :] |= self.solid_mask[:-1, :]
        neighboring_solid[:-1, :] |= self.solid_mask[1:, :]
        neighboring_solid[:, 1:] |= self.solid_mask[:, :-1]
        neighboring_solid[:, :-1] |= self.solid_mask[:, 1:]

        return (
            self.gas_mask
            & neighboring_solid
        )

    def _laplacian(self, field):
        center = field

        up = center.copy()
        down = center.copy()
        left = center.copy()
        right = center.copy()

        valid_up = self.gas_mask[:-1, :]
        valid_down = self.gas_mask[1:, :]
        valid_left = self.gas_mask[:, :-1]
        valid_right = self.gas_mask[:, 1:]

        up[1:, :][valid_up] = field[:-1, :][valid_up]
        down[:-1, :][valid_down] = field[1:, :][valid_down]
        left[:, 1:][valid_left] = field[:, :-1][valid_left]
        right[:, :-1][valid_right] = field[:, 1:][valid_right]

        laplacian = (
            (up + down - 2.0 * center) / self.dy**2
            + (left + right - 2.0 * center) / self.dx**2
        )

        laplacian[~self.gas_mask] = 0.0

        return laplacian

    def _num_steps(self, duration):
        return max(
            1,
            int(np.ceil(duration / self.dt)),
        )

    def _transport_step(self, pulse):
        diffusion = (
            self.diffusivity
            * self._laplacian(self.C)
        )

        adsorption = np.zeros_like(self.C)
        desorption = np.zeros_like(self.theta)

        if pulse:
            adsorption[self.surface_mask] = (
                self.k_ads
                * self.C[self.surface_mask]
                * (1.0 - self.theta[self.surface_mask])
            )

            desorption[self.surface_mask] = (
                self.k_des
                * self.theta[self.surface_mask]
            )

        self.C += self.dt * (
            diffusion - adsorption
        )

        self.C = np.maximum(
            self.C,
            0.0,
        )

        self.C[~self.gas_mask] = 0.0

        if pulse:
            self.theta += self.dt * (
                adsorption - desorption
            )

            self.theta = np.clip(
                self.theta,
                0.0,
                1.0,
            )

            self.C[0, :] = self.inlet_concentration
        else:
            self.C[0, :] = 0.0

    def _pulse(self):
        self.C[0, :] = self.inlet_concentration

        for _ in range(
            self._num_steps(self.pulse_time)
        ):
            self._transport_step(pulse=True)

    def _purge(self):
        self.C[0, :] = 0.0

        for _ in range(
            self._num_steps(self.purge_time)
        ):
            self._transport_step(pulse=False)

        self.C.fill(0.0)

    def _reaction(self):
        for _ in range(
            self._num_steps(self.reaction_time)
        ):
            reaction = np.zeros_like(self.theta)

            reaction[self.surface_mask] = (
                self.k_rxn
                * self.theta[self.surface_mask]
            )

            growth_rate = (
                self.k_growth
                * reaction
            )

            self.h += self.dt * growth_rate

            self.theta -= self.dt * reaction

            self.theta = np.clip(
                self.theta,
                0.0,
                1.0,
            )

    def _surface_metrics(self):
        surface_h = self.h[self.surface_mask]
        surface_theta = self.theta[self.surface_mask]

        top_region = self.surface_mask & (
            self.Y < self.trench_top + 2.5 * self.dy
        )

        bottom_region = self.surface_mask & (
            self.Y > self.trench_bottom - 2.5 * self.dy
        )

        top_thickness = (
            np.mean(self.h[top_region])
            if np.any(top_region)
            else 0.0
        )

        bottom_thickness = (
            np.mean(self.h[bottom_region])
            if np.any(bottom_region)
            else 0.0
        )

        mean_thickness = (
            np.mean(surface_h)
            if surface_h.size
            else 0.0
        )

        mean_coverage = (
            np.mean(surface_theta)
            if surface_theta.size
            else 0.0
        )

        conformality = (
            bottom_thickness / top_thickness
            if top_thickness > 0.0
            else 0.0
        )

        return (
            mean_thickness,
            mean_coverage,
            top_thickness,
            bottom_thickness,
            conformality,
        )

    def _record_cycle(
        self,
        cycle,
        previous_thickness,
    ):
        (
            mean_thickness,
            mean_coverage,
            top_thickness,
            bottom_thickness,
            conformality,
        ) = self._surface_metrics()

        gpc = (
            mean_thickness
            - previous_thickness
        )

        self.history["cycle"].append(cycle)
        self.history["gpc"].append(gpc)
        self.history["mean_thickness"].append(
            mean_thickness
        )
        self.history["top_thickness"].append(
            top_thickness
        )
        self.history["bottom_thickness"].append(
            bottom_thickness
        )
        self.history["conformality"].append(
            conformality
        )
        self.history["surface_coverage"].append(
            mean_coverage
        )

        return mean_thickness

    def run_cycle(self):
        self._pulse()
        self._purge()
        self._reaction()
        self._purge()

    def run(self):
        self.C.fill(0.0)
        self.theta.fill(0.0)
        self.h.fill(0.0)

        self.history = self._empty_history()

        previous_thickness = 0.0

        for cycle in range(
            1,
            self.num_cycles + 1,
        ):
            self.run_cycle()

            previous_thickness = self._record_cycle(
                cycle,
                previous_thickness,
            )

        return (
            self.C,
            self.theta,
            self.h,
        )

    def save_results(self, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(
                output_dir,
                "stage1f_simulation.npz",
            ),
            x=self.x,
            y=self.y,
            concentration=self.C,
            surface_coverage=self.theta,
            film_thickness=self.h,
            surface_mask=self.surface_mask,
            solid_mask=self.solid_mask,
            cycle=np.asarray(
                self.history["cycle"]
            ),
            gpc=np.asarray(
                self.history["gpc"]
            ),
            mean_thickness=np.asarray(
                self.history["mean_thickness"]
            ),
            top_thickness=np.asarray(
                self.history["top_thickness"]
            ),
            bottom_thickness=np.asarray(
                self.history["bottom_thickness"]
            ),
            conformality=np.asarray(
                self.history["conformality"]
            ),
            cycle_coverage=np.asarray(
                self.history["surface_coverage"]
            ),
        )

    def plot_results(self, output_dir="results"):
        figure_dir = os.path.join(
            output_dir,
            "figures",
            "stage1f",
        )

        os.makedirs(
            figure_dir,
            exist_ok=True,
        )

        concentration = self.C.copy()
        concentration[self.solid_mask] = np.nan

        plt.figure(figsize=(8, 5))
        plt.imshow(
            concentration,
            origin="lower",
            extent=[
                self.x.min(),
                self.x.max(),
                self.y.min(),
                self.y.max(),
            ],
            aspect="auto",
        )
        plt.colorbar(
            label="Precursor concentration"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            "Final Precursor Concentration"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "concentration.png",
            ),
            dpi=200,
        )
        plt.close()

        coverage = self.theta.copy()
        coverage[~self.surface_mask] = np.nan

        plt.figure(figsize=(8, 5))
        plt.imshow(
            coverage,
            origin="lower",
            extent=[
                self.x.min(),
                self.x.max(),
                self.y.min(),
                self.y.max(),
            ],
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(
            label="Surface coverage"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            "Final Surface Coverage"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "surface_coverage.png",
            ),
            dpi=200,
        )
        plt.close()

        thickness = self.h.copy()
        thickness[~self.surface_mask] = np.nan

        plt.figure(figsize=(8, 5))
        plt.imshow(
            thickness,
            origin="lower",
            extent=[
                self.x.min(),
                self.x.max(),
                self.y.min(),
                self.y.max(),
            ],
            aspect="auto",
        )
        plt.colorbar(
            label="Film thickness"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            "Final Film Thickness"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "film_thickness.png",
            ),
            dpi=200,
        )
        plt.close()

        cycles = np.asarray(
            self.history["cycle"]
        )

        plt.figure(figsize=(8, 5))
        plt.plot(
            cycles,
            self.history["gpc"],
            marker="o",
        )
        plt.xlabel("ALD cycle")
        plt.ylabel("Growth per cycle")
        plt.title("Growth Per Cycle")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "gpc.png",
            ),
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            cycles,
            self.history["conformality"],
            marker="o",
        )
        plt.xlabel("ALD cycle")
        plt.ylabel("Conformality")
        plt.title(
            "Conformality Evolution"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "conformality.png",
            ),
            dpi=200,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            cycles,
            self.history["top_thickness"],
            marker="o",
            label="Top",
        )
        plt.plot(
            cycles,
            self.history["bottom_thickness"],
            marker="o",
            label="Bottom",
        )
        plt.xlabel("ALD cycle")
        plt.ylabel("Film thickness")
        plt.title(
            "Thickness Evolution"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                figure_dir,
                "thickness_evolution.png",
            ),
            dpi=200,
        )
        plt.close()


def main():
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

    print("Starting cyclic ALD simulation...")
    print(f"Grid: {simulator.nx} x {simulator.ny}")
    print(f"dt = {simulator.dt:.6e}")
    print(f"Cycles = {simulator.num_cycles}")

    simulator.run()

    print("\nCycle results")
    print("-" * 55)

    for i, cycle in enumerate(
        simulator.history["cycle"]
    ):
        print(
            f"Cycle {cycle:02d} | "
            f"GPC={simulator.history['gpc'][i]:.6e} | "
            f"Conformality="
            f"{simulator.history['conformality'][i]:.4f}"
        )

    simulator.save_results()
    simulator.plot_results()

    print("\nSimulation complete.")
    print("Results saved to:")
    print(
        "  results/stage1f_simulation.npz"
    )
    print(
        "  results/figures/stage1f/"
    )


if __name__ == "__main__":
    main()
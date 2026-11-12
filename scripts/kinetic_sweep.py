"""2D kinetic parameter sweeps: diffusivity vs. reaction rate and adsorption vs. desorption."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def run_simulation(
    diffusivity: float, k_rxn: float, k_ads: float, k_des: float
) -> Tuple[float, float, float, float]:
    """Execute single simulation instance and return final thickness and conformality metrics."""
    simulator = ThinFilmDepositionSimulator(
        diffusivity=diffusivity,
        k_rxn=k_rxn,
        k_ads=k_ads,
        k_des=k_des,
        k_growth=0.02,
    )
    simulator.run()

    top = float(simulator.history["top_thickness"][-1])
    bottom = float(simulator.history["bottom_thickness"][-1])
    conformality = bottom / top if top > 0.0 else 0.0
    mean_thickness = float(simulator.h[simulator.surface_mask].mean())

    return top, bottom, mean_thickness, conformality


def run_sweep_d_rxn(
    grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep diffusivity D against surface reaction rate k_rxn."""
    diffusivities = np.linspace(1.0e-3, 8.0e-3, grid_size)
    reaction_rates = np.linspace(0.1, 0.8, grid_size)

    conformality = np.zeros((grid_size, grid_size))
    top_thickness = np.zeros((grid_size, grid_size))
    bottom_thickness = np.zeros((grid_size, grid_size))
    mean_thickness = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    step = 0
    print(f"Sweeping Diffusivity vs Reaction Rate ({grid_size}x{grid_size} grid)...")

    for i, d in enumerate(diffusivities):
        for j, rxn in enumerate(reaction_rates):
            step += 1
            top, bottom, mean, conf = run_simulation(
                diffusivity=d, k_rxn=rxn, k_ads=1.0, k_des=0.05
            )
            conformality[j, i] = conf
            top_thickness[j, i] = top
            bottom_thickness[j, i] = bottom
            mean_thickness[j, i] = mean

            if step % 8 == 0 or step == total:
                print(f"  [{step:02d}/{total:02d}] D={d:.2e}, k_rxn={rxn:.2f} -> Conf={conf:.4f}")

    return (
        diffusivities,
        reaction_rates,
        conformality,
        top_thickness,
        bottom_thickness,
        mean_thickness,
    )


def run_sweep_ads_des(
    grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep adsorption rate k_ads against thermal desorption rate k_des."""
    k_ads_vals = np.linspace(0.5, 1.5, grid_size)
    k_des_vals = np.linspace(0.02, 0.10, grid_size)

    conformality = np.zeros((grid_size, grid_size))
    top_thickness = np.zeros((grid_size, grid_size))
    bottom_thickness = np.zeros((grid_size, grid_size))
    mean_thickness = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    step = 0
    print(f"\nSweeping Adsorption vs Desorption ({grid_size}x{grid_size} grid)...")

    for i, ads in enumerate(k_ads_vals):
        for j, des in enumerate(k_des_vals):
            step += 1
            top, bottom, mean, conf = run_simulation(
                diffusivity=5.0e-3, k_rxn=0.4, k_ads=ads, k_des=des
            )
            conformality[j, i] = conf
            top_thickness[j, i] = top
            bottom_thickness[j, i] = bottom
            mean_thickness[j, i] = mean

            if step % 8 == 0 or step == total:
                print(
                    f"  [{step:02d}/{total:02d}] k_ads={ads:.2f}, k_des={des:.3f} -> Conf={conf:.4f}"
                )

    return (
        k_ads_vals,
        k_des_vals,
        conformality,
        top_thickness,
        bottom_thickness,
        mean_thickness,
    )


def plot_map(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    data: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    plt.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    plt.colorbar(label="Conformality (Bottom / Top)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute 2D parametric sweeps over transport and reaction kinetics."
    )
    parser.add_argument(
        "--grid-size", type=int, default=8, help="Number of sweep points per dimension."
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/kinetic_sweep",
        help="Directory to save sweep diagnostic contour maps.",
    )
    parser.add_argument(
        "--output-npz",
        default="data/kinetic_sweep.npz",
        help="Path for archived sweep array output.",
    )
    args = parser.parse_args()

    (
        d_vals,
        rxn_vals,
        d_rxn_conf,
        d_rxn_top,
        d_rxn_bottom,
        d_rxn_mean,
    ) = run_sweep_d_rxn(grid_size=args.grid_size)

    (
        ads_vals,
        des_vals,
        ads_des_conf,
        ads_des_top,
        ads_des_bottom,
        ads_des_mean,
    ) = run_sweep_ads_des(grid_size=args.grid_size)

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        diffusivities=d_vals,
        reaction_rates=rxn_vals,
        k_ads_vals=ads_vals,
        k_des_vals=des_vals,
        d_rxn_conformality=d_rxn_conf,
        d_rxn_top_thickness=d_rxn_top,
        d_rxn_bottom_thickness=d_rxn_bottom,
        d_rxn_mean_thickness=d_rxn_mean,
        ads_des_conformality=ads_des_conf,
        ads_des_top_thickness=ads_des_top,
        ads_des_bottom_thickness=ads_des_bottom,
        ads_des_mean_thickness=ads_des_mean,
    )

    plot_map(
        d_vals,
        rxn_vals,
        d_rxn_conf,
        "Diffusivity D (m^2/s)",
        "Reaction Rate Constant k_rxn (s^-1)",
        "Step Coverage Conformality: D vs k_rxn",
        output_dir / "diffusivity_reaction_conformality.png",
    )

    plot_map(
        ads_vals,
        des_vals,
        ads_des_conf,
        "Adsorption Rate Constant k_ads (s^-1)",
        "Desorption Rate Constant k_des (s^-1)",
        "Step Coverage Conformality: k_ads vs k_des",
        output_dir / "adsorption_desorption_conformality.png",
    )

    print("\nKinetic sensitivity sweep completed.")
    print(f"Data archived to: {out_npz}")
    print(f"Diagnostic figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

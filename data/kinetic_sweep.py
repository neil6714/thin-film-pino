"""2D kinetic parameter sweeps: diffusivity vs. reaction rate and adsorption vs. desorption."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator


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

    conformality = np.zeros((grid_size, grid_size), dtype=np.float64)
    top_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
    bottom_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
    mean_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i, k_rxn in enumerate(reaction_rates):
        for j, diffusivity in enumerate(diffusivities):
            top, bottom, mean_h, conf = run_simulation(
                diffusivity=diffusivity,
                k_rxn=k_rxn,
                k_ads=1.0,
                k_des=0.05,
            )

            conformality[i, j] = conf
            top_thickness[i, j] = top
            bottom_thickness[i, j] = bottom
            mean_thickness[i, j] = mean_h

            print(
                f"[D-k_rxn] D={diffusivity:.4e} m^2/s, k_rxn={k_rxn:.3f} s^-1, conformality={conf:.4f}"
            )

    return diffusivities, reaction_rates, conformality, top_thickness, bottom_thickness, mean_thickness


def run_sweep_ads_des(
    grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep adsorption rate k_ads against thermal desorption rate k_des."""
    adsorption_rates = np.linspace(0.5, 1.5, grid_size)
    desorption_rates = np.linspace(0.02, 0.10, grid_size)

    conformality = np.zeros((grid_size, grid_size), dtype=np.float64)
    top_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
    bottom_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
    mean_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i, k_des in enumerate(desorption_rates):
        for j, k_ads in enumerate(adsorption_rates):
            top, bottom, mean_h, conf = run_simulation(
                diffusivity=5.0e-3,
                k_rxn=0.4,
                k_ads=k_ads,
                k_des=k_des,
            )

            conformality[i, j] = conf
            top_thickness[i, j] = top
            bottom_thickness[i, j] = bottom
            mean_thickness[i, j] = mean_h

            print(
                f"[k_ads-k_des] k_ads={k_ads:.3f} s^-1, k_des={k_des:.4f} s^-1, conformality={conf:.4f}"
            )

    return adsorption_rates, desorption_rates, conformality, top_thickness, bottom_thickness, mean_thickness


def plot_map(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Render 2D pseudocolor response surface heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(
        z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Step Coverage Conformality (Bottom / Top)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute 2D kinetic sensitivity parameter sweeps across diffusivity and rate constants."
    )
    parser.add_argument("--grid-size", type=int, default=8, help="Resolution along each parameter axis.")
    parser.add_argument(
        "--output-dir",
        default="results/figures/kinetic_sweep",
        help="Destination directory for response surface plots.",
    )
    parser.add_argument(
        "--output-npz",
        default="data/kinetic_sweep.npz",
        help="Path for saved sweep metrics .npz archive.",
    )
    args = parser.parse_args()

    print("Beginning kinetic parameter sensitivity sweep...")
    d_vals, rxn_vals, d_rxn_conf, d_rxn_top, d_rxn_bottom, d_rxn_mean = run_sweep_d_rxn(args.grid_size)
    ads_vals, des_vals, ads_des_conf, ads_des_top, ads_des_bottom, ads_des_mean = run_sweep_ads_des(args.grid_size)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_npz = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        diffusivities=d_vals,
        reaction_rates=rxn_vals,
        adsorption_rates=ads_vals,
        desorption_rates=des_vals,
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

    print("\nKinetic sensitivity sweep completed successfully.")
    print(f"Data archived to: {out_npz}")
    print(f"Diagnostic figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
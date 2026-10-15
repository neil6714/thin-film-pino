"""Parameter sweep mapping step coverage conformality over precursor diffusivity and reaction rates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.simulator import ThinFilmDepositionSimulator


def run_sweep(grid_size: int = 8, output_dir: str = "results/figures/parameter_sweep", output_npz: str = "data/parameter_sweep.npz") -> None:
    """Execute grid sweep over diffusivity and reaction rate, serializing metrics and heatmaps."""
    diffusivities = np.linspace(1.0e-3, 8.0e-3, grid_size)
    reaction_rates = np.linspace(0.1, 0.8, grid_size)
    conformality = np.zeros((len(reaction_rates), len(diffusivities)), dtype=np.float64)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(output_npz).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for i, k_rxn in enumerate(reaction_rates):
        for j, diffusivity in enumerate(diffusivities):
            simulator = ThinFilmDepositionSimulator(
                diffusivity=diffusivity,
                k_rxn=k_rxn,
                k_ads=1.0,
                k_des=0.05,
                k_growth=0.02,
            )
            simulator.run()

            top = float(simulator.history["top_thickness"][-1])
            bottom = float(simulator.history["bottom_thickness"][-1])

            if top > 0.0:
                conformality[i, j] = bottom / top

            print(f"D={diffusivity:.4e} m^2/s, k_rxn={k_rxn:.3f} s^-1, conformality={conformality[i, j]:.4f}")

    np.savez_compressed(
        out_file,
        diffusivities=diffusivities,
        reaction_rates=reaction_rates,
        conformality=conformality,
    )

    plt.figure(figsize=(8, 6))
    image = plt.imshow(
        conformality,
        origin="lower",
        aspect="auto",
        extent=[
            diffusivities.min(),
            diffusivities.max(),
            reaction_rates.min(),
            reaction_rates.max(),
        ],
        cmap="viridis",
    )
    plt.colorbar(image, label="Step Coverage Conformality (Bottom / Top)")
    plt.xlabel("Diffusion Coefficient D (m^2/s)")
    plt.ylabel("Reaction Rate Constant k_rxn (s^-1)")
    plt.title("Conformality Parameter Response Surface")
    plt.tight_layout()
    plt.savefig(out_dir / "conformality_parameter_map.png", dpi=200)
    plt.close()

    best_idx = np.unravel_index(np.argmax(conformality), conformality.shape)
    worst_idx = np.unravel_index(np.argmin(conformality), conformality.shape)

    print("\nParameter sweep completed successfully.")
    print(f"Maximum conformality: {conformality[best_idx]:.4f} (D={diffusivities[best_idx[1]]:.4e}, k_rxn={reaction_rates[best_idx[0]]:.3f})")
    print(f"Minimum conformality: {conformality[worst_idx]:.4f} (D={diffusivities[worst_idx[1]]:.4e}, k_rxn={reaction_rates[worst_idx[0]]:.3f})")
    print(f"Archived dataset: {out_file}")
    print(f"Diagnostic figure: {out_dir / 'conformality_parameter_map.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ALD diffusivity and reaction rates to map step coverage.")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid points per parameter axis.")
    parser.add_argument("--output-dir", default="results/figures/parameter_sweep", help="Directory for output figures.")
    parser.add_argument("--output-npz", default="data/parameter_sweep.npz", help="Path for archived .npz results.")
    args = parser.parse_args()

    run_sweep(grid_size=args.grid_size, output_dir=args.output_dir, output_npz=args.output_npz)


if __name__ == "__main__":
    main()
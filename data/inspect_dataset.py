import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "data/deposition_dataset.npz"
OUTPUT_DIR = "results/figures/dataset_inspection"


def print_summary(data):
    print("Dataset summary")
    print("-" * 50)

    for key in data.files:
        array = data[key]
        print(
            f"{key:20s} "
            f"shape={array.shape!s:18s} "
            f"dtype={array.dtype}"
        )

    print("\nParameter statistics")
    print("-" * 50)

    names = [
        "diffusivity",
        "k_ads",
        "k_des",
        "k_rxn",
        "k_growth",
    ]

    parameters = data["parameters"]

    for i, name in enumerate(names):
        values = parameters[:, i]
        print(
            f"{name:15s} "
            f"min={values.min():.6e} "
            f"max={values.max():.6e} "
            f"mean={values.mean():.6e} "
            f"std={values.std():.6e}"
        )

    print("\nOutput statistics")
    print("-" * 50)

    for name in [
        "concentration",
        "surface_coverage",
        "film_thickness",
        "top_thickness",
        "bottom_thickness",
        "conformality",
    ]:
        values = data[name]
        print(
            f"{name:20s} "
            f"min={values.min():.6e} "
            f"max={values.max():.6e} "
            f"mean={values.mean():.6e} "
            f"std={values.std():.6e}"
        )


def validate_physics(data):
    concentration = data["concentration"]
    surface_coverage = data["surface_coverage"]
    film_thickness = data["film_thickness"]
    conformality = data["conformality"]

    checks = {
        "Concentration >= 0": np.all(concentration >= 0.0),
        "0 <= Surface coverage <= 1": np.all(
            (surface_coverage >= 0.0)
            & (surface_coverage <= 1.0)
        ),
        "Film thickness >= 0": np.all(film_thickness >= 0.0),
        "Conformality >= 0": np.all(conformality >= 0.0),
    }

    print("\nPhysical validation")
    print("-" * 50)

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:5s}  {name}")


def plot_conformality(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conformality = data["conformality"]

    plt.figure(figsize=(8, 5))
    plt.hist(conformality, bins=20)
    plt.xlabel("Conformality")
    plt.ylabel("Number of simulations")
    plt.title("Conformality Distribution")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "conformality_distribution.png"),
        dpi=200,
    )
    plt.close()


def plot_sample(data, index):
    concentration = data["concentration"][index]
    coverage = data["surface_coverage"][index]
    thickness = data["film_thickness"][index]

    parameter_values = data["parameters"][index]
    conformality = data["conformality"][index]

    labels = [
        "Diffusivity",
        "Adsorption rate",
        "Desorption rate",
        "Reaction rate",
        "Growth coefficient",
    ]

    print(f"\nSample {index}")
    print("-" * 50)

    for label, value in zip(labels, parameter_values):
        print(f"{label:20s}: {value:.6e}")

    print(f"Conformality        : {conformality:.6f}")

    plots = [
        (concentration, "Precursor Concentration", "concentration"),
        (coverage, "Surface Coverage", "surface_coverage"),
        (thickness, "Film Thickness", "film_thickness"),
    ]

    for field, title, filename in plots:
        plt.figure(figsize=(8, 5))

        if filename == "surface_coverage":
            plt.imshow(
                field,
                origin="lower",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
        else:
            plt.imshow(
                field,
                origin="lower",
                aspect="auto",
            )

        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"{title} — Sample {index}"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"sample_{index}_{filename}.png",
            ),
            dpi=200,
        )
        plt.close()


def analyze_variation(data):
    parameters = data["parameters"]
    thickness = data["film_thickness"]
    conformality = data["conformality"]

    mean_thickness = thickness.mean(axis=(1, 2))

    print("\nDataset variation")
    print("-" * 50)

    print(
        f"Mean film thickness range: "
        f"{mean_thickness.min():.6e} "
        f"to {mean_thickness.max():.6e}"
    )

    print(
        f"Conformality range: "
        f"{conformality.min():.6f} "
        f"to {conformality.max():.6f}"
    )

    print(
        f"Parameter variation present: "
        f"{np.all(np.std(parameters, axis=0) > 0)}"
    )

    print(
        f"Output variation present: "
        f"{np.std(mean_thickness) > 0}"
    )


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = np.load(DATA_PATH)

    print_summary(data)
    validate_physics(data)
    analyze_variation(data)

    plot_conformality(data)

    sample_indices = [
        0,
        len(data["parameters"]) // 2,
        len(data["parameters"]) - 1,
    ]

    for index in sample_indices:
        plot_sample(data, index)

    print("\nInspection complete.")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
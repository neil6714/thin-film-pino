"""Visualizations comparing predicted and ground-truth thin-film physical fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_field_comparison_trajectory(
    ref_field: np.ndarray,
    pred_field: np.ndarray,
    mask: np.ndarray,
    solid_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    time_values: np.ndarray,
    time_indices: Sequence[int],
    field_name: str,
    field_unit: str,
    output_path: Union[str, Path],
    phase_labels: Optional[Sequence[str]] = None,
) -> None:
    """Plot multi-snapshot comparison panels (Reference, Prediction, Absolute Error).

    Parameters
    ----------
    ref_field : np.ndarray
        Ground-truth field of shape (T, H, W) in physical units.
    pred_field : np.ndarray
        Predicted field of shape (T, H, W) in physical units.
    mask : np.ndarray
        Active domain mask of shape (H, W).
    solid_mask : np.ndarray
        Solid geometry mask of shape (H, W).
    x_coords : np.ndarray
        Spatial horizontal coordinates (W,).
    y_coords : np.ndarray
        Spatial vertical coordinates (H,).
    time_values : np.ndarray
        Time coordinate values (T,).
    time_indices : Sequence[int]
        Specific time indices to plot.
    field_name : str
        Human-readable name of the physical field (e.g., 'Concentration').
    field_unit : str
        Physical unit string (e.g., 'mol/m^3' or 'dimensionless').
    output_path : Union[str, Path]
        Path to save output figure.
    phase_labels : Optional[Sequence[str]], default=None
        Labels corresponding to the chosen time indices.
    """
    n_rows = len(time_indices)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.2 * n_rows), squeeze=False)
    extent = [float(x_coords[0]), float(x_coords[-1]), float(y_coords[-1]), float(y_coords[0])]

    mask_bool = mask.astype(bool)
    solid_bool = solid_mask.astype(bool)

    # Determine global reference color limits across active cells
    active_ref = ref_field[time_indices][:, mask_bool]
    vmin = float(np.min(active_ref)) if active_ref.size > 0 else 0.0
    vmax = float(np.max(active_ref)) if active_ref.size > 0 else 1.0
    if abs(vmax - vmin) < 1.0e-8:
        vmax = vmin + 1.0

    cmap_field = "viridis" if field_name.lower() == "concentration" else "plasma"
    cmap_err = "magma"

    for row_idx, t_idx in enumerate(time_indices):
        t_val = float(time_values[t_idx])
        phase_str = f" ({phase_labels[row_idx]})" if phase_labels else ""
        t_title = f"t = {t_val:.2f} s{phase_str}"

        # 1. Reference array with masked inactive domain
        r_img = np.ma.masked_where(~mask_bool, ref_field[t_idx])
        p_img = np.ma.masked_where(~mask_bool, pred_field[t_idx])
        err_img = np.ma.masked_where(~mask_bool, np.abs(pred_field[t_idx] - ref_field[t_idx]))

        # Reference
        ax_ref = axes[row_idx, 0]
        # Background: gray for solid
        ax_ref.imshow(solid_bool, extent=extent, cmap="Greys", alpha=0.3, vmin=0, vmax=1)
        im_ref = ax_ref.imshow(
            r_img, extent=extent, cmap=cmap_field, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax_ref.set_ylabel(f"{t_title}\ny (m)")
        if row_idx == 0:
            ax_ref.set_title(f"Reference {field_name}")
        if row_idx == n_rows - 1:
            ax_ref.set_xlabel("x (m)")
        fig.colorbar(im_ref, ax=ax_ref, fraction=0.046, pad=0.04, label=field_unit)

        # Prediction
        ax_pred = axes[row_idx, 1]
        ax_pred.imshow(solid_bool, extent=extent, cmap="Greys", alpha=0.3, vmin=0, vmax=1)
        im_pred = ax_pred.imshow(
            p_img, extent=extent, cmap=cmap_field, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        if row_idx == 0:
            ax_pred.set_title(f"Predicted {field_name}")
        if row_idx == n_rows - 1:
            ax_pred.set_xlabel("x (m)")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04, label=field_unit)

        # Absolute Error
        ax_err = axes[row_idx, 2]
        ax_err.imshow(solid_bool, extent=extent, cmap="Greys", alpha=0.3, vmin=0, vmax=1)
        im_err = ax_err.imshow(err_img, extent=extent, cmap=cmap_err, interpolation="nearest")
        if row_idx == 0:
            ax_err.set_title("Absolute Error")
        if row_idx == n_rows - 1:
            ax_err.set_xlabel("x (m)")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04, label=field_unit)

    plt.tight_layout()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def generate_qualitative_figures(
    data: Dict[str, Any],
    pred_physical: np.ndarray,
    trajectory_index: int,
    output_dir: Union[str, Path],
    num_time_samples: int = 4,
) -> List[Path]:
    """Generate qualitative comparison figures for a single test trajectory across fields.

    Parameters
    ----------
    data : Dict[str, Any]
        In-memory dataset dictionary.
    pred_physical : np.ndarray
        Array of shape (T, 3, H, W) containing predictions for this trajectory.
    trajectory_index : int
        Trajectory index within the test subset.
    output_dir : Union[str, Path]
        Directory to save generated figures.
    num_time_samples : int, default=4
        Number of time snapshots to visualize across the trajectory.

    Returns
    -------
    List[Path]
        Paths of saved figures.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_indices = data["test_indices"]
    raw_traj_idx = test_indices[trajectory_index]

    f_mean = data["field_mean"]
    f_std = data["field_std"]
    gas_mask = data["gas_mask"].astype(bool)
    surf_mask = data["surface_mask"].astype(bool)
    solid_mask = data["solid_mask"].astype(bool)

    x_coords = data["x"]
    y_coords = data["y"]
    time_arr = data["time"][raw_traj_idx]
    n_time = len(time_arr)

    # Reference fields in physical units
    ref_c = data["concentration"][raw_traj_idx] * f_std[0] + f_mean[0]
    ref_c[:, ~gas_mask] = 0.0

    ref_theta = data["surface_coverage"][raw_traj_idx] * f_std[1] + f_mean[1]
    ref_theta[:, ~surf_mask] = 0.0

    ref_h = data["film_thickness"][raw_traj_idx] * f_std[2] + f_mean[2]
    ref_h[:, ~surf_mask] = 0.0

    # Pick representative time indices across trajectory
    phase_arr = data["phase"][raw_traj_idx] if "phase" in data else None
    phase_names_map = {1: "dose", 2: "purge1", 3: "reaction", 4: "purge2"}

    time_indices: List[int] = []
    phase_labels: List[str] = []

    if phase_arr is not None:
        # Pick one representative snapshot from each distinct phase
        for p_val in (1, 2, 3, 4):
            matches = np.where(phase_arr == p_val)[0]
            if len(matches) > 0:
                mid_idx = matches[len(matches) // 2]
                time_indices.append(int(mid_idx))
                phase_labels.append(phase_names_map.get(p_val, f"phase_{p_val}"))

    if len(time_indices) < 2:
        # Uniform time sampling fallback
        time_indices = [
            int(idx) for idx in np.linspace(0, n_time - 1, num_time_samples, dtype=int)
        ]
        phase_labels = [f"step_{idx}" for idx in time_indices]

    created_plots: List[Path] = []

    fields_to_plot = [
        ("concentration", ref_c, pred_physical[:, 0], gas_mask, "Concentration", "mol/m^3"),
        (
            "surface_coverage",
            ref_theta,
            pred_physical[:, 1],
            surf_mask,
            "Surface Coverage",
            "fraction",
        ),
        ("film_thickness", ref_h, pred_physical[:, 2], surf_mask, "Film Thickness", "m"),
    ]

    for f_id, ref_f, pred_f, mask_f, f_name, f_unit in fields_to_plot:
        out_path = out_dir / f"qualitative_{f_id}.png"
        plot_field_comparison_trajectory(
            ref_field=ref_f,
            pred_field=pred_f,
            mask=mask_f,
            solid_mask=solid_mask,
            x_coords=x_coords,
            y_coords=y_coords,
            time_values=time_arr,
            time_indices=time_indices,
            field_name=f_name,
            field_unit=f_unit,
            output_path=out_path,
            phase_labels=phase_labels,
        )
        created_plots.append(out_path)

    return created_plots

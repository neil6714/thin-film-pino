"""Spatial grid convergence study evaluating discretization error and observed convergence order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thin_film_pino.physics.geometry import compute_stable_dt
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for grid convergence study."""
    parser = argparse.ArgumentParser(
        description="Evaluate spatial discretization error and observed convergence order across mesh resolutions."
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["40x30", "80x60", "160x120"],
        help="List of resolutions (Nx x Ny) from coarsest to finest.",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="Number of ALD cycles to simulate.",
    )
    parser.add_argument(
        "--pulse-time",
        type=float,
        default=5.0,
        help="Precursor dose duration in seconds.",
    )
    parser.add_argument(
        "--output",
        default="results/convergence/grid_convergence.json",
        help="Path for saving convergence metrics JSON.",
    )
    return parser.parse_args(args)


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse resolution string 'Nx x Ny' into integer pair (nx, ny)."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format '{res_str}'. Expected 'Nx x Ny' (e.g. 80x60).")
    return int(parts[0]), int(parts[1])


def run_single_grid(
    nx: int,
    ny: int,
    pulse_time: float = 2.0,
    num_cycles: int = 1,
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute reference simulation for a specific grid resolution."""
    diffusivity = 1.0e-3
    k_ads = 1.0
    k_des = 0.1
    k_rxn = 0.5
    k_growth = 0.02
    purge_time = 1.0
    reaction_time = 1.0

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    if dt is None:
        dt = compute_stable_dt(
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
        )

    sim = ThinFilmDepositionSimulator(
        diffusivity=diffusivity,
        k_ads=k_ads,
        k_des=k_des,
        k_rxn=k_rxn,
        k_growth=k_growth,
        pulse_time=pulse_time,
        purge_time=purge_time,
        reaction_time=reaction_time,
        num_cycles=num_cycles,
        nx=nx,
        ny=ny,
        dt=dt,
        dtype="float64",
    )
    sim.run()

    h_surf = sim.h[sim.surface_mask]
    mean_h = float(np.mean(h_surf)) if h_surf.size > 0 else 0.0
    field_h = float(np.mean(sim.h[sim.field_mask])) if np.any(sim.field_mask) else mean_h
    floor_h = float(np.mean(sim.h[sim.floor_mask])) if np.any(sim.floor_mask) else mean_h
    step_cov = float(floor_h / field_h) if field_h > 0.0 else 0.0

    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "dt": float(sim.dt),
        "mean_thickness": mean_h,
        "field_thickness": field_h,
        "floor_thickness": floor_h,
        "step_coverage": step_cov,
        "dose_coverage": float(sim.dose_theta.mean()) if hasattr(sim, "dose_theta") else 0.0,
    }


def compute_convergence_rates(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute relative errors against the finest grid and calculate observed order p."""
    if len(runs) < 3:
        raise ValueError("Grid convergence study requires at least 3 grid resolutions.")

    # Assume runs are ordered coarsest to finest: [coarse (4h), medium (2h), fine (h)]
    fine_run = runs[-1]
    metrics = ["mean_thickness", "step_coverage", "field_thickness", "floor_thickness"]
    convergence_summary: Dict[str, Any] = {"resolutions": [f"{r['nx']}x{r['ny']}" for r in runs]}

    for m in metrics:
        fine_val = fine_run[m]
        errors = [abs(r[m] - fine_val) for r in runs[:-1]]

        # Compute observed order of accuracy between the two coarser levels
        # p = ln(E_coarse / E_medium) / ln(r), where r = h_coarse / h_medium
        e_coarse = errors[0]
        e_med = errors[1]

        r_mesh = runs[0]["dx"] / runs[1]["dx"]
        if e_med > 1.0e-15 and e_coarse > 1.0e-15:
            observed_order = float(np.log(e_coarse / e_med) / np.log(r_mesh))
        else:
            observed_order = float("nan")

        convergence_summary[m] = {
            "fine_benchmark_value": fine_val,
            "coarse_error": float(e_coarse),
            "medium_error": float(e_med),
            "mesh_refinement_ratio": float(r_mesh),
            "observed_order_p": observed_order,
        }

    return convergence_summary


def format_convergence_table(summary: Dict[str, Any], runs: List[Dict[str, Any]]) -> str:
    """Format grid convergence study results as a technical markdown table."""
    coarse_res = f"{runs[0]['nx']}x{runs[0]['ny']}"
    med_res = f"{runs[1]['nx']}x{runs[1]['ny']}"
    fine_res = f"{runs[-1]['nx']}x{runs[-1]['ny']}"

    lines = [
        "=" * 85,
        "Spatial Grid Convergence Study (FTCS Central-Space Stencil)",
        f"Mesh Levels: Coarse ({coarse_res}) -> Medium ({med_res}) -> Fine ({fine_res})",
        "-" * 85,
        f"{'Quantity':<22} | {'Fine Benchmark':<16} | {f'Error ({coarse_res})':<16} | {f'Error ({med_res})':<16} | {'Observed Order (p)':<18}",
        "-" * 85,
    ]

    metric_names = [
        ("mean_thickness", "Mean Thickness (m)"),
        ("step_coverage", "Step Coverage (-)"),
        ("field_thickness", "Field Thickness (m)"),
        ("floor_thickness", "Floor Thickness (m)"),
    ]

    for m_key, m_label in metric_names:
        data = summary.get(m_key, {})
        fine_v = data.get("fine_benchmark_value", 0.0)
        e_c = data.get("coarse_error", 0.0)
        e_m = data.get("medium_error", 0.0)
        p_val = data.get("observed_order_p", float("nan"))
        p_str = f"{p_val:.2f}" if np.isfinite(p_val) else "N/A"
        lines.append(f"{m_label:<22} | {fine_v:<16.6e} | {e_c:<16.6e} | {e_m:<16.6e} | {p_str:<18}")

    lines.append("=" * 85)
    return "\n".join(lines)


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for spatial grid convergence study."""
    parsed = parse_args(args)
    res_list = [parse_resolution(r) for r in parsed.resolutions]
    print(f"Running grid convergence across resolutions: {parsed.resolutions}...")

    # Run from finest resolution to establish shared stable time step
    finest_nx, finest_ny = res_list[-1]
    dx_fine = 1.0 / (finest_nx - 1)
    dy_fine = 1.0 / (finest_ny - 1)
    shared_dt = compute_stable_dt(
        dx=dx_fine,
        dy=dy_fine,
        diffusivity=1.0e-3,
        k_ads=1.0,
        k_des=0.1,
        k_rxn=0.5,
    )

    runs: List[Dict[str, Any]] = []
    for nx, ny in res_list:
        print(f"  Simulating {nx}x{ny} grid (dt = {shared_dt:.4e} s)...")
        r = run_single_grid(
            nx=nx,
            ny=ny,
            pulse_time=parsed.pulse_time,
            num_cycles=parsed.num_cycles,
            dt=shared_dt,
        )
        runs.append(r)

    summary = compute_convergence_rates(runs)
    table_text = format_convergence_table(summary, runs)
    print("\n" + table_text)

    output_path = Path(parsed.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_data = {
        "summary": summary,
        "runs": runs,
        "shared_dt": shared_dt,
        "pulse_time": parsed.pulse_time,
        "num_cycles": parsed.num_cycles,
    }
    output_path.write_text(json.dumps(full_data, indent=2), encoding="utf-8")
    print(f"Grid convergence metrics written to: {output_path}")


if __name__ == "__main__":
    main()

"""Multi-sample parity verification comparing NumPy CPU reference and PyTorch solvers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from thin_film_pino.data.parameters import sample_parameter_ensemble
from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.geometry import compute_stable_dt
from thin_film_pino.physics.simulator import ThinFilmDepositionSimulator


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for parity verification."""
    parser = argparse.ArgumentParser(
        description="Verify numerical parity between NumPy CPU reference and PyTorch solvers."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of parameter sets to evaluate (at least 20).",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=20,
        help="Horizontal grid resolution.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=15,
        help="Vertical grid resolution.",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="Number of ALD cycles to evaluate.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration containing verification thresholds.",
    )
    parser.add_argument(
        "--max-abs-err-tol",
        type=float,
        default=None,
        help="Maximum absolute error tolerance threshold.",
    )
    parser.add_argument(
        "--rel-l2-tol",
        type=float,
        default=None,
        help="Relative L2 error tolerance threshold.",
    )
    parser.add_argument(
        "--output",
        default="results/parity/parity_metrics.json",
        help="Path to output JSON summary file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for parameter generation.",
    )
    return parser.parse_args(args)


def load_tolerances(
    config_path: Path,
    cli_abs_tol: Optional[float] = None,
    cli_rel_tol: Optional[float] = None,
) -> tuple[float, float]:
    """Load parity thresholds from config with CLI override fallback."""
    cfg_abs_tol = 1.0e-12
    cfg_rel_tol = 1.0e-12

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            parity_cfg = raw_cfg.get("parity", {})
            cfg_abs_tol = float(parity_cfg.get("max_abs_error_tol", cfg_abs_tol))
            cfg_rel_tol = float(parity_cfg.get("rel_l2_tol", cfg_rel_tol))
        except Exception as exc:
            print(f"Notice: Failed to load config tolerances ({exc}), using defaults.")

    max_abs_tol = cli_abs_tol if cli_abs_tol is not None else cfg_abs_tol
    rel_l2_tol = cli_rel_tol if cli_rel_tol is not None else cfg_rel_tol
    return max_abs_tol, rel_l2_tol


def run_parity_benchmark(
    parameters_batch: np.ndarray,
    nx: int = 20,
    ny: int = 15,
    num_cycles: int = 1,
    max_abs_tol: float = 1.0e-12,
    rel_l2_tol: float = 1.0e-12,
) -> Dict[str, Any]:
    """Execute trajectory parity verification across multiple parameter sets."""
    n_samples = len(parameters_batch)
    if n_samples < 1:
        raise ValueError("At least one parameter sample is required for parity verification.")

    device = torch.device("cpu")
    dtype = "float64"

    # Compute shared stable time step
    max_d = float(np.max(parameters_batch[:, 0]))
    max_k_ads = float(np.max(parameters_batch[:, 1]))
    max_k_des = float(np.max(parameters_batch[:, 2]))
    max_k_rxn = float(np.max(parameters_batch[:, 3]))
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    shared_dt = compute_stable_dt(
        dx=dx,
        dy=dy,
        diffusivity=max_d,
        k_ads=max_k_ads,
        k_des=max_k_des,
        k_rxn=max_k_rxn,
    )

    results: List[Dict[str, Any]] = []
    failed_samples: List[int] = []

    for i in range(n_samples):
        p = parameters_batch[i]
        diffusivity, k_ads, k_des, k_rxn, k_growth, pulse_time, purge_time, rxn_time = p

        # 1. Reference NumPy simulator
        ref_sim = ThinFilmDepositionSimulator(
            diffusivity=diffusivity,
            k_ads=k_ads,
            k_des=k_des,
            k_rxn=k_rxn,
            k_growth=k_growth,
            pulse_time=pulse_time,
            purge_time=purge_time,
            reaction_time=rxn_time,
            num_cycles=num_cycles,
            nx=nx,
            ny=ny,
            dt=shared_dt,
            dtype=dtype,
        )

        # 2. PyTorch accelerated simulator (float64 on CPU)
        fast_sim = BatchedTransientALDCudaSimulator(
            parameters=np.asarray([p]),
            nx=nx,
            ny=ny,
            device=device,
            dtype=dtype,
            dt=shared_dt,
        )

        pulse_steps = int(fast_sim.pulse_steps().item())
        purge_steps = int(fast_sim.fixed_steps(fast_sim.purge_time))
        reaction_steps = int(fast_sim.fixed_steps(fast_sim.reaction_time))
        active_mask = torch.ones(1, dtype=torch.bool, device=device)

        mid_pulse = max(1, pulse_steps // 2)
        sample_metrics: Dict[str, Any] = {"sample_index": i, "snapshots": {}}

        for cycle_idx in range(num_cycles):
            # --- Stage 1: Precursor pulse ---
            # 1a. Step until mid-dose (C is actively non-zero in gas cells)
            ref_sim.run_pulse_stage(duration=mid_pulse * shared_dt)
            for _ in range(mid_pulse):
                fast_sim.transport_step(active_mask, pulse=True)

            c_f, th_f, h_f = fast_sim.snapshot()
            diff_c_mid = np.abs(ref_sim.C - c_f[0])
            diff_th_mid = np.abs(ref_sim.theta - th_f[0])
            norm_c_ref = float(np.linalg.norm(ref_sim.C))
            sample_metrics["snapshots"]["mid_dose"] = {
                "c_max_abs_err": float(np.max(diff_c_mid)),
                "c_rel_l2": float(np.linalg.norm(diff_c_mid) / max(norm_c_ref, 1.0e-12)),
                "c_reference_norm": norm_c_ref,
                "theta_max_abs_err": float(np.max(diff_th_mid)),
            }

            # 1b. Step remainder of pulse
            rem_steps = pulse_steps - mid_pulse
            if rem_steps > 0:
                ref_sim.run_pulse_stage(duration=rem_steps * shared_dt)
                for _ in range(rem_steps):
                    fast_sim.transport_step(active_mask, pulse=True)

            c_f, th_f, h_f = fast_sim.snapshot()
            sample_metrics["snapshots"]["end_dose"] = {
                "theta_max_abs_err": float(np.max(np.abs(ref_sim.theta - th_f[0]))),
                "theta_rel_l2": float(
                    np.linalg.norm(ref_sim.theta - th_f[0])
                    / max(float(np.linalg.norm(ref_sim.theta)), 1.0e-12)
                ),
            }

            # --- Stage 2: Post-dose purge ---
            ref_sim.run_purge_stage()
            for _ in range(purge_steps):
                fast_sim.transport_step(active_mask, pulse=False)
            fast_sim.C.zero_()

            # --- Stage 3: Reaction stage ---
            ref_sim.run_reaction_stage()
            for _ in range(reaction_steps):
                fast_sim.reaction_step()

            c_f, th_f, h_f = fast_sim.snapshot()
            diff_h_rxn = np.abs(ref_sim.h - h_f[0])
            norm_h_ref = float(np.linalg.norm(ref_sim.h))
            sample_metrics["snapshots"]["end_reaction"] = {
                "h_max_abs_err": float(np.max(diff_h_rxn)),
                "h_rel_l2": float(np.linalg.norm(diff_h_rxn) / max(norm_h_ref, 1.0e-12)),
            }

            # --- Stage 4: Post-reaction purge ---
            ref_sim.run_purge_stage(duration=ref_sim.post_purge_time)
            for _ in range(purge_steps):
                fast_sim.transport_step(active_mask, pulse=False)
            fast_sim.C.zero_()

        # Final state comparison
        c_f, th_f, h_f = fast_sim.snapshot()
        err_c = np.max(np.abs(ref_sim.C - c_f[0]))
        err_th = np.max(np.abs(ref_sim.theta - th_f[0]))
        err_h = np.max(np.abs(ref_sim.h - h_f[0]))

        norm_th = float(np.linalg.norm(ref_sim.theta))
        norm_h = float(np.linalg.norm(ref_sim.h))
        rel_th = float(np.linalg.norm(ref_sim.theta - th_f[0]) / max(norm_th, 1.0e-12))
        rel_h = float(np.linalg.norm(ref_sim.h - h_f[0]) / max(norm_h, 1.0e-12))

        sample_metrics["final_state"] = {
            "concentration_max_abs_err": float(err_c),
            "surface_coverage_max_abs_err": float(err_th),
            "surface_coverage_rel_l2": rel_th,
            "film_thickness_max_abs_err": float(err_h),
            "film_thickness_rel_l2": rel_h,
        }

        # Check threshold compliance
        is_failed = (
            err_c > max_abs_tol
            or err_th > max_abs_tol
            or err_h > max_abs_tol
            or rel_th > rel_l2_tol
            or rel_h > rel_l2_tol
            or sample_metrics["snapshots"]["mid_dose"]["c_max_abs_err"] > max_abs_tol
        )
        sample_metrics["status"] = "failed" if is_failed else "passed"
        if is_failed:
            failed_samples.append(i)

        results.append(sample_metrics)

    max_c_err = max(r["final_state"]["concentration_max_abs_err"] for r in results)
    max_th_err = max(r["final_state"]["surface_coverage_max_abs_err"] for r in results)
    max_h_err = max(r["final_state"]["film_thickness_max_abs_err"] for r in results)
    max_mid_c_err = max(r["snapshots"]["mid_dose"]["c_max_abs_err"] for r in results)
    max_mid_c_rel = max(r["snapshots"]["mid_dose"]["c_rel_l2"] for r in results)

    summary = {
        "num_samples_evaluated": n_samples,
        "grid": [ny, nx],
        "dtype": dtype,
        "device": str(device),
        "dt": float(shared_dt),
        "tolerances": {
            "max_abs_error_tol": max_abs_tol,
            "rel_l2_tol": rel_l2_tol,
        },
        "worst_case_errors": {
            "mid_dose_concentration_max_abs_err": max_mid_c_err,
            "mid_dose_concentration_rel_l2": max_mid_c_rel,
            "final_concentration_max_abs_err": max_c_err,
            "final_coverage_max_abs_err": max_th_err,
            "final_thickness_max_abs_err": max_h_err,
        },
        "failed_samples": failed_samples,
        "all_passed": len(failed_samples) == 0,
        "per_sample": results,
    }
    return summary


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for parity verification."""
    parsed = parse_args(args)
    config_path = Path(parsed.config).resolve()
    max_abs_tol, rel_l2_tol = load_tolerances(
        config_path, parsed.max_abs_err_tol, parsed.rel_l2_tol
    )

    print(
        f"Starting parity verification across {parsed.num_samples} parameter sets "
        f"on {parsed.nx}x{parsed.ny} grid (float64 CPU)..."
    )

    params, _ = sample_parameter_ensemble(
        num_samples=parsed.num_samples,
        seed=parsed.seed,
    )

    summary = run_parity_benchmark(
        parameters_batch=params,
        nx=parsed.nx,
        ny=parsed.ny,
        num_cycles=parsed.num_cycles,
        max_abs_tol=max_abs_tol,
        rel_l2_tol=rel_l2_tol,
    )

    wc = summary["worst_case_errors"]
    print("Parity Verification Results (NumPy vs. PyTorch):")
    print(
        f"  - Mid-dose Concentration Max Abs Error: {wc['mid_dose_concentration_max_abs_err']:.4e} (Rel L2: {wc['mid_dose_concentration_rel_l2']:.4e})"
    )
    print(f"  - Final Concentration Max Abs Error:   {wc['final_concentration_max_abs_err']:.4e}")
    print(f"  - Final Surface Coverage Max Abs Error:{wc['final_coverage_max_abs_err']:.4e}")
    print(f"  - Final Film Thickness Max Abs Error:  {wc['final_thickness_max_abs_err']:.4e}")
    print(
        f"  - Evaluated: {summary['num_samples_evaluated']} samples | Status: {'PASSED' if summary['all_passed'] else 'FAILED'}"
    )

    out_path = Path(parsed.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Parity metrics written to: {out_path}")

    if not summary["all_passed"]:
        print(
            f"Error: Parity verification failed for {len(summary['failed_samples'])} samples exceeding tolerance.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Unified dataset generator for ALD simulation ensembles with reproducible SeedSequence chunking.

Supports both full temporal trajectory datasets (N, T, ny, nx) and final-state
datasets (N, ny, nx) after N ALD cycles, writing incrementally in chunks to avoid
high RAM usage.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from thin_film_pino.data.parameters import (
    PARAMETER_NAMES,
    PHASE_NAMES,
    load_priors,
    sample_parameter_ensemble,
)
from thin_film_pino.physics.batched_transient_cuda import BatchedTransientALDCudaSimulator
from thin_film_pino.physics.geometry import build_geometry, compute_stable_dt

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0.0"


def get_git_commit() -> str:
    """Retrieve current short git commit hash, or 'uncommitted' if unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit
    except Exception:
        return "uncommitted"


def get_package_versions() -> Dict[str, str]:
    """Return dictionary of installed core package versions."""
    return {
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "thin_film_pino": "0.1.0",
    }


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute sha256 hash of configuration dictionary."""
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def simulate_batch_trajectory(
    parameters: np.ndarray,
    args: argparse.Namespace,
    snapshots_per_phase: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Simulate a vectorized batch of trajectories and return multi-field states and cycle metrics.

    Parameters
    ----------
    parameters : np.ndarray
        Parameter array of shape (B, 8).
    args : argparse.Namespace
        CLI and simulation execution arguments.
    snapshots_per_phase : int
        Number of snapshots to record per phase.

    Returns
    -------
    states : dict
        Multi-field trajectory arrays shaped (B, T, ny, nx) or (B, T).
    cycle_metrics : dict
        Per-cycle metrics arrays shaped (B, num_cycles).
    """
    sim = BatchedTransientALDCudaSimulator(
        parameters,
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        trench_left=args.trench_left,
        trench_right=args.trench_right,
        trench_top=args.trench_top,
        trench_bottom=args.trench_bottom,
        dt=getattr(args, "dt", None),
        dtype=args.dtype,
        device=args.device,
    )
    batch_size = sim.batch_size

    states: Dict[str, List[np.ndarray]] = {
        "concentration": [],
        "surface_coverage": [],
        "film_thickness": [],
        "time": [],
        "cycle": [],
        "phase": [],
    }

    cycle_metric_records: Dict[str, List[np.ndarray]] = {
        "gpc": [],
        "mean_thickness": [],
        "field_thickness": [],
        "floor_thickness": [],
        "sidewall_thickness": [],
        "step_coverage": [],
        "dose_coverage": [],
        "surface_coverage": [],
    }

    current_time = torch.zeros(batch_size, dtype=sim.dtype, device=sim.device)

    # Initial state (t=0, cycle=0, phase=0)
    c0, th0, h0 = sim.snapshot()
    states["concentration"].append(c0)
    states["surface_coverage"].append(th0)
    states["film_thickness"].append(h0)
    states["time"].append(current_time.detach().cpu().numpy().copy())
    states["cycle"].append(np.zeros(batch_size, dtype=np.int16))
    states["phase"].append(np.zeros(batch_size, dtype=np.int8))

    pulse_steps = sim.pulse_steps()
    purge_steps = sim.fixed_steps(sim.purge_time)
    reaction_steps = sim.fixed_steps(sim.reaction_time)
    post_purge_steps = sim.fixed_steps(sim.post_purge_time)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=sim.device)

    prev_thickness = np.zeros(batch_size, dtype=np.float32)

    for cycle in range(1, args.num_cycles + 1):
        # 1. Precursor dose phase (phase 1)
        max_pulse = int(pulse_steps.max().item())
        pulse_capture = set(
            np.unique(np.round(np.linspace(1, max_pulse, snapshots_per_phase)).astype(int))
        )
        for s in range(1, max_pulse + 1):
            sim.transport_step(pulse_steps >= s, pulse=True)
            if s in pulse_capture:
                c, th, h = sim.snapshot()
                sub_time = current_time + torch.clamp(pulse_steps, max=s) * sim.dt
                states["concentration"].append(c)
                states["surface_coverage"].append(th)
                states["film_thickness"].append(h)
                states["time"].append(sub_time.detach().cpu().numpy().copy())
                states["cycle"].append(np.full(batch_size, cycle, dtype=np.int16))
                states["phase"].append(np.full(batch_size, 1, dtype=np.int8))

        current_time += pulse_steps * sim.dt
        dose_cov = sim.record_dose_coverage()

        # 2. Inert purge phase (phase 2)
        purge_capture = set(
            np.unique(np.round(np.linspace(1, purge_steps, snapshots_per_phase)).astype(int))
        )
        for s in range(1, purge_steps + 1):
            sim.transport_step(active_mask, pulse=False)
            if s == purge_steps:
                sim.C.zero_()
            if s in purge_capture:
                c, th, h = sim.snapshot()
                sub_time = current_time + s * sim.dt
                states["concentration"].append(c)
                states["surface_coverage"].append(th)
                states["film_thickness"].append(h)
                states["time"].append(sub_time.detach().cpu().numpy().copy())
                states["cycle"].append(np.full(batch_size, cycle, dtype=np.int16))
                states["phase"].append(np.full(batch_size, 2, dtype=np.int8))

        sim.C.zero_()
        current_time += purge_steps * sim.dt

        # 3. Co-reactant reaction and growth phase (phase 3)
        rxn_capture = set(
            np.unique(np.round(np.linspace(1, reaction_steps, snapshots_per_phase)).astype(int))
        )
        for s in range(1, reaction_steps + 1):
            sim.reaction_step()
            if s in rxn_capture:
                c, th, h = sim.snapshot()
                sub_time = current_time + s * sim.dt
                states["concentration"].append(c)
                states["surface_coverage"].append(th)
                states["film_thickness"].append(h)
                states["time"].append(sub_time.detach().cpu().numpy().copy())
                states["cycle"].append(np.full(batch_size, cycle, dtype=np.int16))
                states["phase"].append(np.full(batch_size, 3, dtype=np.int8))

        current_time += reaction_steps * sim.dt

        # 4. Post-reaction purge phase (phase 4)
        post_capture = set(
            np.unique(np.round(np.linspace(1, post_purge_steps, snapshots_per_phase)).astype(int))
        )
        for s in range(1, post_purge_steps + 1):
            sim.transport_step(active_mask, pulse=False)
            if s == post_purge_steps:
                sim.C.zero_()
            if s in post_capture:
                c, th, h = sim.snapshot()
                sub_time = current_time + s * sim.dt
                states["concentration"].append(c)
                states["surface_coverage"].append(th)
                states["film_thickness"].append(h)
                states["time"].append(sub_time.detach().cpu().numpy().copy())
                states["cycle"].append(np.full(batch_size, cycle, dtype=np.int16))
                states["phase"].append(np.full(batch_size, 4, dtype=np.int8))

        sim.C.zero_()
        current_time += post_purge_steps * sim.dt

        # Record cycle metrics for this cycle across batch
        surf_3d = sim.surface_mask.expand_as(sim.h)
        field_3d = sim.field_mask.expand_as(sim.h)
        floor_3d = sim.floor_mask.expand_as(sim.h)
        sidewall_3d = sim.sidewall_mask.expand_as(sim.h)

        h_surf = sim.h.masked_select(surf_3d).reshape(batch_size, -1)
        h_field = sim.h.masked_select(field_3d).reshape(batch_size, -1)
        h_floor = sim.h.masked_select(floor_3d).reshape(batch_size, -1)
        h_sidewall = sim.h.masked_select(sidewall_3d).reshape(batch_size, -1)
        th_surf = sim.theta.masked_select(surf_3d).reshape(batch_size, -1)

        m_h = h_surf.mean(dim=1).detach().cpu().numpy()
        m_field = h_field.mean(dim=1).detach().cpu().numpy()
        m_floor = h_floor.mean(dim=1).detach().cpu().numpy()
        m_sidewall = h_sidewall.mean(dim=1).detach().cpu().numpy()
        m_cov = th_surf.mean(dim=1).detach().cpu().numpy()

        sc = np.where(m_field > 0.0, m_floor / m_field, 0.0)
        gpc = m_h - prev_thickness
        prev_thickness = m_h

        cycle_metric_records["gpc"].append(gpc)
        cycle_metric_records["mean_thickness"].append(m_h)
        cycle_metric_records["field_thickness"].append(m_field)
        cycle_metric_records["floor_thickness"].append(m_floor)
        cycle_metric_records["sidewall_thickness"].append(m_sidewall)
        cycle_metric_records["step_coverage"].append(sc)
        cycle_metric_records["dose_coverage"].append(dose_cov)
        cycle_metric_records["surface_coverage"].append(m_cov)

    stacked_states = {
        # Shape: (B, T, ny, nx) for fields, (B, T) for scalars
        key: np.stack(val, axis=1)
        for key, val in states.items()
    }
    stacked_metrics = {
        # Shape: (B, num_cycles)
        key: np.stack(val, axis=1)
        for key, val in cycle_metric_records.items()
    }

    return stacked_states, stacked_metrics


def simulate_batch_final_state(
    parameters: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Simulate batch to final state after N cycles, capturing non-zero dose concentration."""
    sim = BatchedTransientALDCudaSimulator(
        parameters,
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        trench_left=args.trench_left,
        trench_right=args.trench_right,
        trench_top=args.trench_top,
        trench_bottom=args.trench_bottom,
        dt=getattr(args, "dt", None),
        dtype=args.dtype,
        device=args.device,
    )
    batch_size = sim.batch_size

    cycle_metric_records: Dict[str, List[np.ndarray]] = {
        "gpc": [],
        "mean_thickness": [],
        "field_thickness": [],
        "floor_thickness": [],
        "sidewall_thickness": [],
        "step_coverage": [],
        "dose_coverage": [],
        "surface_coverage": [],
    }

    current_time = torch.zeros(batch_size, dtype=sim.dtype, device=sim.device)
    pulse_steps = sim.pulse_steps()
    purge_steps = sim.fixed_steps(sim.purge_time)
    reaction_steps = sim.fixed_steps(sim.reaction_time)
    post_purge_steps = sim.fixed_steps(sim.post_purge_time)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=sim.device)

    prev_thickness = np.zeros(batch_size, dtype=np.float32)
    final_dose_c: Optional[np.ndarray] = None

    for cycle in range(1, args.num_cycles + 1):
        max_pulse = int(pulse_steps.max().item())
        for s in range(1, max_pulse + 1):
            sim.transport_step(pulse_steps >= s, pulse=True)

        current_time += pulse_steps * sim.dt
        dose_cov = sim.record_dose_coverage()

        # Capture non-zero concentration profile at conclusion of final dose
        if cycle == args.num_cycles:
            final_dose_c = sim.C.detach().cpu().numpy().copy()

        for _ in range(purge_steps):
            sim.transport_step(active_mask, pulse=False)
        sim.C.zero_()
        current_time += purge_steps * sim.dt

        for _ in range(reaction_steps):
            sim.reaction_step()
        current_time += reaction_steps * sim.dt

        for _ in range(post_purge_steps):
            sim.transport_step(active_mask, pulse=False)
        sim.C.zero_()
        current_time += post_purge_steps * sim.dt

        # Cycle metrics
        surf_3d = sim.surface_mask.expand_as(sim.h)
        field_3d = sim.field_mask.expand_as(sim.h)
        floor_3d = sim.floor_mask.expand_as(sim.h)
        sidewall_3d = sim.sidewall_mask.expand_as(sim.h)

        h_surf = sim.h.masked_select(surf_3d).reshape(batch_size, -1)
        h_field = sim.h.masked_select(field_3d).reshape(batch_size, -1)
        h_floor = sim.h.masked_select(floor_3d).reshape(batch_size, -1)
        h_sidewall = sim.h.masked_select(sidewall_3d).reshape(batch_size, -1)
        th_surf = sim.theta.masked_select(surf_3d).reshape(batch_size, -1)

        m_h = h_surf.mean(dim=1).detach().cpu().numpy()
        m_field = h_field.mean(dim=1).detach().cpu().numpy()
        m_floor = h_floor.mean(dim=1).detach().cpu().numpy()
        m_sidewall = h_sidewall.mean(dim=1).detach().cpu().numpy()
        m_cov = th_surf.mean(dim=1).detach().cpu().numpy()

        sc = np.where(m_field > 0.0, m_floor / m_field, 0.0)
        gpc = m_h - prev_thickness
        prev_thickness = m_h

        cycle_metric_records["gpc"].append(gpc)
        cycle_metric_records["mean_thickness"].append(m_h)
        cycle_metric_records["field_thickness"].append(m_field)
        cycle_metric_records["floor_thickness"].append(m_floor)
        cycle_metric_records["sidewall_thickness"].append(m_sidewall)
        cycle_metric_records["step_coverage"].append(sc)
        cycle_metric_records["dose_coverage"].append(dose_cov)
        cycle_metric_records["surface_coverage"].append(m_cov)

    _, th_final, h_final = sim.snapshot()

    states = {
        # Shape: (B, ny, nx) for final 2D fields, (B,) for scalars
        "concentration": final_dose_c if final_dose_c is not None else np.zeros_like(th_final),
        "surface_coverage": th_final,
        "film_thickness": h_final,
        "time": current_time.detach().cpu().numpy(),
        "cycle": np.full(batch_size, args.num_cycles, dtype=np.int16),
        "phase": np.full(batch_size, 4, dtype=np.int8),
    }

    stacked_metrics = {key: np.stack(val, axis=1) for key, val in cycle_metric_records.items()}

    return states, stacked_metrics


def assemble_chunks(
    chunk_files: List[Path],
    output_path: Path,
    common_arrays: Dict[str, np.ndarray],
) -> None:
    """Concatenate chunk archives array by array into final npz archive with low memory footprint."""
    if not chunk_files:
        raise ValueError("No chunk files provided for assembly.")

    # Inspect keys from first chunk
    first_chunk = np.load(chunk_files[0])
    chunk_keys = list(first_chunk.files)
    first_chunk.close()

    temp_zip = output_path.with_suffix(".tmp.npz")
    with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write common grid and geometry arrays
        for name, arr in common_arrays.items():
            with zf.open(f"{name}.npy", mode="w") as entry:
                np.lib.format.write_array(entry, np.asanyarray(arr), allow_pickle=False)

        # Concatenate chunked data arrays one key at a time
        for key in chunk_keys:
            pieces = []
            for cf in chunk_files:
                with np.load(cf) as cdata:
                    pieces.append(cdata[key])
            concatenated = np.concatenate(pieces, axis=0)
            with zf.open(f"{key}.npy", mode="w") as entry:
                np.lib.format.write_array(entry, np.asanyarray(concatenated), allow_pickle=False)
            del pieces, concatenated

    # Atomic rename
    temp_zip.replace(output_path)


def generate(
    mode: str = "trajectory",
    num_samples: int = 64,
    batch_size: int = 16,
    num_cycles: int = 10,
    snapshots_per_phase: int = 8,
    nx: int = 80,
    ny: int = 60,
    seed: int = 42,
    output_path: str | Path = "data/transient_ald_dataset.npz",
    priors_path: Optional[str | Path] = None,
    dtype: str = "float32",
    device: str = "cpu",
) -> Tuple[Path, Path]:
    """Execute unified dataset generation pipeline and serialize archive and metadata JSON.

    Parameters
    ----------
    mode : {'trajectory', 'final_state'}, default='trajectory'
        Generation mode. 'trajectory' stores full time evolution; 'final_state' stores states after N cycles.
    num_samples : int, default=64
        Total ensemble size.
    batch_size : int, default=16
        Vectorized simulation batch size per chunk.
    num_cycles : int, default=10
        Number of ALD cycles per sample.
    snapshots_per_phase : int, default=8
        Snapshots per stage in trajectory mode (must be >= 1, defaults to >= 8).
    nx : int, default=80
        Horizontal grid resolution.
    ny : int, default=60
        Vertical grid resolution.
    seed : int, default=42
        Master random seed.
    output_path : str or Path, default='data/transient_ald_dataset.npz'
        Destination output archive.
    priors_path : str or Path, optional
        Path to priors YAML configuration.
    dtype : str, default='float32'
        Floating-point calculation precision.
    device : str, default='cpu'
        Target execution device ('cpu' or 'cuda').

    Returns
    -------
    out_archive : Path
        Path to generated .npz archive.
    out_metadata : Path
        Path to generated .json metadata descriptor.
    """
    if mode not in ("trajectory", "final_state"):
        raise ValueError(f"Unsupported mode: {mode}. Must be 'trajectory' or 'final_state'.")
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive integers.")
    if snapshots_per_phase < 1:
        raise ValueError("snapshots_per_phase must be >= 1.")

    priors = load_priors(priors_path)
    grid_cfg = priors.get("grid", {})
    lx = float(grid_cfg.get("lx", 1.0))
    ly = float(grid_cfg.get("ly", 1.0))
    trench_left = float(grid_cfg.get("trench_left", 0.375))
    trench_right = float(grid_cfg.get("trench_right", 0.625))
    trench_top = float(grid_cfg.get("trench_top", 0.50))
    trench_bottom = float(grid_cfg.get("trench_bottom", 0.95))

    geom = build_geometry(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        trench_left=trench_left,
        trench_right=trench_right,
        trench_top=trench_top,
        trench_bottom=trench_bottom,
    )

    # Time step determination: compute once globally from prior bounds
    global_dt = compute_stable_dt(
        geom.dx,
        geom.dy,
        float(priors.get("parameters", {}).get("diffusivity", {}).get("high", 8.0e-3)),
        float(priors.get("parameters", {}).get("k_ads", {}).get("high", 1.5)),
        float(priors.get("parameters", {}).get("k_des", {}).get("high", 0.10)),
        float(priors.get("parameters", {}).get("k_rxn", {}).get("high", 0.80)),
    )

    args = argparse.Namespace(
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        trench_left=trench_left,
        trench_right=trench_right,
        trench_top=trench_top,
        trench_bottom=trench_bottom,
        num_cycles=num_cycles,
        dt=global_dt,
        dtype=dtype,
        device=device,
    )

    # Draw reproducible parameter ensemble with SeedSequence
    parameters, rejections = sample_parameter_ensemble(
        num_samples=num_samples, seed=seed, priors=priors
    )
    if rejections:
        logger.warning("Parameter sampling rejections: %s", rejections)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    chunk_files: List[Path] = []
    num_chunks = int(np.ceil(num_samples / batch_size))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for chunk_idx in range(num_chunks):
            start = chunk_idx * batch_size
            end = min(start + batch_size, num_samples)
            chunk_params = parameters[start:end]

            if mode == "trajectory":
                states, metrics = simulate_batch_trajectory(chunk_params, args, snapshots_per_phase)
            else:
                states, metrics = simulate_batch_final_state(chunk_params, args)

            chunk_arrays = {
                "parameters": chunk_params.astype(np.float32),
                **states,
                **{f"metric_{k}": v.astype(np.float32) for k, v in metrics.items()},
            }

            chunk_file = tmp_path / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(chunk_file, **chunk_arrays)
            chunk_files.append(chunk_file)

        # Assemble chunks into final archive
        common_arrays = {
            "solid_mask": geom.solid_mask,
            "gas_mask": geom.gas_mask,
            "surface_mask": geom.surface_mask,
            "field_mask": geom.field_mask,
            "sidewall_mask": geom.sidewall_mask,
            "floor_mask": geom.floor_mask,
            "x": geom.x.astype(np.float32),
            "y": geom.y.astype(np.float32),
            "parameter_names": np.asarray(PARAMETER_NAMES),
            "phase_names": np.asarray(PHASE_NAMES),
        }
        assemble_chunks(chunk_files, out_file, common_arrays)

    # Compute relative path for portable metadata
    try:
        rel_archive_path = str(out_file.relative_to(Path.cwd()))
    except ValueError:
        rel_archive_path = out_file.name

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(),
        "config_hash": compute_config_hash(priors),
        "num_samples": num_samples,
        "num_cycles": num_cycles,
        "snapshots_per_phase": snapshots_per_phase if mode == "trajectory" else 1,
        "grid": {
            "nx": nx,
            "ny": ny,
            "lx": lx,
            "ly": ly,
            "trench_left": trench_left,
            "trench_right": trench_right,
            "trench_top": trench_top,
            "trench_bottom": trench_bottom,
            "aspect_ratio": (trench_bottom - trench_top) / (trench_right - trench_left),
        },
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "sampling": {
            "rejections": rejections,
            "seed_sequence_spawn": True,
        },
        "concentration_snapshot_policy": (
            "end_of_dose" if mode == "final_state" else "phase_end_and_intermediate_snapshots"
        ),
        "units": (
            "Nondimensional spatial coordinates (x/Lx, y/Ly), normalized concentration "
            "(C/C0), fractional coverage theta in [0, 1], film thickness h in length units. "
            "See docs/formulation.md."
        ),
        "archive_path": rel_archive_path,
    }

    meta_file = out_file.with_suffix(".json")
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return out_file, meta_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified ALD dataset generator for trajectory and final-state ensembles."
    )
    parser.add_argument(
        "--mode",
        choices=["trajectory", "final_state"],
        default="trajectory",
        help="Dataset generation mode: full temporal trajectory or final state after N cycles.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Total number of trajectory samples."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Vectorized simulation chunk size."
    )
    parser.add_argument(
        "--num-cycles", type=int, default=10, help="Number of ALD cycles per sample."
    )
    parser.add_argument(
        "--snapshots-per-phase",
        type=int,
        default=8,
        help="Number of snapshots recorded per phase (trajectory mode).",
    )
    parser.add_argument("--nx", type=int, default=80, help="Horizontal grid resolution.")
    parser.add_argument("--ny", type=int, default=60, help="Vertical grid resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument(
        "--output",
        default="data/transient_ald_dataset.npz",
        help="Destination path for .npz archive.",
    )
    parser.add_argument(
        "--priors",
        default="configs/priors.yaml",
        help="Path to priors YAML configuration.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Precision for floating-point calculations.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_archive, out_meta = generate(
        mode=args.mode,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_cycles=args.num_cycles,
        snapshots_per_phase=args.snapshots_per_phase,
        nx=args.nx,
        ny=args.ny,
        seed=args.seed,
        output_path=args.output,
        priors_path=args.priors,
        dtype=args.dtype,
        device=args.device,
    )
    print(f"Generated {args.mode} dataset ({args.num_samples} samples): {out_archive}")
    print(f"Saved dataset metadata: {out_meta}")


if __name__ == "__main__":
    main()

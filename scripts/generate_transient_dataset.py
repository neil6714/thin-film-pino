"""Generate time-resolved trajectories from the continuum ALD simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from thin_film_pino.data.parameters import (
    PARAMETER_NAMES,
    PHASE_NAMES,
    sample_parameters,
)
from thin_film_pino.physics.transient_cuda import TransientALDCudaSimulator

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-resolved trajectories from the continuum ALD simulator."
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--snapshots-per-phase", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="data/transient_ald_dataset.npz")
    return parser.parse_args()


def build_simulator(parameters: np.ndarray, num_cycles: int) -> TransientALDCudaSimulator:
    return TransientALDCudaSimulator(
        diffusivity=float(parameters[0]),
        k_ads=float(parameters[1]),
        k_des=float(parameters[2]),
        k_rxn=float(parameters[3]),
        k_growth=float(parameters[4]),
        pulse_time=float(parameters[5]),
        purge_time=float(parameters[6]),
        reaction_time=float(parameters[7]),
        num_cycles=num_cycles,
    )


def snapshot_steps(num_steps: int, snapshots_per_phase: int) -> Set[int]:
    return set(np.unique(np.ceil(np.linspace(1, num_steps, snapshots_per_phase)).astype(int)))


def append_snapshot(
    trajectory: Dict[str, list],
    simulator: TransientALDCudaSimulator,
    time: float,
    cycle: int,
    phase: int,
) -> None:
    c, theta, h = simulator.snapshot()
    trajectory["time"].append(time)
    trajectory["cycle"].append(cycle)
    trajectory["phase"].append(phase)
    trajectory["concentration"].append(c)
    trajectory["surface_coverage"].append(theta)
    trajectory["film_thickness"].append(h)


def advance_pulse_phase(
    trajectory: Dict[str, list],
    simulator: TransientALDCudaSimulator,
    duration: float,
    time: float,
    cycle: int,
    snapshots_per_phase: int,
) -> float:
    steps = max(1, math.ceil(duration / simulator.dt))
    sub_dt = duration / steps
    capture_steps = snapshot_steps(steps, snapshots_per_phase)
    for step in range(1, steps + 1):
        simulator.advance_pulse_step(sub_dt)
        time += sub_dt
        if step in capture_steps:
            append_snapshot(trajectory, simulator, time, cycle, 1)
    return time


def advance_purge_phase(
    trajectory: Dict[str, list],
    simulator: TransientALDCudaSimulator,
    duration: float,
    time: float,
    cycle: int,
    phase_id: int,
    snapshots_per_phase: int,
) -> float:
    steps = max(1, math.ceil(duration / simulator.dt))
    sub_dt = duration / steps
    capture_steps = snapshot_steps(steps, snapshots_per_phase)
    simulator.C[0, :] = 0.0
    for step in range(1, steps + 1):
        simulator.advance_purge_step(sub_dt)
        time += sub_dt
        if step == steps:
            simulator.C.zero_()
        if step in capture_steps:
            append_snapshot(trajectory, simulator, time, cycle, phase_id)
    return time


def advance_reaction_phase(
    trajectory: Dict[str, list],
    simulator: TransientALDCudaSimulator,
    duration: float,
    time: float,
    cycle: int,
    snapshots_per_phase: int,
) -> float:
    steps = max(1, math.ceil(duration / simulator.dt))
    sub_dt = duration / steps
    capture_steps = snapshot_steps(steps, snapshots_per_phase)
    for step in range(1, steps + 1):
        simulator.advance_reaction_step(sub_dt)
        time += sub_dt
        if step in capture_steps:
            append_snapshot(trajectory, simulator, time, cycle, 3)
    return time


def simulate_trajectory(
    parameters: np.ndarray, num_cycles: int, snapshots_per_phase: int
) -> Tuple[TransientALDCudaSimulator, Dict[str, list]]:
    simulator = build_simulator(parameters, num_cycles)
    trajectory: Dict[str, list] = {
        "time": [],
        "cycle": [],
        "phase": [],
        "concentration": [],
        "surface_coverage": [],
        "film_thickness": [],
    }
    append_snapshot(trajectory, simulator, time=0.0, cycle=0, phase=0)

    time = 0.0
    previous_thickness = 0.0

    for cycle in range(1, num_cycles + 1):
        time = advance_pulse_phase(
            trajectory, simulator, simulator.pulse_time, time, cycle, snapshots_per_phase
        )
        time = advance_purge_phase(
            trajectory, simulator, simulator.purge_time, time, cycle, 2, snapshots_per_phase
        )
        time = advance_reaction_phase(
            trajectory, simulator, simulator.reaction_time, time, cycle, snapshots_per_phase
        )
        time = advance_purge_phase(
            trajectory, simulator, simulator.post_purge_time, time, cycle, 4, snapshots_per_phase
        )
        previous_thickness = simulator.record_cycle_metrics(cycle, previous_thickness)

    return simulator, trajectory


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.num_cycles < 1 or args.snapshots_per_phase < 1:
        raise ValueError("Sample, cycle, and snapshot counts must be positive.")

    rng = np.random.default_rng(args.seed)
    parameters: List[np.ndarray] = []
    trajectories: List[Dict[str, list]] = []
    simulator: TransientALDCudaSimulator | None = None

    for i in range(args.num_samples):
        p = sample_parameters(rng)
        simulator, traj = simulate_trajectory(p, args.num_cycles, args.snapshots_per_phase)
        parameters.append(p)
        trajectories.append(traj)
        print(f"Sample {i + 1:03d}/{args.num_samples:03d} simulated.")

    if simulator is None:
        raise RuntimeError("No trajectories generated.")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        parameters=np.asarray(parameters, dtype=np.float32),
        parameter_names=np.asarray(PARAMETER_NAMES),
        phase_names=np.asarray(PHASE_NAMES),
        time=np.asarray([t["time"] for t in trajectories], dtype=np.float32),
        cycle=np.asarray([t["cycle"] for t in trajectories], dtype=np.int16),
        phase=np.asarray([t["phase"] for t in trajectories], dtype=np.int8),
        concentration=np.asarray([t["concentration"] for t in trajectories], dtype=np.float32),
        surface_coverage=np.asarray(
            [t["surface_coverage"] for t in trajectories], dtype=np.float32
        ),
        film_thickness=np.asarray([t["film_thickness"] for t in trajectories], dtype=np.float32),
        x=simulator.x.detach().cpu().numpy(),
        y=simulator.y.detach().cpu().numpy(),
        solid_mask=simulator.solid_mask.detach().cpu().numpy(),
        surface_mask=simulator.surface_mask.detach().cpu().numpy(),
        seed=args.seed,
        num_cycles=args.num_cycles,
        snapshots_per_phase=args.snapshots_per_phase,
    )

    metadata: Dict[str, Any] = {
        "num_trajectories": len(trajectories),
        "num_cycles": args.num_cycles,
        "snapshots_per_phase": args.snapshots_per_phase,
        "snapshots_per_trajectory": len(trajectories[0]["time"]),
        "parameter_names": list(PARAMETER_NAMES),
        "phase_names": list(PHASE_NAMES),
        "seed": args.seed,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTransient dataset generation complete.")
    print(f"Saved data: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

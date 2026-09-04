import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics.transient_cuda import TransientALDCudaSimulator


SEED = 42
PARAMETER_NAMES = (
    "diffusivity",
    "k_ads",
    "k_des",
    "k_rxn",
    "k_growth",
    "pulse_time",
    "purge_time",
    "reaction_time",
)
PARAMETER_RANGES = {
    "diffusivity": (1.0e-3, 8.0e-3),
    "k_ads": (0.5, 1.5),
    "k_des": (0.02, 0.10),
    "k_rxn": (0.1, 0.8),
    "k_growth": (0.02, 0.02),
    "pulse_time": (10.0, 100.0),
    "purge_time": (1.0, 1.0),
    "reaction_time": (1.0, 1.0),
}
PHASE_NAMES = ("initial", "pulse", "purge", "reaction", "post_reaction_purge")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate time-resolved trajectories from the validated ALD simulator."
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--snapshots-per-phase", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        default="data/transient_ald_dataset.npz",
    )
    return parser.parse_args()


def sample_pulse_time(rng):
    if rng.random() < 0.85:
        return rng.uniform(10.0, 60.0)
    return rng.uniform(60.0, 100.0)


def sample_parameters(rng):
    return np.array(
        [
            rng.uniform(*PARAMETER_RANGES["diffusivity"]),
            rng.uniform(*PARAMETER_RANGES["k_ads"]),
            rng.uniform(*PARAMETER_RANGES["k_des"]),
            rng.uniform(*PARAMETER_RANGES["k_rxn"]),
            PARAMETER_RANGES["k_growth"][0],
            sample_pulse_time(rng),
            PARAMETER_RANGES["purge_time"][0],
            PARAMETER_RANGES["reaction_time"][0],
        ],
        dtype=np.float64,
    )


def build_simulator(parameters, num_cycles):
    return TransientALDCudaSimulator(
        diffusivity=parameters[0],
        k_ads=parameters[1],
        k_des=parameters[2],
        k_rxn=parameters[3],
        k_growth=parameters[4],
        pulse_time=parameters[5],
        purge_time=parameters[6],
        reaction_time=parameters[7],
        num_cycles=num_cycles,
    )


def snapshot_steps(num_steps, snapshots_per_phase):
    return set(
        np.unique(
            np.ceil(
                np.linspace(1, num_steps, snapshots_per_phase)
            ).astype(int)
        )
    )


def append_snapshot(trajectory, simulator, time, cycle, phase):
    concentration, surface_coverage, film_thickness = simulator.snapshot()
    trajectory["time"].append(time)
    trajectory["cycle"].append(cycle)
    trajectory["phase"].append(phase)
    trajectory["concentration"].append(concentration)
    trajectory["surface_coverage"].append(surface_coverage)
    trajectory["film_thickness"].append(film_thickness)


def advance_transport_phase(
    trajectory,
    simulator,
    duration,
    pulse,
    time,
    cycle,
    phase,
    snapshots_per_phase,
):
    if pulse:
        simulator.C[0, :] = simulator.inlet_concentration
    else:
        simulator.C[0, :] = 0.0

    num_steps = simulator._num_steps(duration)
    capture_steps = snapshot_steps(num_steps, snapshots_per_phase)

    for step in range(1, num_steps + 1):
        simulator.transport_step(pulse=pulse)
        time += simulator.dt

        if not pulse and step == num_steps:
            simulator.C.zero_()

        if step in capture_steps:
            append_snapshot(trajectory, simulator, time, cycle, phase)

    return time


def advance_reaction_phase(
    trajectory,
    simulator,
    time,
    cycle,
    snapshots_per_phase,
):
    num_steps = simulator._num_steps(simulator.reaction_time)
    capture_steps = snapshot_steps(num_steps, snapshots_per_phase)

    for step in range(1, num_steps + 1):
        simulator.reaction_step()
        time += simulator.dt

        if step in capture_steps:
            append_snapshot(trajectory, simulator, time, cycle, 3)

    return time


def record_cycle_metrics(simulator, cycle, previous_thickness):
    return simulator.record_cycle_metrics(cycle, previous_thickness)


def simulate_trajectory(parameters, num_cycles, snapshots_per_phase):
    simulator = build_simulator(parameters, num_cycles)
    simulator.C.zero_()
    simulator.theta.zero_()
    simulator.h.zero_()
    simulator.history = simulator._empty_history()

    trajectory = {
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
        time = advance_transport_phase(
            trajectory,
            simulator,
            simulator.pulse_time,
            pulse=True,
            time=time,
            cycle=cycle,
            phase=1,
            snapshots_per_phase=snapshots_per_phase,
        )
        time = advance_transport_phase(
            trajectory,
            simulator,
            simulator.purge_time,
            pulse=False,
            time=time,
            cycle=cycle,
            phase=2,
            snapshots_per_phase=snapshots_per_phase,
        )
        time = advance_reaction_phase(
            trajectory,
            simulator,
            time,
            cycle,
            snapshots_per_phase,
        )
        time = advance_transport_phase(
            trajectory,
            simulator,
            simulator.purge_time,
            pulse=False,
            time=time,
            cycle=cycle,
            phase=4,
            snapshots_per_phase=snapshots_per_phase,
        )
        previous_thickness = record_cycle_metrics(
            simulator,
            cycle,
            previous_thickness,
        )

    return simulator, trajectory


def validate_trajectory(simulator, trajectory):
    fields = (
        trajectory["concentration"],
        trajectory["surface_coverage"],
        trajectory["film_thickness"],
    )
    return (
        all(np.all(np.isfinite(field)) for field in fields)
        and np.all(np.asarray(trajectory["concentration"]) >= 0.0)
        and np.all(np.asarray(trajectory["surface_coverage"]) >= 0.0)
        and np.all(np.asarray(trajectory["surface_coverage"]) <= 1.0)
        and np.all(np.asarray(trajectory["film_thickness"]) >= 0.0)
        and np.all(np.isfinite(simulator.history["gpc"]))
        and np.all(np.asarray(simulator.history["gpc"]) >= 0.0)
    )


def stack_trajectories(trajectories, key, dtype):
    return np.asarray([trajectory[key] for trajectory in trajectories], dtype=dtype)


def main():
    args = parse_args()
    if args.num_samples < 1 or args.num_cycles < 1 or args.snapshots_per_phase < 1:
        raise ValueError("Sample, cycle, and snapshot counts must be positive.")

    rng = np.random.default_rng(args.seed)
    parameters = []
    trajectories = []
    cycle_metrics = []

    print("Generating transient ALD dataset...")
    print(f"Target trajectories: {args.num_samples}")
    print(f"ALD cycles per trajectory: {args.num_cycles}")
    print(f"Snapshots per phase: {args.snapshots_per_phase}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    while len(trajectories) < args.num_samples:
        sample = sample_parameters(rng)
        simulator, trajectory = simulate_trajectory(
            sample,
            args.num_cycles,
            args.snapshots_per_phase,
        )

        if not validate_trajectory(simulator, trajectory):
            print(f"Rejected trajectory {len(trajectories) + 1}")
            continue

        parameters.append(sample)
        trajectories.append(trajectory)
        cycle_metrics.append(
            np.column_stack(
                (
                    simulator.history["gpc"],
                    simulator.history["mean_thickness"],
                    simulator.history["top_thickness"],
                    simulator.history["bottom_thickness"],
                    simulator.history["conformality"],
                    simulator.history["surface_coverage"],
                )
            )
        )
        print(
            f"Generated {len(trajectories):03d}/{args.num_samples} | "
            f"pulse={sample[5]:.2f} | "
            f"final conformality={simulator.history['conformality'][-1]:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        parameters=np.asarray(parameters, dtype=np.float64),
        parameter_names=np.asarray(PARAMETER_NAMES),
        time=stack_trajectories(trajectories, "time", np.float64),
        cycle=stack_trajectories(trajectories, "cycle", np.int16),
        phase=stack_trajectories(trajectories, "phase", np.int8),
        phase_names=np.asarray(PHASE_NAMES),
        concentration=stack_trajectories(trajectories, "concentration", np.float32),
        surface_coverage=stack_trajectories(trajectories, "surface_coverage", np.float32),
        film_thickness=stack_trajectories(trajectories, "film_thickness", np.float32),
        cycle_metrics=np.asarray(cycle_metrics, dtype=np.float32),
        cycle_metric_names=np.asarray(
            (
                "gpc",
                "mean_thickness",
                "top_thickness",
                "bottom_thickness",
                "conformality",
                "surface_coverage",
            )
        ),
        x=simulator.x.detach().cpu().numpy(),
        y=simulator.y.detach().cpu().numpy(),
        solid_mask=simulator.solid_mask.detach().cpu().numpy(),
        surface_mask=simulator.surface_mask.detach().cpu().numpy(),
        seed=args.seed,
        num_cycles=args.num_cycles,
        snapshots_per_phase=args.snapshots_per_phase,
    )

    metadata = {
        "schema_version": 1,
        "num_trajectories": len(trajectories),
        "num_cycles": args.num_cycles,
        "snapshots_per_phase": args.snapshots_per_phase,
        "snapshots_per_trajectory": len(trajectories[0]["time"]),
        "parameter_names": PARAMETER_NAMES,
        "phase_names": PHASE_NAMES,
        "cycle_metric_names": (
            "gpc",
            "mean_thickness",
            "top_thickness",
            "bottom_thickness",
            "conformality",
            "surface_coverage",
        ),
        "seed": args.seed,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTransient dataset generation complete.")
    print(f"Saved data: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

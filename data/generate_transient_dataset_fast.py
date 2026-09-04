import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.generate_transient_dataset import PARAMETER_NAMES, sample_parameters
from physics.batched_transient_cuda import BatchedTransientALDCudaSimulator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate batched CUDA ALD trajectories.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--output", default="data/transient_ald_dataset_fast.npz")
    return parser.parse_args()


def append_state(states, simulator, time, cycle, phase):
    c, theta, h = simulator.snapshot()
    states["concentration"].append(c)
    states["surface_coverage"].append(theta)
    states["film_thickness"].append(h)
    states["time"].append(time.cpu().numpy())
    states["cycle"].append(np.full(simulator.batch_size, cycle, dtype=np.int16))
    states["phase"].append(np.full(simulator.batch_size, phase, dtype=np.int8))


def run_batch(parameters, args):
    simulator = BatchedTransientALDCudaSimulator(parameters, nx=args.nx, ny=args.ny)
    states = {key: [] for key in ("concentration", "surface_coverage", "film_thickness", "time", "cycle", "phase")}
    time = torch.zeros(simulator.batch_size, device=simulator.device)
    append_state(states, simulator, time, 0, 0)
    pulse_steps = simulator.pulse_steps()
    purge_steps = simulator.fixed_steps(simulator.purge_time)
    reaction_steps = simulator.fixed_steps(simulator.reaction_time)

    for cycle in range(1, args.num_cycles + 1):
        for step in range(1, int(pulse_steps.max().item()) + 1):
            simulator.transport_step(pulse_steps >= step, pulse=True)
        time += pulse_steps * simulator.dt
        append_state(states, simulator, time, cycle, 1)
        for _ in range(purge_steps):
            simulator.transport_step(torch.ones(simulator.batch_size, dtype=torch.bool, device=simulator.device), pulse=False)
        simulator.C.zero_()
        time += purge_steps * simulator.dt
        append_state(states, simulator, time, cycle, 2)
        for _ in range(reaction_steps):
            simulator.reaction_step()
        time += reaction_steps * simulator.dt
        append_state(states, simulator, time, cycle, 3)
        for _ in range(purge_steps):
            simulator.transport_step(torch.ones(simulator.batch_size, dtype=torch.bool, device=simulator.device), pulse=False)
        simulator.C.zero_()
        time += purge_steps * simulator.dt
        append_state(states, simulator, time, cycle, 4)

    return simulator, {key: np.stack(value, axis=1) for key, value in states.items()}


def main():
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("Sample and batch sizes must be positive.")
    rng = np.random.default_rng(args.seed)
    parameters = np.stack([sample_parameters(rng) for _ in range(args.num_samples)])
    batches = []
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    for start in range(0, args.num_samples, args.batch_size):
        stop = min(start + args.batch_size, args.num_samples)
        simulator, batch = run_batch(parameters[start:stop], args)
        batches.append(batch)
        print(f"Generated {stop}/{args.num_samples} trajectories")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        parameters=parameters,
        parameter_names=np.asarray(PARAMETER_NAMES),
        concentration=np.concatenate([batch["concentration"] for batch in batches]),
        surface_coverage=np.concatenate([batch["surface_coverage"] for batch in batches]),
        film_thickness=np.concatenate([batch["film_thickness"] for batch in batches]),
        time=np.concatenate([batch["time"] for batch in batches]),
        cycle=np.concatenate([batch["cycle"] for batch in batches]),
        phase=np.concatenate([batch["phase"] for batch in batches]),
        x=simulator.x.detach().cpu().numpy(),
        y=simulator.y.detach().cpu().numpy(),
        solid_mask=simulator.solid_mask.detach().cpu().numpy(),
        surface_mask=simulator.surface_mask.detach().cpu().numpy(),
        seed=args.seed,
        num_cycles=args.num_cycles,
    )
    output.with_suffix(".json").write_text(json.dumps({"grid": [args.ny, args.nx], "snapshots_per_trajectory": 1 + 4 * args.num_cycles, "batch_size": args.batch_size}, indent=2))
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.train_temporal_operator import TemporalFourierOperator


def laplacian(field, dx, dy):
    padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
    return (padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2] - 2 * field) / dx**2 + (padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1] - 2 * field) / dy**2


def physics_loss(pred, parameters, time, phase, masks, stats, dx, dy):
    mean = stats[0][None, None, :, None, None]
    scale = stats[1][None, None, :, None, None]
    physical = pred * scale + mean
    c, theta, thickness = physical.unbind(dim=2)
    dt = (time[:, 2:] - time[:, :-2]).clamp_min(1e-8)[:, :, None, None]
    dc_dt = (c[:, 2:] - c[:, :-2]) / dt
    dtheta_dt = (theta[:, 2:] - theta[:, :-2]) / dt
    dh_dt = (thickness[:, 2:] - thickness[:, :-2]) / dt
    middle_phase = phase[:, 1:-1]
    pulse = (middle_phase == 1).float()[:, :, None, None]
    reaction = (middle_phase == 3).float()[:, :, None, None]
    diffusivity = parameters[:, 0, None, None, None]
    k_ads = parameters[:, 1, None, None, None]
    k_des = parameters[:, 2, None, None, None]
    k_rxn = parameters[:, 3, None, None, None]
    k_growth = parameters[:, 4, None, None, None]
    c_mid = c[:, 1:-1]
    theta_mid = theta[:, 1:-1]
    adsorption = pulse * k_ads * c_mid * (1.0 - theta_mid)
    desorption = pulse * k_des * theta_mid
    transport = dc_dt - diffusivity * laplacian(c_mid, dx, dy) + adsorption
    coverage = dtheta_dt - adsorption + desorption + reaction * k_rxn * theta_mid
    growth = dh_dt - reaction * k_growth * k_rxn * theta_mid
    return (
        transport.square() * masks[0]
        + coverage.square() * masks[1]
        + growth.square() * masks[2]
    ).mean()


def main():
    parser = argparse.ArgumentParser(description="Train a transient PDE-PINO on CUDA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physics-weight", type=float, default=1.0e-4)
    parser.add_argument("--output-dir", default="results/stage4d_temporal_pino")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4D requires CUDA.")
    data = np.load(args.input)
    device = torch.device("cuda")
    fields = np.stack([data["concentration"], data["surface_coverage"], data["film_thickness"]], axis=2)
    def loader(indices, shuffle):
        tensors = (torch.from_numpy(data["parameters"][indices]), torch.from_numpy(data["time"][indices]), torch.from_numpy(data["phase"][indices]), torch.from_numpy(fields[indices]))
        return DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=shuffle)
    train_loader = loader(data["train_indices"], True)
    val_loader = loader(data["validation_indices"], False)
    coords = torch.stack(torch.meshgrid(torch.as_tensor(data["y"], device=device), torch.as_tensor(data["x"], device=device), indexing="ij"), dim=0).float() * 2 - 1
    mask = torch.stack((torch.as_tensor(~data["solid_mask"], device=device), torch.as_tensor(data["surface_mask"], device=device), torch.as_tensor(data["surface_mask"], device=device))).float()
    stats = (torch.as_tensor(data["field_mean"], device=device), torch.as_tensor(data["field_std"], device=device))
    parameter_mean = torch.as_tensor(data["parameter_mean"], device=device)
    parameter_std = torch.as_tensor(data["parameter_std"], device=device)
    time_mean = torch.as_tensor(data["time_mean"], device=device)
    time_std = torch.as_tensor(data["time_std"], device=device)
    model = TemporalFourierOperator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    history = {"train_total": [], "train_data": [], "train_physics": [], "validation_data": []}
    best = float("inf")
    def evaluate(loader):
        model.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for parameters, time, phase, target in loader:
                parameters, time, phase, target = parameters.to(device), time.to(device), phase.to(device), target.to(device)
                pred = model(parameters, time, coords)
                total += (((pred - target).square() * mask[None, None]).sum()).item(); count += target.shape[0]
        return total / max(count * float(mask.sum()) * target.shape[1], 1.0)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = data_total = physics_total = 0.0
        for parameters, time, phase, target in train_loader:
            parameters, time, phase, target = parameters.to(device), time.to(device), phase.to(device), target.to(device)
            pred = model(parameters, time, coords)
            data_loss = ((pred - target).square() * mask[None, None]).mean()
            pde_loss = physics_loss(pred, parameters * parameter_std + parameter_mean, time * time_std + time_mean, phase, mask, stats, float(data["x"][1] - data["x"][0]), float(data["y"][1] - data["y"][0]))
            loss = data_loss + args.physics_weight * torch.nan_to_num(pde_loss, nan=0.0, posinf=1.0e6, neginf=1.0e6)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += loss.item()
            data_total += data_loss.item(); physics_total += pde_loss.item()
        val = evaluate(val_loader); history["train_total"].append(total / len(train_loader)); history["train_data"].append(data_total / len(train_loader)); history["train_physics"].append(physics_total / len(train_loader)); history["validation_data"].append(val)
        if val < best: best = val; torch.save({"model": model.state_dict(), "physics_weight": args.physics_weight}, output / "best_model.pt")
        print(f"Epoch {epoch:03d}/{args.epochs} | data={data_total/len(train_loader):.6e} | physics={physics_total/len(train_loader):.6e} | val={val:.6e}")
    np.savez(output / "training_history.npz", **history)
    (output / "metrics.json").write_text(json.dumps({"best_validation_data_loss": best, "physics_weight": args.physics_weight, "test_evaluation": "use shared comparison script"}, indent=2))
    print(f"Saved PINO checkpoint, losses, and validation metrics to: {output}")


if __name__ == "__main__":
    main()

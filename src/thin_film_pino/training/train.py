"""Training pipeline for temporal Fourier neural operators and physics-informed models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from thin_film_pino.models.fno import TemporalFourierOperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal Fourier neural operators with optional physics residuals."
    )
    parser.add_argument(
        "--model",
        choices=["fno", "pino"],
        default="fno",
        help="Model architecture: 'fno' (pure data-driven) or 'pino' (physics-informed).",
    )
    parser.add_argument(
        "--input", required=True, help="Path to prepared transient ALD dataset (.npz)."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for trajectories.")
    parser.add_argument(
        "--learning-rate", type=float, default=1.0e-3, help="Initial learning rate."
    )
    parser.add_argument(
        "--width", type=int, default=32, help="Hidden channel width in Fourier layers."
    )
    parser.add_argument("--modes", type=int, default=12, help="Fourier frequency cutoff modes.")
    parser.add_argument("--seed", type=int, default=123, help="Random number generator seed.")
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0e-4,
        help="Regularization weight for PDE residual loss term (used when model=pino).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save model checkpoints and training logs. Defaults to runs/<model>.",
    )
    return parser.parse_args()


def laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Compute 2D spatial discrete Laplacian using replicate-padded central differences."""
    padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode="replicate")
    d2_dx2 = (padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2] - 2.0 * field) / dx**2
    d2_dy2 = (padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1] - 2.0 * field) / dy**2
    return d2_dx2 + d2_dy2


def physics_loss(
    pred: torch.Tensor,
    parameters: torch.Tensor,
    time: torch.Tensor,
    phase: torch.Tensor,
    masks: torch.Tensor,
    stats: Tuple[torch.Tensor, torch.Tensor],
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute PDE residuals governing transient transport, surface coverage, and film growth."""
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
        transport.square() * masks[0] + coverage.square() * masks[1] + growth.square() * masks[2]
    ).mean()


def loss_and_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked mean squared error, mean absolute error, and valid cell count."""
    diff = prediction - target
    active = mask.expand_as(diff)
    valid = active.sum().clamp_min(1.0)
    mse = (diff.square() * active).sum() / valid
    mae = (diff.abs() * active).sum() / valid
    return mse, mae, valid


def make_loader(
    data: Mapping[str, Any],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    include_phase: bool = False,
) -> DataLoader:
    """Build PyTorch DataLoader from preprocessed trajectory array archive."""
    fields = np.stack(
        [
            data["concentration"][indices],
            data["surface_coverage"][indices],
            data["film_thickness"][indices],
        ],
        axis=2,
    )
    tensors = [
        torch.from_numpy(data["parameters"][indices]),
        torch.from_numpy(data["time"][indices]),
    ]
    if include_phase and "phase" in data:
        tensors.append(torch.from_numpy(data["phase"][indices]))
    tensors.append(torch.from_numpy(fields))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")

    output_dir = Path(args.output_dir or f"runs/{args.model}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")

    input_path = Path(args.input).resolve()
    data = np.load(input_path)

    is_pino = args.model == "pino"
    train_loader = make_loader(
        data, data["train_indices"], args.batch_size, shuffle=True, include_phase=is_pino
    )
    val_loader = make_loader(
        data, data["validation_indices"], args.batch_size, shuffle=False, include_phase=is_pino
    )
    test_loader = make_loader(
        data, data["test_indices"], args.batch_size, shuffle=False, include_phase=is_pino
    )

    coords = (
        torch.stack(
            torch.meshgrid(
                torch.as_tensor(data["y"], device=device),
                torch.as_tensor(data["x"], device=device),
                indexing="ij",
            ),
            dim=0,
        ).float()
        * 2.0
        - 1.0
    )

    gas_mask = ~data["solid_mask"]
    surface_mask = data["surface_mask"]
    mask_np = np.stack([gas_mask, surface_mask, surface_mask])
    channel_masks = torch.as_tensor(mask_np, device=device).float()
    eval_mask = channel_masks[None, None]

    dx = float(data["x"][1] - data["x"][0])
    dy = float(data["y"][1] - data["y"][0])

    field_mean = torch.as_tensor(data["field_mean"], device=device)
    field_std = torch.as_tensor(data["field_std"], device=device)
    param_mean = torch.as_tensor(data["parameter_mean"], device=device)
    param_std = torch.as_tensor(data["parameter_std"], device=device)
    time_mean = torch.as_tensor(data["time_mean"], device=device)
    time_std = torch.as_tensor(data["time_std"], device=device)

    model = TemporalFourierOperator(
        parameter_dim=data["parameters"].shape[-1],
        spatial_channels=2,
        output_channels=3,
        width=args.width,
        modes=args.modes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }
    if is_pino:
        history["train_data_loss"] = []
        history["train_pde_loss"] = []

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_valid = 0.0
        data_total = 0.0
        pde_total = 0.0

        for batch in train_loader:
            if is_pino:
                parameters, time, phase, targets = batch
                phase = phase.to(device)
            else:
                parameters, time, targets = batch
                phase = None

            parameters = parameters.to(device)
            time = time.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(parameters, time, coords)
            data_mse, mae, valid = loss_and_metrics(predictions, targets, eval_mask)

            if is_pino:
                pde_loss = physics_loss(
                    predictions,
                    parameters * param_std + param_mean,
                    time * time_std + time_mean,
                    phase,
                    channel_masks,
                    (field_mean, field_std),
                    dx,
                    dy,
                )
                pde_loss_clamped = torch.nan_to_num(pde_loss, nan=0.0, posinf=1.0e6, neginf=1.0e6)
                loss = data_mse + args.physics_weight * pde_loss_clamped
                data_total += data_mse.item() * valid.item()
                pde_total += pde_loss.item() * valid.item()
            else:
                loss = data_mse

            loss.backward()
            optimizer.step()

            valid_count = valid.item()
            train_loss += loss.item() * valid_count
            train_mae += mae.item() * valid_count
            train_valid += valid_count

        scheduler.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_valid = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if is_pino:
                    parameters, time, _, targets = batch
                else:
                    parameters, time, targets = batch

                parameters = parameters.to(device)
                time = time.to(device)
                targets = targets.to(device)

                predictions = model(parameters, time, coords)
                loss, mae, valid = loss_and_metrics(predictions, targets, eval_mask)

                valid_count = valid.item()
                val_loss += loss.item() * valid_count
                val_mae += mae.item() * valid_count
                val_valid += valid_count

        epoch_train_loss = train_loss / max(train_valid, 1.0)
        epoch_val_loss = val_loss / max(val_valid, 1.0)
        epoch_train_mae = train_mae / max(train_valid, 1.0)
        epoch_val_mae = val_mae / max(val_valid, 1.0)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_mae"].append(epoch_train_mae)
        history["val_mae"].append(epoch_val_mae)

        if is_pino:
            history["train_data_loss"].append(data_total / max(train_valid, 1.0))
            history["train_pde_loss"].append(pde_total / max(train_valid, 1.0))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": epoch_val_loss,
                    "model_type": args.model,
                    "physics_weight": args.physics_weight if is_pino else 0.0,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {epoch_train_loss:.6e} | "
                f"Val Loss: {epoch_val_loss:.6e} | "
                f"Val MAE: {epoch_val_mae:.6e}"
            )

    # Test evaluation
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_valid = 0.0

    with torch.no_grad():
        for batch in test_loader:
            if is_pino:
                parameters, time, _, targets = batch
            else:
                parameters, time, targets = batch

            parameters = parameters.to(device)
            time = time.to(device)
            targets = targets.to(device)

            predictions = model(parameters, time, coords)
            loss, mae, valid = loss_and_metrics(predictions, targets, eval_mask)

            valid_count = valid.item()
            test_loss += loss.item() * valid_count
            test_mae += mae.item() * valid_count
            test_valid += valid_count

    epoch_test_loss = test_loss / max(test_valid, 1.0)
    epoch_test_mae = test_mae / max(test_valid, 1.0)

    # Save artifacts
    np.savez(output_dir / "training_history.npz", **history)

    metrics = {
        "model_type": args.model,
        "best_validation_loss": float(best_val_loss),
        "test_loss": float(epoch_test_loss),
        "test_mae_normalized": float(epoch_test_mae),
        "train_trajectories": int(len(data["train_indices"])),
        "validation_trajectories": int(len(data["validation_indices"])),
        "test_trajectories": int(len(data["test_indices"])),
        "grid": [int(data["y"].shape[0]), int(data["x"].shape[0])],
        "epochs": int(args.epochs),
        "device": device_name,
    }
    if is_pino:
        metrics["physics_weight"] = float(args.physics_weight)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(9, 5))
    plt.semilogy(history["train_loss"], label="Train Loss")
    plt.semilogy(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()

    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

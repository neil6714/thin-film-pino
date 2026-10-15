"""Temporal Fourier Neural Operator (FNO) model architecture and training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal Fourier neural operator on ALD transient dynamics.")
    parser.add_argument("--input", required=True, help="Path to prepared transient ALD dataset (.npz).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for trajectories.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Initial learning rate.")
    parser.add_argument("--width", type=int, default=32, help="Hidden channel width in Fourier layers.")
    parser.add_argument("--modes", type=int, default=12, help="Fourier frequency cutoff modes.")
    parser.add_argument("--seed", type=int, default=123, help="Random number generator seed.")
    parser.add_argument(
        "--output-dir",
        default="results/stage4c_temporal_operator",
        help="Directory to save model checkpoints and training logs.",
    )
    return parser.parse_args()


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer operating in Fourier space."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        transformed = torch.fft.rfft2(x)
        output = torch.zeros_like(transformed)
        modes_y = min(self.modes, height)
        modes_x = min(self.modes, transformed.shape[-1])
        output[:, :, :modes_y, :modes_x] = torch.einsum(
            "bihw,iohw->bohw",
            transformed[:, :, :modes_y, :modes_x],
            self.weight[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(output, s=(height, width))


class FourierBlock(nn.Module):
    """Spectral convolution block with bypass projection and GELU non-linearity."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x))


class TemporalFourierOperator(nn.Module):
    """Temporal Fourier neural operator mapping physical parameters and time to field states."""

    def __init__(
        self,
        parameter_dim: int = 8,
        spatial_channels: int = 2,
        output_channels: int = 3,
        width: int = 32,
        modes: int = 12,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        in_channels = parameter_dim + 1 + spatial_channels
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([FourierBlock(width, modes) for _ in range(num_blocks)])
        self.projection = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, output_channels, kernel_size=1),
        )

    def forward(
        self,
        parameters: torch.Tensor,
        time: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps = time.shape[:2]
        height, width = coordinates.shape[-2:]

        p_expanded = parameters.unsqueeze(1).expand(-1, time_steps, -1)
        t_expanded = time.unsqueeze(-1)
        conditioned = torch.cat([p_expanded, t_expanded], dim=-1)

        conditioned = (
            conditioned.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width).reshape(batch_size * time_steps, -1, height, width)
        )
        coords = coordinates.unsqueeze(0).expand(batch_size * time_steps, -1, -1, -1)
        x = torch.cat([conditioned, coords], dim=1)

        h = self.lifting(x)
        for block in self.blocks:
            h = block(h)
        out = self.projection(h)
        return out.reshape(batch_size, time_steps, -1, height, width)


def loss_and_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked mean squared error and mean absolute error."""
    diff = prediction - target
    active = mask.expand_as(diff)
    valid = active.sum().clamp_min(1.0)
    mse = (diff.square() * active).sum() / valid
    mae = (diff.abs() * active).sum() / valid
    return mse, mae, valid


def make_loader(
    data: Mapping[str, Any], indices: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    fields = np.stack(
        [
            data["concentration"][indices],
            data["surface_coverage"][indices],
            data["film_thickness"][indices],
        ],
        axis=2,
    )
    tensors = (
        torch.from_numpy(data["parameters"][indices]),
        torch.from_numpy(data["time"][indices]),
        torch.from_numpy(fields),
    )
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    coordinates: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    is_train: bool,
) -> Tuple[float, float]:
    model.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    total_valid = 0.0

    for parameters, time, targets in loader:
        parameters = parameters.to(device)
        time = time.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            predictions = model(parameters, time, coordinates)
            loss, mae, valid = loss_and_metrics(predictions, targets, mask)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        valid_count = valid.item()
        total_loss += loss.item() * valid_count
        total_mae += mae.item() * valid_count
        total_valid += valid_count

    return total_loss / max(total_valid, 1.0), total_mae / max(total_valid, 1.0)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive integers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Executing temporal operator training on device: {device_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input).resolve()
    data = np.load(input_path)

    train_indices = data["train_indices"]
    validation_indices = data["validation_indices"]
    test_indices = data["test_indices"]
    height, width = data["concentration"].shape[-2:]

    coordinates = torch.stack(
        torch.meshgrid(
            torch.as_tensor(data["y"], dtype=torch.float32, device=device),
            torch.as_tensor(data["x"], dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=0,
    )
    coordinates = coordinates * 2.0 - 1.0

    mask = torch.stack(
        (
            torch.as_tensor(~data["solid_mask"], device=device),
            torch.as_tensor(data["surface_mask"], device=device),
            torch.as_tensor(data["surface_mask"], device=device),
        ),
        dim=0,
    ).to(torch.float32)

    train_loader = make_loader(data, train_indices, args.batch_size, True)
    validation_loader = make_loader(data, validation_indices, args.batch_size, False)
    test_loader = make_loader(data, test_indices, args.batch_size, False)

    model = TemporalFourierOperator(width=args.width, modes=args.modes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "validation_loss": [], "train_mae": [], "validation_mae": []}
    best_validation = float("inf")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {height} x {width} | Partition: train={len(train_indices)}, val={len(validation_indices)}, test={len(test_indices)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, coordinates, mask, device, True)
        validation_loss, validation_mae = run_epoch(model, validation_loader, optimizer, coordinates, mask, device, False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_mae"].append(train_mae)
        history["validation_mae"].append(validation_mae)

        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.6e} | val={validation_loss:.6e} | val MAE={validation_mae:.6e}")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_mae = run_epoch(model, test_loader, optimizer, coordinates, mask, device, False)

    metrics = {
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "test_mae_normalized": test_mae,
        "train_trajectories": int(len(train_indices)),
        "validation_trajectories": int(len(validation_indices)),
        "test_trajectories": int(len(test_indices)),
        "grid": [int(height), int(width)],
        "epochs": args.epochs,
        "device": device_name,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(output_dir / "training_history.npz", **{key: np.asarray(value) for key, value in history.items()})

    plt.figure(figsize=(8, 5))
    plt.semilogy(history["train_loss"], label="Training Loss")
    plt.semilogy(history["validation_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE Loss")
    plt.title("Temporal Operator Training Trajectory")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()

    print(f"\nFinal Test Loss = {test_loss:.6e} | Test MAE = {test_mae:.6e}")
    print(f"Saved model checkpoint and metrics to: {output_dir}")


if __name__ == "__main__":
    main()

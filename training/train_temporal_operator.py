import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CUDA temporal Fourier neural operator.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="results/stage4c_temporal_operator")
    return parser.parse_args()


class SpectralConv2d(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
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
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.spectral(x) + self.pointwise(x))


class TemporalFourierOperator(nn.Module):
    def __init__(self, input_channels=11, output_channels=3, width=32, modes=12):
        super().__init__()
        self.lift = nn.Conv2d(input_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FourierBlock(width, modes) for _ in range(4))
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, output_channels, kernel_size=1),
        )

    def forward(self, parameters, time, coordinates):
        batch, steps = time.shape
        height, width = coordinates.shape[-2:]
        params = parameters[:, None, :, None, None].expand(batch, steps, -1, height, width)
        clock = time[:, :, None, None, None].expand(batch, steps, 1, height, width)
        coords = coordinates[None, None].expand(batch, steps, -1, -1, -1)
        features = torch.cat((params, clock, coords), dim=2)
        features = features.reshape(batch * steps, features.shape[2], height, width)
        hidden = self.lift(features)
        for block in self.blocks:
            hidden = block(hidden)
        prediction = self.project(hidden)
        return prediction.reshape(batch, steps, -1, height, width)


def make_loader(data, indices, batch_size, shuffle):
    tensors = (
        torch.from_numpy(data["parameters"][indices]),
        torch.from_numpy(data["time"][indices]),
        torch.from_numpy(np.stack(
            [data["concentration"][indices], data["surface_coverage"][indices], data["film_thickness"][indices]],
            axis=2,
        )),
    )
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def loss_and_metrics(prediction, target, mask):
    expanded_mask = mask[None, None]
    squared = (prediction - target).square()
    absolute = (prediction - target).abs()
    denominator = expanded_mask.sum().clamp_min(1.0)
    loss = (squared * expanded_mask).sum() / denominator
    mae = (absolute * expanded_mask).sum() / denominator
    return loss, mae


def run_epoch(model, loader, optimizer, coordinates, mask, device, train):
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    for parameters, time, target in loader:
        parameters = parameters.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            prediction = model(parameters, time, coordinates)
            loss, mae = loss_and_metrics(prediction, target, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = parameters.shape[0]
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size
        total_count += batch_size
    return total_loss / total_count, total_mae / total_count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4C requires a CUDA-enabled Colab runtime.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    data = np.load(args.input)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Grid: {height} x {width}; train/validation/test: {len(train_indices)}/{len(validation_indices)}/{len(test_indices)}")
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
        "device": torch.cuda.get_device_name(0),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(output_dir / "training_history.npz", **{key: np.asarray(value) for key, value in history.items()})
    plt.figure(figsize=(8, 5))
    plt.semilogy(history["train_loss"], label="Train")
    plt.semilogy(history["validation_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE")
    plt.title("Stage 4C Temporal Operator Training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()
    print(f"Test loss={test_loss:.6e} | test MAE={test_mae:.6e}")
    print(f"Saved checkpoint and metrics to: {output_dir}")


if __name__ == "__main__":
    main()

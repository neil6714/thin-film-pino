"""Unified training pipeline for temporal Fourier neural operators and physics-informed surrogates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thin_film_pino.models.fno import TemporalFourierOperator
from thin_film_pino.training.pino import compute_physics_loss


def set_seed(seed: int) -> None:
    """Set random number generator seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal Fourier neural operators (FNO/PINO) with unified hyperparameters."
    )
    parser.add_argument(
        "--model",
        choices=["fno", "pino"],
        default="fno",
        help="Model architecture: 'fno' (data-driven) or 'pino' (physics-informed).",
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
        "--weight-decay", type=float, default=1.0e-5, help="Weight decay for AdamW optimizer."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Maximum gradient norm for clipping."
    )
    parser.add_argument(
        "--width", type=int, default=32, help="Hidden channel width in Fourier layers."
    )
    parser.add_argument("--modes", type=int, default=12, help="Fourier frequency cutoff modes.")
    parser.add_argument(
        "--num-blocks", type=int, default=4, help="Number of stacked Fourier blocks."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random number generator seed.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated list of seeds for multi-seed execution (e.g. '123,456,789').",
    )
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0e-4,
        help="Regularization weight for PDE residual penalty (PINO mode).",
    )
    parser.add_argument(
        "--max-train-trajectories",
        type=int,
        default=None,
        help="Optional maximum number of training trajectories for few-shot experiments.",
    )
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=2,
        help="Number of collocation points per phase for continuous physics loss.",
    )
    parser.add_argument(
        "--channel-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Loss weights for concentration, surface coverage, and thickness channels.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save model checkpoints and logs. Defaults to runs/<model>.",
    )
    return parser.parse_args(args)


def loss_and_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: Sequence[float] | torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked MSE, MAE, and valid cell count, averaging per channel before combining.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted field tensor with channels at dimension -3 (e.g. (B, T, C, H, W) or (C, H, W)).
    target : torch.Tensor
        Ground truth field tensor with identical shape to prediction.
    mask : torch.Tensor
        Boolean or float mask tensor with channels at dimension -3 or broadcastable to prediction.
    channel_weights : optional
        Per-channel weighting factors. If None, channels are weighted equally.

    Returns
    -------
    mse : torch.Tensor
        Weighted mean squared error averaged over active channel regions.
    mae : torch.Tensor
        Weighted mean absolute error averaged over active channel regions.
    valid_cells : torch.Tensor
        Total active element count across all evaluated channels.
    """
    diff = prediction - target
    num_channels = prediction.shape[-3]

    if channel_weights is None:
        weights = (
            torch.ones(num_channels, device=prediction.device, dtype=prediction.dtype)
            / num_channels
        )
    else:
        w_tensor = torch.as_tensor(
            channel_weights, device=prediction.device, dtype=prediction.dtype
        )
        weights = w_tensor / w_tensor.sum().clamp_min(1.0e-8)

    channel_mses = []
    channel_maes = []
    total_valid = torch.tensor(0.0, device=prediction.device)

    for c in range(num_channels):
        diff_c = diff.select(dim=-3, index=c)
        if mask.ndim >= 3 and mask.shape[-3] == num_channels:
            mask_c = mask.select(dim=-3, index=c)
        elif mask.ndim >= 3 and mask.shape[-3] == 1:
            mask_c = mask.select(dim=-3, index=0)
        else:
            mask_c = mask

        active_c = mask_c.expand_as(diff_c).float()
        valid_c = active_c.sum().clamp_min(1.0)
        total_valid = total_valid + valid_c

        mse_c = (diff_c.square() * active_c).sum() / valid_c
        mae_c = (diff_c.abs() * active_c).sum() / valid_c

        channel_mses.append(mse_c)
        channel_maes.append(mae_c)

    stacked_mse = torch.stack(channel_mses)
    stacked_mae = torch.stack(channel_maes)

    weighted_mse = (stacked_mse * weights).sum()
    weighted_mae = (stacked_mae * weights).sum()

    return weighted_mse, weighted_mae, total_valid


def load_dataset_in_memory(path: Path) -> Dict[str, np.ndarray]:
    """Load all arrays from an .npz archive into memory to avoid repeated decompression."""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def compute_normalized_coordinates(
    x: np.ndarray, y: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Scale spatial coordinates to [-1, 1] using min and max bounds."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_scaled = (torch.as_tensor(x, device=device).float() - x_min) / x_range * 2.0 - 1.0
    y_scaled = (torch.as_tensor(y, device=device).float() - y_min) / y_range * 2.0 - 1.0

    return torch.stack(torch.meshgrid(y_scaled, x_scaled, indexing="ij"), dim=0)


def make_loader(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_cycles: int = 10,
) -> DataLoader:
    """Build PyTorch DataLoader with in-memory arrays and complete conditioning inputs."""
    c = data["concentration"][indices]
    theta = data["surface_coverage"][indices]
    h = data["film_thickness"][indices]
    fields = np.stack([c, theta, h], axis=2)

    params = data["parameters"][indices]
    time = data["time"][indices]

    tensors: List[torch.Tensor] = [
        torch.from_numpy(params).float(),
        torch.from_numpy(time).float(),
    ]

    # Phase conditioning: (B, T)
    if "phase" in data:
        tensors.append(torch.from_numpy(data["phase"][indices]).long())
    else:
        tensors.append(torch.zeros(len(indices), time.shape[1], dtype=torch.long))

    # Cycle fraction conditioning: (B, T)
    if "cycle" in data:
        cycle_raw = data["cycle"][indices].astype(np.float32)
        max_c = float(np.max(data["cycle"])) if np.max(data["cycle"]) > 0 else float(num_cycles)
        cycle_frac = cycle_raw / max(max_c, 1.0)
        tensors.append(torch.from_numpy(cycle_frac).float())
    else:
        tensors.append(torch.zeros(len(indices), time.shape[1], dtype=torch.float32))

    tensors.append(torch.from_numpy(fields).float())

    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


class ZeroBaseline(nn.Module):
    """Baseline predictor outputting normalized zero (exact train mean)."""

    def forward(
        self,
        params: torch.Tensor,
        time: torch.Tensor,
        coords: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        b_size = params.shape[0]
        t_size = time.shape[1] if time.ndim >= 2 else 1
        h, w = coords.shape[-2:]
        return torch.zeros((b_size, t_size, 3, h, w), device=params.device, dtype=torch.float32)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    coords: torch.Tensor,
    eval_mask: torch.Tensor,
    channel_weights: Sequence[float],
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader and compute weighted MSE and MAE."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            params, time, phase, cycle_frac, targets = (t.to(device) for t in batch)
            preds = model(
                params,
                time,
                coords,
                phase=phase,
                cycle_fraction=cycle_frac,
            )
            mse, mae, _ = loss_and_metrics(preds, targets, eval_mask, channel_weights)
            b_size = params.shape[0]
            total_mse += mse.item() * b_size
            total_mae += mae.item() * b_size
            total_samples += b_size

    return total_mse / max(total_samples, 1), total_mae / max(total_samples, 1)


def train_single_run(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    data: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute training run for a single seed with unified hyperparameters."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_pino = args.model == "pino"
    num_cycles = int(data.get("cycle", np.zeros(1)).max()) or 10

    train_indices = data["train_indices"]
    if (
        getattr(args, "max_train_trajectories", None) is not None
        and args.max_train_trajectories > 0
    ):
        train_indices = train_indices[: args.max_train_trajectories]

    train_loader = make_loader(
        data, train_indices, args.batch_size, shuffle=True, num_cycles=num_cycles
    )
    val_loader = make_loader(
        data, data["validation_indices"], args.batch_size, shuffle=False, num_cycles=num_cycles
    )
    test_loader = make_loader(
        data, data["test_indices"], args.batch_size, shuffle=False, num_cycles=num_cycles
    )

    coords_2d = compute_normalized_coordinates(data["x"], data["y"], device)

    # Spatial geometry mask channel
    gas_mask_t = torch.as_tensor(~data["solid_mask"], device=device).float().unsqueeze(0)
    spatial_coords = torch.cat([coords_2d, gas_mask_t], dim=0)

    # Evaluation masks for loss computation
    surface_mask_t = torch.as_tensor(data["surface_mask"], device=device).float().unsqueeze(0)
    channel_masks = torch.cat([gas_mask_t, surface_mask_t, surface_mask_t], dim=0)
    eval_mask = channel_masks[None, None]

    dx = float(data["x"][1] - data["x"][0])
    dy = float(data["y"][1] - data["y"][0])

    field_mean = torch.as_tensor(data["field_mean"], device=device)
    field_std = torch.as_tensor(data["field_std"], device=device)
    param_mean = torch.as_tensor(data["parameter_mean"], device=device)
    param_std = torch.as_tensor(data["parameter_std"], device=device)
    time_mean = torch.as_tensor(data["time_mean"], device=device)
    time_std = torch.as_tensor(data["time_std"], device=device)

    param_dim = data["parameters"].shape[-1]
    num_phases = 5

    model = TemporalFourierOperator(
        parameter_dim=param_dim,
        spatial_channels=3,
        num_phases=num_phases,
        include_cycle=True,
        output_channels=3,
        width=args.width,
        modes=args.modes,
        num_blocks=args.num_blocks,
    ).to(device)

    # Unified optimizer and scheduler across FNO and PINO
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-6
    )

    # Built-in sanity check: evaluate train-mean baseline (predicting 0 in normalized units)
    mean_mse, mean_mae = evaluate_loader(
        ZeroBaseline(),
        train_loader,
        spatial_coords,
        eval_mask,
        args.channel_weights,
        device,
    )
    print(
        f"Train-mean baseline (normalized zero predictor): MSE = {mean_mse:.6f}, MAE = {mean_mae:.6f}"
    )

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }
    if is_pino:
        history["train_data_loss"] = []
        history["train_pde_loss"] = []
        history["train_transport_res"] = []
        history["train_coverage_res"] = []
        history["train_growth_res"] = []

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        data_loss_sum = 0.0
        pde_loss_sum = 0.0
        transport_res_sum = 0.0
        coverage_res_sum = 0.0
        growth_res_sum = 0.0
        total_samples = 0

        for batch in train_loader:
            params, time, phase, cycle_frac, targets = (t.to(device) for t in batch)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                params,
                time,
                spatial_coords,
                phase=phase,
                cycle_fraction=cycle_frac,
            )

            data_mse, mae, _ = loss_and_metrics(
                predictions, targets, eval_mask, args.channel_weights
            )

            b_size = params.shape[0]

            if is_pino:
                num_colloc = getattr(args, "collocation_points", 2)
                pde_loss, pde_metrics = compute_physics_loss(
                    model=model,
                    parameters_norm=params,
                    parameters_phys=params * param_std + param_mean,
                    spatial_coords=spatial_coords,
                    gas_mask=gas_mask_t.squeeze(0).bool(),
                    surface_mask=surface_mask_t.squeeze(0).bool(),
                    stats=(field_mean, field_std, time_mean, time_std),
                    dx=dx,
                    dy=dy,
                    num_collocation_per_phase=num_colloc,
                    num_cycles=num_cycles,
                    k_growth=0.02,
                )
                loss = data_mse + args.physics_weight * pde_loss
                data_loss_sum += data_mse.item() * b_size
                pde_loss_sum += pde_loss.item() * b_size
                transport_res_sum += pde_metrics["transport_mse"] * b_size
                coverage_res_sum += pde_metrics["coverage_mse"] * b_size
                growth_res_sum += pde_metrics["growth_mse"] * b_size
            else:
                loss = data_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item() * b_size
            train_mae_sum += mae.item() * b_size
            total_samples += b_size

        scheduler.step()

        epoch_train_loss = train_loss_sum / max(total_samples, 1)
        epoch_train_mae = train_mae_sum / max(total_samples, 1)

        epoch_val_loss, epoch_val_mae = evaluate_loader(
            model, val_loader, spatial_coords, eval_mask, args.channel_weights, device
        )

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_mae"].append(epoch_train_mae)
        history["val_mae"].append(epoch_val_mae)

        if is_pino:
            history["train_data_loss"].append(data_loss_sum / max(total_samples, 1))
            history["train_pde_loss"].append(pde_loss_sum / max(total_samples, 1))
            history["train_transport_res"].append(transport_res_sum / max(total_samples, 1))
            history["train_coverage_res"].append(coverage_res_sum / max(total_samples, 1))
            history["train_growth_res"].append(growth_res_sum / max(total_samples, 1))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch

            # Save state_dict alone for weights_only=True compatibility
            torch.save(model.state_dict(), output_dir / "best_model.pt")

            # Save JSON metadata config alongside checkpoint
            input_file = Path(args.input)
            dataset_hash = compute_file_hash(input_file) if input_file.exists() else "unknown"
            git_commit = get_git_commit()

            model_config = {
                "model_type": args.model,
                "epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "architecture": {
                    "parameter_dim": int(param_dim),
                    "spatial_channels": 3,
                    "num_phases": int(num_phases),
                    "include_cycle": True,
                    "output_channels": 3,
                    "width": int(args.width),
                    "modes": int(args.modes),
                    "num_blocks": int(args.num_blocks),
                },
                "normalization_statistics": {
                    "field_mean": data["field_mean"].tolist(),
                    "field_std": data["field_std"].tolist(),
                    "parameter_mean": data["parameter_mean"].tolist(),
                    "parameter_std": data["parameter_std"].tolist(),
                    "time_mean": float(data["time_mean"]),
                    "time_std": float(data["time_std"]),
                },
                "dataset_hash": dataset_hash,
                "git_commit": git_commit,
                "seed": int(seed),
                "channel_weights": [float(w) for w in args.channel_weights],
                "args": vars(args),
            }
            with open(output_dir / "best_model_config.json", "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {epoch_train_loss:.6e} | "
                f"Val Loss: {epoch_val_loss:.6e} | "
                f"Val MAE: {epoch_val_mae:.6e}"
            )

    # Test set evaluation
    # Load best checkpoint weights with weights_only=True
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_loss, test_mae = evaluate_loader(
        model, test_loader, spatial_coords, eval_mask, args.channel_weights, device
    )

    np.savez(output_dir / "training_history.npz", **history)

    metrics = {
        "model_type": args.model,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_mae_normalized": float(test_mae),
        "train_mean_baseline_mse": float(mean_mse),
        "train_mean_baseline_mae": float(mean_mae),
        "train_trajectories": int(len(train_indices)),
        "validation_trajectories": int(len(data["validation_indices"])),
        "test_trajectories": int(len(data["test_indices"])),
        "grid": [int(data["y"].shape[0]), int(data["x"].shape[0])],
        "epochs": int(args.epochs),
        "device": str(device),
    }
    if is_pino:
        metrics["physics_weight"] = float(args.physics_weight)
        if history.get("train_pde_loss"):
            metrics["final_train_pde_loss"] = float(history["train_pde_loss"][-1])
            metrics["final_transport_mse"] = float(history["train_transport_res"][-1])
            metrics["final_coverage_mse"] = float(history["train_coverage_res"][-1])
            metrics["final_growth_mse"] = float(history["train_growth_res"][-1])

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

    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset archive not found: {input_path}")

    # Load dataset archive into memory once
    data = load_dataset_in_memory(input_path)

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    )

    base_output = Path(args.output_dir or f"runs/{args.model}").resolve()

    if len(seeds) == 1:
        train_single_run(args, seeds[0], base_output, data, device)
        print(f"Training completed. Metrics written to {base_output / 'metrics.json'}")
    else:
        multi_metrics = {}
        for seed in seeds:
            print(f"\n--- Training {args.model.upper()} with Seed {seed} ---")
            seed_dir = base_output / f"seed_{seed}"
            m = train_single_run(args, seed, seed_dir, data, device)
            multi_metrics[f"seed_{seed}"] = m

        summary_path = base_output / "multi_seed_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(multi_metrics, f, indent=2)
        print(f"Multi-seed training completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""Compare trained temporal models on the shared held-out trajectories."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from training.train_temporal_operator import TemporalFourierOperator

def evaluate(data, checkpoint_path, width=32, modes=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    model = TemporalFourierOperator(width=cfg.get("width", width), modes=cfg.get("modes", modes)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    coords = torch.stack(torch.meshgrid(torch.as_tensor(data["y"], device=device), torch.as_tensor(data["x"], device=device), indexing="ij"), 0).float() * 2 - 1
    idx = data["test_indices"]
    p = torch.as_tensor(data["parameters"][idx], device=device); t = torch.as_tensor(data["time"][idx], device=device)
    target = torch.as_tensor(np.stack([data["concentration"][idx], data["surface_coverage"][idx], data["film_thickness"][idx]], 2), device=device)
    with torch.no_grad(): pred = model(p, t, coords)
    mask = torch.as_tensor(np.stack([~data["solid_mask"], data["surface_mask"], data["surface_mask"]]), device=device).float()[None, None]
    scale = torch.as_tensor(data["field_std"], device=device)[None, None, :, None, None]; mean = torch.as_tensor(data["field_mean"], device=device)[None, None, :, None, None]
    err = (pred - target).abs() * scale
    vals = (err * mask).sum(dim=(0,1,3,4)) / mask.sum(dim=(0,1,3,4)).clamp_min(1)
    return {"concentration_mae": float(vals[0]), "coverage_mae": float(vals[1]), "thickness_mae": float(vals[2]), "overall_mae": float(vals.mean())}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", default="results/model_comparison"); ap.add_argument("--operator", default="results/stage4c_temporal_operator/best_model.pt"); ap.add_argument("--pino", default="results/stage4d_temporal_pino/best_model.pt"); args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); data = np.load(args.input)
    rows = {}
    for name, path in (("temporal_operator", args.operator), ("temporal_pino", args.pino)):
        if Path(path).exists(): rows[name] = evaluate(data, path)
    for name, path in (("temporal_operator", Path(args.operator).parent), ("temporal_pino", Path(args.pino).parent)):
        metrics = path / "metrics.json"
        if metrics.exists(): rows[name].update({"test_loss": json.loads(metrics.read_text()).get("test_loss")})
    (out / "comparison_metrics.json").write_text(json.dumps(rows, indent=2))
    names = list(rows); x = np.arange(len(names)); w = .25
    plt.figure(figsize=(9,5))
    for j, key in enumerate(("concentration_mae", "coverage_mae", "thickness_mae")): plt.bar(x + (j-1)*w, [rows[n][key] for n in names], w, label=key.replace("_", " "))
    plt.xticks(x, names, rotation=15); plt.ylabel("Physical-unit MAE"); plt.legend(); plt.tight_layout(); plt.savefig(out / "physical_mae_comparison.png", dpi=200); plt.close()
    print(json.dumps(rows, indent=2)); print(f"Saved comparison to {out}")
if __name__ == "__main__": main()

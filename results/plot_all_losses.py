"""Plot every available model loss in one figure."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results"); ap.add_argument("--output", default="results/all_model_losses.png"); args = ap.parse_args()
    root = Path(args.root); plt.figure(figsize=(11, 6)); plotted = set()
    for history in sorted(root.glob("**/training_history.npz")):
        data = np.load(history)
        for key in data.files:
            if "loss" in key.lower():
                label = f"{history.parent.name}: {key}"
                plt.semilogy(np.maximum(data[key], 1e-12), label=label); plotted.add(history.parent)
    for metrics in sorted(root.glob("**/metrics.json")):
        if metrics.parent in plotted: continue
        try: values = json.loads(metrics.read_text())
        except (OSError, json.JSONDecodeError): continue
        loss = values.get("test_loss", values.get("best_validation_loss"))
        if isinstance(loss, (int, float)) and np.isfinite(loss):
            plt.axhline(max(float(loss), 1e-12), linestyle="--", label=f"{metrics.parent.name}: reported loss")
    if not plotted and not list(root.glob("**/metrics.json")): raise FileNotFoundError(f"No model losses found below {root}")
    plt.xlabel("Epoch (curves) / reference (dashed lines)"); plt.ylabel("Loss (log scale)"); plt.title("All Available Model Losses"); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=220); print(f"Saved: {out}")
if __name__ == "__main__": main()

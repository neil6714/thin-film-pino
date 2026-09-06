"""Validate transient ALD dataset integrity and physical invariants."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", default="results/stage4_validation"); args = ap.parse_args()
    d = np.load(args.input, allow_pickle=False); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); checks = {}
    fields = [d["concentration"], d["surface_coverage"], d["film_thickness"]]
    checks["finite"] = all(bool(np.isfinite(x).all()) for x in fields)
    checks["concentration_nonnegative"] = bool((fields[0] >= 0).all())
    checks["coverage_bounded"] = bool(((fields[1] >= -1e-6) & (fields[1] <= 1 + 1e-6)).all())
    checks["thickness_nonnegative"] = bool((fields[2] >= -1e-8).all())
    checks["thickness_non_decreasing"] = bool((np.diff(fields[2], axis=1) >= -1e-7).all())
    checks["time_monotonic"] = bool((np.diff(d["time"], axis=1) >= -1e-8).all())
    checks["phase_ids_valid"] = bool(np.isin(d["phase"], np.unique(d["phase"])).all())
    splits = [set(map(int, d[k].tolist())) for k in ("train_indices", "validation_indices", "test_indices") if k in d]
    checks["splits_disjoint"] = len(splits) == 3 and not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    checks["normalization_stats_present"] = all(k in d for k in ("field_mean", "field_std", "parameter_mean", "parameter_std"))
    report = {"input": str(args.input), "shapes": {k: list(d[k].shape) for k in ("parameters", "time", "phase", "concentration", "surface_coverage", "film_thickness") if k in d}, "checks": checks, "passed": all(checks.values())}
    (out / "transient_dataset_audit.json").write_text(json.dumps(report, indent=2)); (out / "transient_dataset_audit.md").write_text("# Transient Dataset Audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report, indent=2)); raise SystemExit(0 if report["passed"] else 1)
if __name__ == "__main__": main()

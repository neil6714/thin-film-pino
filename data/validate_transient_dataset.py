"""Integrity and physical invariant validation for transient ALD datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify physical invariants, bounds, and split integrity in ALD datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to transient ALD .npz archive.",
    )
    parser.add_argument(
        "--output",
        default="results/stage4_validation",
        help="Directory to save validation audit reports.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Target dataset not found: {input_path}")

    data = np.load(input_path, allow_pickle=False)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, bool] = {}
    raw_fields = [data["concentration"], data["surface_coverage"], data["film_thickness"]]

    is_normalized = "field_mean" in data and "field_std" in data
    if is_normalized:
        fields = [x * s + m for x, m, s in zip(raw_fields, data["field_mean"], data["field_std"])]
        checks["fields_denormalized_for_physics_checks"] = True
    else:
        fields = raw_fields
        checks["fields_denormalized_for_physics_checks"] = False

    checks["finite"] = all(bool(np.isfinite(x).all()) for x in fields)
    checks["concentration_nonnegative"] = bool((fields[0] >= -1e-6).all())
    checks["coverage_bounded"] = bool(((fields[1] >= -1e-6) & (fields[1] <= 1.0 + 1e-6)).all())
    checks["thickness_nonnegative"] = bool((fields[2] >= -1e-8).all())

    if fields[2].ndim >= 2:
        checks["thickness_non_decreasing"] = bool((np.diff(fields[2], axis=1) >= -1e-7).all())

    if "time" in data:
        checks["time_monotonic"] = bool((np.diff(data["time"], axis=1) >= -1e-8).all())

    if "phase" in data:
        # Physical ALD phase IDs: 0 (initial), 1 (pulse), 2 (purge), 3 (reaction), 4 (purge)
        valid_phase_ids = np.arange(5)
        checks["phase_ids_valid"] = bool(np.isin(data["phase"], valid_phase_ids).all())

    is_prepared = "train_indices" in data or "test_indices" in data
    if is_prepared:
        splits = [
            set(map(int, data[k].tolist()))
            for k in ("train_indices", "validation_indices", "test_indices")
            if k in data
        ]
        checks["splits_disjoint"] = (
            len(splits) == 3 and not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
        )
        checks["normalization_stats_present"] = all(
            k in data for k in ("field_mean", "field_std", "parameter_mean", "parameter_std")
        )

    passed = all(checks.values())
    report: Dict[str, Any] = {
        "input": str(input_path),
        "shapes": {
            k: list(data[k].shape)
            for k in ("parameters", "time", "phase", "concentration", "surface_coverage", "film_thickness")
            if k in data
        },
        "checks": checks,
        "passed": passed,
    }

    audit_json = output_dir / "transient_dataset_audit.json"
    audit_json.write_text(json.dumps(report, indent=2))

    audit_md = output_dir / "transient_dataset_audit.md"
    audit_md.write_text(f"# Transient Dataset Audit Report\n\n```json\n{json.dumps(report, indent=2)}\n```\n")

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""CLI entry point for ALD dataset physical invariant and split validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from thin_film_pino.data.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify physical invariants, bounds, and split integrity in ALD datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ALD .npz dataset archive.",
    )
    parser.add_argument(
        "--output",
        default="results/validation",
        help="Directory to save validation audit reports.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Target dataset not found: {input_path}")

    data = np.load(input_path, allow_pickle=False)
    passed, report = validate_dataset(data, input_path=input_path)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_json = output_dir / f"{input_path.stem}_audit.json"
    audit_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status_str = "PASS" if passed else "FAIL"
    print(f"\nDataset Validation [{status_str}]: {input_path}")
    for name, res in report["checks"].items():
        print(f"[{'PASS' if res else 'FAIL':4s}] {name}")

    print(f"\nAudit report saved to: {audit_json}")

    if not passed:
        print("Dataset validation failed with errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

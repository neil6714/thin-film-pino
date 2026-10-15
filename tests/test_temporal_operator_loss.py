"""Unit test validating masked loss aggregation and reduction semantics in neural operator training."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_temporal_operator import loss_and_metrics


def test_masked_loss_aggregation() -> None:
    """Verify that masked MSE, MAE, and valid cell counts match exact analytical reductions."""
    prediction = torch.tensor([[[[[1.0, 3.0], [5.0, 7.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss, mae, valid = loss_and_metrics(prediction, target, mask)

    assert torch.equal(valid, torch.tensor(2.0)), f"Expected 2.0 valid cells, got {valid}"
    assert torch.equal(loss, torch.tensor(13.0)), f"Expected MSE of 13.0, got {loss}"
    assert torch.equal(mae, torch.tensor(3.0)), f"Expected MAE of 3.0, got {mae}"


def main() -> None:
    test_masked_loss_aggregation()
    print("Masked loss aggregation test passed successfully (MSE=13.0, MAE=3.0, valid=2).")


if __name__ == "__main__":
    main()

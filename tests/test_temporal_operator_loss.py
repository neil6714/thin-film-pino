"""Unit test validating masked loss aggregation and reduction semantics in neural operator training."""

from __future__ import annotations

import torch

from thin_film_pino.training.train import loss_and_metrics


def test_masked_loss_aggregation_single_channel() -> None:
    """Verify that single-channel masked MSE, MAE, and valid cell count match exact reductions."""
    prediction = torch.tensor([[[[[1.0, 3.0], [5.0, 7.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    loss, mae, valid = loss_and_metrics(prediction, target, mask)

    assert torch.equal(valid, torch.tensor(2.0)), f"Expected 2.0 valid cells, got {valid}"
    assert torch.equal(loss, torch.tensor(13.0)), f"Expected MSE of 13.0, got {loss}"
    assert torch.equal(mae, torch.tensor(3.0)), f"Expected MAE of 3.0, got {mae}"


def test_masked_loss_aggregation_unequal_channels() -> None:
    """Verify that channels with unequal active cell counts are not dominated by dense channels.

    Channel 0 (gas): 4 active cells with error 1.0 -> MSE_0 = 1.0, MAE_0 = 1.0
    Channel 1 (surface): 1 active cell with error 3.0 -> MSE_1 = 9.0, MAE_1 = 3.0
    Under element-wise averaging: MSE = (4*1 + 1*9)/5 = 2.6
    Under per-channel masked mean: MSE = (1.0 + 9.0)/2 = 5.0, MAE = (1.0 + 3.0)/2 = 2.0
    """
    prediction = torch.zeros(1, 1, 2, 2, 2)
    prediction[0, 0, 0] = 1.0
    prediction[0, 0, 1, 0, 0] = 3.0
    target = torch.zeros_like(prediction)

    mask = torch.zeros(2, 2, 2)
    mask[0] = 1.0  # 4 active cells in channel 0
    mask[1, 0, 0] = 1.0  # 1 active cell in channel 1

    loss, mae, valid = loss_and_metrics(prediction, target, mask)

    assert loss.item() == 5.0, f"Expected channel-averaged MSE of 5.0, got {loss.item()}"
    assert mae.item() == 2.0, f"Expected channel-averaged MAE of 2.0, got {mae.item()}"
    assert valid.item() == 5.0, f"Expected total 5.0 valid cells, got {valid.item()}"

    # Test with custom channel weighting [0.25, 0.75]
    weighted_loss, weighted_mae, _ = loss_and_metrics(
        prediction, target, mask, channel_weights=[0.25, 0.75]
    )
    # Expected MSE: 0.25 * 1.0 + 0.75 * 9.0 = 7.0
    # Expected MAE: 0.25 * 1.0 + 0.75 * 3.0 = 2.5
    assert weighted_loss.item() == 7.0, f"Expected weighted MSE of 7.0, got {weighted_loss.item()}"
    assert weighted_mae.item() == 2.5, f"Expected weighted MAE of 2.5, got {weighted_mae.item()}"


def main() -> None:
    test_masked_loss_aggregation_single_channel()
    test_masked_loss_aggregation_unequal_channels()
    print("All masked loss aggregation tests passed.")


if __name__ == "__main__":
    main()

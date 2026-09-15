"""Deterministic smoke test for Stage 4C masked loss aggregation."""
import torch
from training.train_temporal_operator import loss_and_metrics

def main():
    prediction = torch.tensor([[[[[1.0, 3.0], [5.0, 7.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    loss, mae, valid = loss_and_metrics(prediction, target, mask)
    assert torch.equal(valid, torch.tensor(2.0))
    assert torch.equal(loss, torch.tensor(13.0))
    assert torch.equal(mae, torch.tensor(3.0))
    print("Stage 4C masked-loss smoke test passed: MSE=13.0, MAE=3.0, valid=2")

if __name__ == "__main__":
    main()

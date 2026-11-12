"""Training routines, loss functions, and physics residuals for neural operators."""

from thin_film_pino.training.pino import compute_physics_loss, sample_collocation_times
from thin_film_pino.training.train import loss_and_metrics

# Backward-compatible alias
physics_loss = compute_physics_loss

__all__ = [
    "compute_physics_loss",
    "loss_and_metrics",
    "physics_loss",
    "sample_collocation_times",
]

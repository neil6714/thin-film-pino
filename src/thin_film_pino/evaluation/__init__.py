"""Evaluation suites, metrics, diagnostics, and visualization tools for neural operators."""

from thin_film_pino.evaluation.evaluate import (
    evaluate_baselines,
    evaluate_checkpoint,
    evaluate_single_model,
    predict_batched,
)
from thin_film_pino.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_baseline_predictions,
    compute_bounds_violations,
    compute_channel_errors,
    compute_derived_cycle_quantities,
    compute_per_phase_metrics,
)
from thin_film_pino.evaluation.visualize import (
    generate_qualitative_figures,
    plot_field_comparison_trajectory,
)

__all__ = [
    "evaluate_checkpoint",
    "evaluate_single_model",
    "evaluate_baselines",
    "predict_batched",
    "compute_channel_errors",
    "compute_per_phase_metrics",
    "compute_derived_cycle_quantities",
    "compute_bounds_violations",
    "bootstrap_confidence_intervals",
    "compute_baseline_predictions",
    "generate_qualitative_figures",
    "plot_field_comparison_trajectory",
]

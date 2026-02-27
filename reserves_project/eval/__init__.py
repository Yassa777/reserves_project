"""Evaluation utilities and statistical tests."""

from .metrics import compute_metrics, naive_mae_scale, asymmetric_loss
from .leakage_checks import assert_no_future_in_history
from .windowing import compute_common_dates_by_group, filter_to_common_window
from .msvar_irf import generalized_irf, girf_to_long_df, summarize_regime_comparison
from .information_loss import (
    load_aligned_aggregation_forecasts,
    evaluate_information_loss_by_segment,
    compute_cancellation_index,
    summarize_information_loss,
)
from .mechanism_synthesis import (
    load_disentangling_metrics,
    load_regime_metrics,
    load_irf_metrics,
    load_information_loss_metrics,
    build_mechanism_synthesis_table,
    build_mechanism_detail_table,
)

__all__ = [
    "compute_metrics",
    "naive_mae_scale",
    "asymmetric_loss",
    "assert_no_future_in_history",
    "compute_common_dates_by_group",
    "filter_to_common_window",
    "generalized_irf",
    "girf_to_long_df",
    "summarize_regime_comparison",
    "load_aligned_aggregation_forecasts",
    "evaluate_information_loss_by_segment",
    "compute_cancellation_index",
    "summarize_information_loss",
    "load_disentangling_metrics",
    "load_regime_metrics",
    "load_irf_metrics",
    "load_information_loss_metrics",
    "build_mechanism_synthesis_table",
    "build_mechanism_detail_table",
]

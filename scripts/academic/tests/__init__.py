# Statistical tests for forecast evaluation
# Diebold-Mariano, MCS, density evaluation, encompassing tests

from .diebold_mariano import (
    diebold_mariano_test,
    dm_test_hln,
    dm_test_matrix,
    dm_test_vs_benchmark,
    format_dm_table_for_paper,
)

from .model_confidence_set import (
    model_confidence_set,
    mcs_summary_table,
    mcs_with_pvalues,
    format_mcs_table_for_paper,
)

from .density_evaluation import (
    crps_normal,
    crps_empirical,
    compute_crps_series,
    compare_crps,
    log_score_normal,
    compute_log_score_series,
    compare_log_scores,
    compute_pit,
    pit_histogram_test,
    pit_ks_test,
    evaluate_density_forecasts,
    density_evaluation_summary,
)

from .encompassing import (
    forecast_encompassing_test,
    fair_shiller_test,
    pairwise_encompassing_matrix,
    format_encompassing_table,
    encompassing_summary,
    optimal_combination_weights,
)

__all__ = [
    # Diebold-Mariano
    "diebold_mariano_test",
    "dm_test_hln",
    "dm_test_matrix",
    "dm_test_vs_benchmark",
    "format_dm_table_for_paper",
    # Model Confidence Set
    "model_confidence_set",
    "mcs_summary_table",
    "mcs_with_pvalues",
    "format_mcs_table_for_paper",
    # Density Evaluation
    "crps_normal",
    "crps_empirical",
    "compute_crps_series",
    "compare_crps",
    "log_score_normal",
    "compute_log_score_series",
    "compare_log_scores",
    "compute_pit",
    "pit_histogram_test",
    "pit_ks_test",
    "evaluate_density_forecasts",
    "density_evaluation_summary",
    # Encompassing
    "forecast_encompassing_test",
    "fair_shiller_test",
    "pairwise_encompassing_matrix",
    "format_encompassing_table",
    "encompassing_summary",
    "optimal_combination_weights",
]

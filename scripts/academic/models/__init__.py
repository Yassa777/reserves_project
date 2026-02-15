# Model implementations for academic pipeline
# BVAR, TVP-VAR, FAVAR, TVAR, MIDAS, DMA/DMS

from .tvp_var import TVP_VAR, fit_tvp_var_from_dataframe
from .tvp_var_diagnostics import (
    plot_trace,
    plot_convergence_diagnostics,
    plot_coefficient_paths,
    plot_tvp_vs_breaks,
    plot_volatility_paths,
    plot_forecast_fan,
    compute_rolling_forecast_metrics,
    rolling_backtest,
    create_comparison_summary
)

from .bvar import BayesianVAR, rolling_cv_rmse, grid_search_hyperparameters
from .bvar_diagnostics import (
    compute_rhat,
    compute_ess,
    diagnose_convergence,
    trace_plot_data,
    autocorrelation_data,
    posterior_predictive_check,
    geweke_test,
    create_diagnostic_report,
)

from .combination_methods import (
    equal_weight_combination,
    mse_weight_combination,
    granger_ramanathan_combination,
    trimmed_mean_combination,
    median_combination,
    get_combination_weights
)

from .forecast_combiner import (
    ForecastCombiner,
    rolling_combination_backtest,
    compare_all_methods
)

from .combination_analysis import (
    compute_forecast_metrics,
    relative_combination_value,
    combination_efficiency,
    compute_combination_diagnostics,
    analyze_weight_stability,
    create_summary_report
)

from .favar import FAVAR, rolling_favar_forecast, compute_forecast_metrics as favar_compute_metrics
from .factor_selection import (
    bai_ng_criteria,
    kaiser_criterion,
    elbow_detection,
    select_n_factors,
)

from .tvar import (
    ThresholdVAR,
    compute_threshold_variable,
    load_threshold_variable_from_fx,
)
from .tvar_tests import (
    linearity_test,
    bootstrap_linearity_test,
    threshold_confidence_interval,
    compare_tvar_msvar,
    regime_persistence_test,
)

from .midas import (
    MIDAS,
    MIDAS_AR,
    UMIDAS,
    prepare_hf_exchange_rate,
    align_midas_data,
    midas_information_gain,
)
from .midas_weights import (
    exp_almon_weights,
    beta_weights,
    step_weights,
    uniform_weights,
    declining_weights,
    compute_weights,
    weight_initial_params,
)

from .dma import (
    DynamicModelAveraging,
    DMAResults,
    StateDependentDMA,
    run_dma_grid_search,
    rolling_dma_backtest,
    compute_dma_metrics,
)
from .dma_visualization import (
    plot_dma_weights_stacked,
    plot_dms_selection_path,
    plot_weight_evolution_by_model,
    plot_alpha_sensitivity,
    plot_dma_vs_individual,
    plot_selection_frequency,
    plot_weight_heatmap,
    plot_performance_comparison,
    create_dma_report_figures,
)

__all__ = [
    # FAVAR
    'FAVAR',
    'rolling_favar_forecast',
    'favar_compute_metrics',
    'bai_ng_criteria',
    'kaiser_criterion',
    'elbow_detection',
    'select_n_factors',
    # TVP-VAR
    'TVP_VAR',
    'fit_tvp_var_from_dataframe',
    'plot_trace',
    'plot_convergence_diagnostics',
    'plot_coefficient_paths',
    'plot_tvp_vs_breaks',
    'plot_volatility_paths',
    'plot_forecast_fan',
    'compute_rolling_forecast_metrics',
    'rolling_backtest',
    'create_comparison_summary',
    # BVAR
    'BayesianVAR',
    'rolling_cv_rmse',
    'grid_search_hyperparameters',
    # BVAR diagnostics
    'compute_rhat',
    'compute_ess',
    'diagnose_convergence',
    'trace_plot_data',
    'autocorrelation_data',
    'posterior_predictive_check',
    'geweke_test',
    'create_diagnostic_report',
    # Combination methods
    'equal_weight_combination',
    'mse_weight_combination',
    'granger_ramanathan_combination',
    'trimmed_mean_combination',
    'median_combination',
    'get_combination_weights',
    # Combiner class
    'ForecastCombiner',
    'rolling_combination_backtest',
    'compare_all_methods',
    # Analysis
    'compute_forecast_metrics',
    'relative_combination_value',
    'combination_efficiency',
    'compute_combination_diagnostics',
    'analyze_weight_stability',
    'create_summary_report',
    # TVAR
    'ThresholdVAR',
    'compute_threshold_variable',
    'load_threshold_variable_from_fx',
    'linearity_test',
    'bootstrap_linearity_test',
    'threshold_confidence_interval',
    'compare_tvar_msvar',
    'regime_persistence_test',
    # MIDAS
    'MIDAS',
    'MIDAS_AR',
    'UMIDAS',
    'prepare_hf_exchange_rate',
    'align_midas_data',
    'midas_information_gain',
    'exp_almon_weights',
    'beta_weights',
    'step_weights',
    'uniform_weights',
    'declining_weights',
    'compute_weights',
    'weight_initial_params',
    # DMA/DMS
    'DynamicModelAveraging',
    'DMAResults',
    'StateDependentDMA',
    'run_dma_grid_search',
    'rolling_dma_backtest',
    'compute_dma_metrics',
    'plot_dma_weights_stacked',
    'plot_dms_selection_path',
    'plot_weight_evolution_by_model',
    'plot_alpha_sensitivity',
    'plot_dma_vs_individual',
    'plot_selection_frequency',
    'plot_weight_heatmap',
    'plot_performance_comparison',
    'create_dma_report_figures',
]

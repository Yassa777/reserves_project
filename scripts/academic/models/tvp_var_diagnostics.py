"""
TVP-VAR Diagnostics and Visualization.

Provides:
- Convergence diagnostics (trace plots, ESS, autocorrelation)
- Time-varying coefficient plots with credible intervals
- Comparison with structural break dates
- Volatility path plots
- Rolling backtest evaluation

Reference: Specification 04 - TVP-VAR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def plot_trace(
    chain: np.ndarray,
    param_name: str = "Parameter",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot trace of MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        MCMC samples (n_draws,)
    param_name : str
        Parameter name for title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Trace plot
    axes[0].plot(chain, alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Trace: {param_name}')
    axes[0].axhline(np.mean(chain), color='red', linestyle='--', alpha=0.7)

    # Posterior histogram
    axes[1].hist(chain, bins=50, density=True, alpha=0.7, edgecolor='white')
    axes[1].axvline(np.mean(chain), color='red', linestyle='--', label='Mean')
    axes[1].axvline(np.percentile(chain, 2.5), color='orange', linestyle=':', label='95% CI')
    axes[1].axvline(np.percentile(chain, 97.5), color='orange', linestyle=':')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Posterior: {param_name}')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_convergence_diagnostics(
    model,
    n_params: int = 6,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot comprehensive convergence diagnostics.

    Parameters
    ----------
    model : TVP_VAR
        Fitted TVP-VAR model
    n_params : int
        Number of parameters to show
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    # Get final period coefficients
    beta_final = model.beta_posterior[:, -1, :]  # (n_draws, n_states)
    n_states = beta_final.shape[1]
    n_show = min(n_params, n_states)

    fig, axes = plt.subplots(n_show, 2, figsize=figsize)

    for i in range(n_show):
        chain = beta_final[:, i]

        # Trace plot
        axes[i, 0].plot(chain, alpha=0.5, linewidth=0.3)
        axes[i, 0].axhline(np.mean(chain), color='red', linestyle='--', alpha=0.7)
        axes[i, 0].set_ylabel(f'$\\beta_{{{i}}}$')
        if i == 0:
            axes[i, 0].set_title('Trace Plots')

        # ACF plot (simplified)
        max_lag = min(50, len(chain) // 4)
        acf = np.zeros(max_lag)
        chain_centered = chain - np.mean(chain)
        var = np.var(chain)
        if var > 0:
            for lag in range(max_lag):
                acf[lag] = np.mean(chain_centered[lag:] * chain_centered[:-lag or None]) / var

        axes[i, 1].bar(range(max_lag), acf, alpha=0.7, width=0.8)
        axes[i, 1].axhline(0, color='black', linewidth=0.5)
        axes[i, 1].axhline(1.96 / np.sqrt(len(chain)), color='red', linestyle='--', alpha=0.5)
        axes[i, 1].axhline(-1.96 / np.sqrt(len(chain)), color='red', linestyle='--', alpha=0.5)
        axes[i, 1].set_xlim(0, max_lag)
        if i == 0:
            axes[i, 1].set_title('Autocorrelation')

    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Lag')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_coefficient_paths(
    coefs: Dict[str, np.ndarray],
    var_name: str = "Target",
    coef_indices: Optional[List[int]] = None,
    break_dates: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot time-varying coefficient paths with credible intervals.

    Parameters
    ----------
    coefs : dict
        Output from model.get_time_varying_coefficients()
    var_name : str
        Variable name for title
    coef_indices : list, optional
        Indices of coefficients to plot (default: all except constant)
    break_dates : list, optional
        Structural break dates to overlay
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    dates = coefs['dates']
    mean = coefs['mean']
    lower_16 = coefs['lower_16']
    upper_84 = coefs['upper_84']
    lower_5 = coefs['lower_5']
    upper_95 = coefs['upper_95']
    coef_names = coefs['coef_names']

    # Select coefficients to plot
    if coef_indices is None:
        # Skip constant, plot all lag coefficients
        coef_indices = list(range(1, len(coef_names)))

    n_coefs = len(coef_indices)
    n_cols = 2
    n_rows = (n_coefs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, coef_idx in enumerate(coef_indices):
        ax = axes[idx // n_cols, idx % n_cols]

        # Plot credible intervals
        ax.fill_between(dates, lower_5[:, coef_idx], upper_95[:, coef_idx],
                       alpha=0.2, color='blue', label='90% CI')
        ax.fill_between(dates, lower_16[:, coef_idx], upper_84[:, coef_idx],
                       alpha=0.3, color='blue', label='68% CI')
        ax.plot(dates, mean[:, coef_idx], color='blue', linewidth=1.5, label='Mean')

        # Zero line
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Structural breaks
        if break_dates:
            for bd in break_dates:
                ax.axvline(pd.Timestamp(bd), color='red', linestyle='--',
                          alpha=0.7, linewidth=1)

        ax.set_title(coef_names[coef_idx], fontsize=10)
        ax.set_ylabel('Coefficient')

        # Format dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

    # Hide empty subplots
    for idx in range(len(coef_indices), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Time-Varying Coefficients: {var_name} Equation', fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_tvp_vs_breaks(
    coefs: Dict[str, np.ndarray],
    break_dates: List[str],
    break_labels: Optional[List[str]] = None,
    target_coef_idx: int = 1,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare TVP coefficient evolution with structural break dates.

    Parameters
    ----------
    coefs : dict
        Output from model.get_time_varying_coefficients()
    break_dates : list
        Structural break dates from Bai-Perron
    break_labels : list, optional
        Labels for break dates
    target_coef_idx : int
        Index of coefficient to highlight
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    dates = coefs['dates']
    mean = coefs['mean']
    lower_16 = coefs['lower_16']
    upper_84 = coefs['upper_84']
    coef_names = coefs['coef_names']

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: Target coefficient with breaks
    ax1 = axes[0]
    ax1.fill_between(dates, lower_16[:, target_coef_idx], upper_84[:, target_coef_idx],
                     alpha=0.3, color='blue')
    ax1.plot(dates, mean[:, target_coef_idx], color='blue', linewidth=2,
             label=coef_names[target_coef_idx])
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Add break dates with labels
    for i, bd in enumerate(break_dates):
        bd_ts = pd.Timestamp(bd)
        label = break_labels[i] if break_labels and i < len(break_labels) else bd
        ax1.axvline(bd_ts, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ypos = ax1.get_ylim()[1] * 0.9
        ax1.annotate(label, xy=(bd_ts, ypos), rotation=45, fontsize=8,
                    ha='left', va='bottom')

    ax1.set_ylabel('Coefficient Value')
    ax1.set_title(f'Time-Varying {coef_names[target_coef_idx]} vs Structural Breaks')
    ax1.legend(loc='lower right')

    # Bottom: Coefficient change magnitude
    ax2 = axes[1]
    coef_diff = np.abs(np.diff(mean[:, target_coef_idx]))
    ax2.fill_between(dates[1:], 0, coef_diff, alpha=0.5, color='green')
    ax2.set_ylabel('|Change|')
    ax2.set_title('Coefficient Change Magnitude')

    # Add break dates
    for bd in break_dates:
        bd_ts = pd.Timestamp(bd)
        ax2.axvline(bd_ts, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_volatility_paths(
    volatility: Dict[str, np.ndarray],
    var_names: List[str],
    break_dates: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot stochastic volatility paths.

    Parameters
    ----------
    volatility : dict
        Output from model.get_volatility_path()
    var_names : list
        Variable names
    break_dates : list, optional
        Structural break dates to overlay
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    dates = volatility['dates']
    mean = volatility['mean']
    lower_16 = volatility['lower_16']
    upper_84 = volatility['upper_84']

    k = mean.shape[1]
    fig, axes = plt.subplots(k, 1, figsize=figsize, sharex=True)

    if k == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        vname = var_names[i] if i < len(var_names) else f'Variable {i}'

        ax.fill_between(dates, lower_16[:, i], upper_84[:, i],
                       alpha=0.3, color='orange')
        ax.plot(dates, mean[:, i], color='orange', linewidth=1.5)

        if break_dates:
            for bd in break_dates:
                ax.axvline(pd.Timestamp(bd), color='red', linestyle='--', alpha=0.7)

        ax.set_ylabel(f'$\\sigma({vname})$')
        ax.set_title(f'Stochastic Volatility: {vname}')

    axes[-1].xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_forecast_fan(
    forecast: Dict[str, np.ndarray],
    actuals: Optional[pd.Series] = None,
    history: Optional[pd.Series] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None,
    var_name: str = "Target",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot forecast fan chart.

    Parameters
    ----------
    forecast : dict
        Output from model.forecast()
    actuals : pd.Series, optional
        Actual values for comparison
    history : pd.Series, optional
        Historical values
    forecast_dates : pd.DatetimeIndex, optional
        Dates for forecast horizon
    var_name : str
        Variable name for title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    h = forecast['mean'].shape[0]

    if forecast_dates is None:
        if history is not None:
            last_date = history.index[-1]
            forecast_dates = pd.date_range(last_date, periods=h + 1, freq='MS')[1:]
        else:
            forecast_dates = pd.date_range('2020-01-01', periods=h, freq='MS')

    # Plot history
    if history is not None:
        ax.plot(history.index, history.values, color='black', linewidth=1.5, label='Historical')

    # Target variable (first column)
    target_idx = 0
    mean = forecast['mean'][:, target_idx]
    lower_10 = forecast['lower_10'][:, target_idx]
    upper_90 = forecast['upper_90'][:, target_idx]
    lower_5 = forecast['lower_5'][:, target_idx]
    upper_95 = forecast['upper_95'][:, target_idx]

    # Fan chart
    ax.fill_between(forecast_dates, lower_5, upper_95, alpha=0.2, color='blue', label='90% CI')
    ax.fill_between(forecast_dates, lower_10, upper_90, alpha=0.3, color='blue', label='80% CI')
    ax.plot(forecast_dates, mean, color='blue', linewidth=2, label='Forecast Mean')

    # Actuals if available
    if actuals is not None:
        ax.plot(actuals.index, actuals.values, color='red', linewidth=1.5,
                linestyle='--', label='Actual')

    ax.set_xlabel('Date')
    ax.set_ylabel(var_name)
    ax.set_title(f'TVP-VAR Forecast: {var_name}')
    ax.legend(loc='best')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_rolling_forecast_metrics(
    actuals: np.ndarray,
    forecasts: np.ndarray
) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    actuals : np.ndarray
        Actual values
    forecasts : np.ndarray
        Forecast means

    Returns
    -------
    dict
        MAE, RMSE, MAPE, directional accuracy
    """
    errors = actuals - forecasts
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / np.abs(actuals + 1e-8)) * 100

    # Directional accuracy
    if len(actuals) > 1:
        actual_dir = np.sign(np.diff(actuals))
        forecast_dir = np.sign(np.diff(forecasts))
        dir_acc = np.mean(actual_dir == forecast_dir) * 100
    else:
        dir_acc = np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': dir_acc
    }


def rolling_backtest(
    df: pd.DataFrame,
    model_class,
    system_vars: List[str],
    target_var: str,
    train_start: str,
    test_start: str,
    test_end: str,
    h: int = 1,
    step: int = 1,
    n_lags: int = 1,
    n_draws: int = 1000,
    n_burn: int = 500,
    stochastic_vol: bool = False,
    fast_mode: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform rolling window backtest for TVP-VAR.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with date index
    model_class : class
        TVP_VAR class
    system_vars : list
        Variables in VAR system
    target_var : str
        Target variable name
    train_start : str
        Training start date
    test_start : str
        Test period start
    test_end : str
        Test period end
    h : int
        Forecast horizon
    step : int
        Rolling step size
    n_lags : int
        VAR lags
    n_draws : int
        MCMC draws (reduced for speed)
    n_burn : int
        Burn-in
    stochastic_vol : bool
        Enable SV
    verbose : bool
        Print progress

    Returns
    -------
    dict
        'forecasts': DataFrame of forecasts
        'actuals': Actual values
        'metrics': Accuracy metrics
    """
    train_start_dt = pd.Timestamp(train_start)
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)

    # Get test dates
    test_dates = df.loc[test_start_dt:test_end_dt].index[::step]

    forecasts = []
    actuals = []

    target_idx = system_vars.index(target_var)

    for i, forecast_origin in enumerate(test_dates):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Rolling backtest: {i + 1}/{len(test_dates)}")

        # Training data up to forecast origin
        train_df = df.loc[train_start_dt:forecast_origin]

        if len(train_df) < 50:
            continue

        # Fit model
        try:
            Y = train_df[system_vars].values
            dates = train_df.index

            model = model_class(
                n_lags=n_lags,
                stochastic_vol=stochastic_vol,
                n_draws=n_draws,
                n_burn=n_burn,
                fast_mode=fast_mode
            )
            model.fit(Y, dates=dates, var_names=system_vars, verbose=False)

            # Generate forecast
            fc = model.forecast(h=h)

            # Get actual value
            forecast_date = forecast_origin + pd.DateOffset(months=h)
            if forecast_date in df.index:
                actual = df.loc[forecast_date, target_var]
                forecast_mean = fc['mean'][h - 1, target_idx]
                forecast_lower = fc['lower_10'][h - 1, target_idx]
                forecast_upper = fc['upper_90'][h - 1, target_idx]

                forecasts.append({
                    'origin': forecast_origin,
                    'target': forecast_date,
                    'forecast_mean': forecast_mean,
                    'forecast_lower': forecast_lower,
                    'forecast_upper': forecast_upper,
                    'actual': actual
                })
                actuals.append(actual)

        except Exception as e:
            if verbose:
                print(f"    Error at {forecast_origin}: {e}")
            continue

    # Compile results
    if not forecasts:
        return {'error': 'No valid forecasts generated'}

    fc_df = pd.DataFrame(forecasts)
    fc_df.set_index('target', inplace=True)

    # Compute metrics
    metrics = compute_rolling_forecast_metrics(
        fc_df['actual'].values,
        fc_df['forecast_mean'].values
    )

    # Coverage probability
    in_interval = ((fc_df['actual'] >= fc_df['forecast_lower']) &
                   (fc_df['actual'] <= fc_df['forecast_upper']))
    metrics['coverage_80'] = in_interval.mean() * 100

    return {
        'forecasts': fc_df,
        'metrics': metrics,
        'n_forecasts': len(fc_df)
    }


def create_comparison_summary(
    tvp_results: Dict[str, Any],
    break_dates: List[str],
    segment_stats: List[Dict],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create summary comparing TVP dynamics with structural breaks.

    Parameters
    ----------
    tvp_results : dict
        Contains 'coefs' from get_time_varying_coefficients
    break_dates : list
        Structural break dates
    segment_stats : list
        Segment statistics from Bai-Perron
    save_path : Path, optional
        Path to save CSV

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    coefs = tvp_results['coefs']
    dates = coefs['dates']
    mean = coefs['mean']
    coef_names = coefs['coef_names']

    rows = []

    # Add segments
    for i, stats in enumerate(segment_stats):
        start = pd.Timestamp(stats['start_date'])
        end = pd.Timestamp(stats['end_date'])

        # Get coefficient stats for this period
        mask = (dates >= start) & (dates <= end)
        period_coefs = mean[mask]

        row = {
            'segment': i + 1,
            'start_date': stats['start_date'],
            'end_date': stats['end_date'],
            'n_obs': stats['n_obs'],
            'reserves_mean': stats['mean'],
            'reserves_std': stats['std']
        }

        # Add coefficient summaries
        for j, name in enumerate(coef_names):
            if len(period_coefs) > 0:
                row[f'coef_{name}_mean'] = np.mean(period_coefs[:, j])
                row[f'coef_{name}_std'] = np.std(period_coefs[:, j])

        rows.append(row)

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


__all__ = [
    'plot_trace',
    'plot_convergence_diagnostics',
    'plot_coefficient_paths',
    'plot_tvp_vs_breaks',
    'plot_volatility_paths',
    'plot_forecast_fan',
    'compute_rolling_forecast_metrics',
    'rolling_backtest',
    'create_comparison_summary'
]

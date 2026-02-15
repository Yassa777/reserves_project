"""
Visualization functions for DMA/DMS results.

Provides publication-quality figures for:
- Time-varying weight evolution (stacked area)
- DMS model selection paths
- Alpha sensitivity analysis
- DMA vs individual model comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


# Color palette for models (colorblind-friendly)
MODEL_COLORS = {
    'ARIMA': '#1f77b4',      # Blue
    'VECM': '#ff7f0e',       # Orange
    'MS-VAR': '#2ca02c',     # Green
    'MS-VECM': '#d62728',    # Red
    'Naive': '#9467bd',      # Purple
    'BVAR': '#8c564b',       # Brown
    'TVP-VAR': '#e377c2',    # Pink
    'FAVAR': '#7f7f7f',      # Gray
    'TVAR': '#bcbd22',       # Olive
    'MIDAS': '#17becf',      # Cyan
    'EqualWeight': '#aec7e8',    # Light blue
    'MSE-Weight': '#ffbb78',     # Light orange
    'GR-Convex': '#98df8a',      # Light green
}


def get_model_color(model_name: str) -> str:
    """Get color for a model, with fallback for unknown models."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # Generate consistent color for unknown models
    hash_val = hash(model_name) % 10
    fallback_colors = plt.cm.tab10.colors
    return fallback_colors[hash_val]


def plot_dma_weights_stacked(
    weights_history: np.ndarray,
    model_names: List[str],
    dates: pd.DatetimeIndex,
    title: str = "DMA: Time-Varying Model Weights",
    figsize: Tuple[int, int] = (14, 6),
    crisis_periods: Optional[List[Tuple[str, str]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time-varying model weights as stacked area chart.

    Parameters
    ----------
    weights_history : np.array
        Weight matrix (T, K)
    model_names : list
        Names of models
    dates : pd.DatetimeIndex
        Date index
    title : str
        Plot title
    figsize : tuple
        Figure size
    crisis_periods : list, optional
        List of (start, end) tuples for crisis shading
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for each model
    colors = [get_model_color(m) for m in model_names]

    # Create stacked area plot
    ax.stackplot(
        dates,
        weights_history.T,
        labels=model_names,
        colors=colors,
        alpha=0.8
    )

    # Add crisis period shading
    if crisis_periods:
        for start, end in crisis_periods:
            ax.axvspan(
                pd.Timestamp(start),
                pd.Timestamp(end),
                alpha=0.15,
                color='red',
                zorder=0
            )

    ax.set_ylabel('Model Weight', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(dates[0], dates[-1])

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Legend outside plot
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_dms_selection_path(
    weights_history: np.ndarray,
    model_names: List[str],
    dates: pd.DatetimeIndex,
    title: str = "DMS: Model Selection Over Time",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot which model was selected at each time point (DMS).

    Parameters
    ----------
    weights_history : np.array
        Weight matrix (T, K)
    model_names : list
        Names of models
    dates : pd.DatetimeIndex
        Date index
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Find selected model at each time
    selected = np.argmax(weights_history, axis=1)
    colors = [get_model_color(m) for m in model_names]

    # Plot scatter for each model
    for k, m in enumerate(model_names):
        mask = selected == k
        if mask.any():
            ax.scatter(
                dates[mask],
                np.ones(mask.sum()) * k,
                c=colors[k],
                s=30,
                label=m,
                alpha=0.8
            )

    ax.set_ylabel('Selected Model', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlim(dates[0], dates[-1])

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)

    # Add horizontal lines for clarity
    for k in range(len(model_names)):
        ax.axhline(y=k, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_weight_evolution_by_model(
    weights_history: np.ndarray,
    model_names: List[str],
    dates: pd.DatetimeIndex,
    model_subset: Optional[List[str]] = None,
    title: str = "Weight Evolution by Model",
    figsize: Tuple[int, int] = (12, None),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Individual weight paths for each model as subplots.

    Parameters
    ----------
    weights_history : np.array
        Weight matrix (T, K)
    model_names : list
        Names of models
    dates : pd.DatetimeIndex
        Date index
    model_subset : list, optional
        Subset of models to plot
    title : str
        Overall title
    figsize : tuple
        Figure size (height auto-calculated if None)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    if model_subset is None:
        model_subset = model_names

    n_plots = len(model_subset)
    fig_height = figsize[1] if figsize[1] else 2.5 * n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], fig_height), sharex=True)

    if n_plots == 1:
        axes = [axes]

    for ax, model in zip(axes, model_subset):
        idx = model_names.index(model)
        color = get_model_color(model)

        ax.plot(dates, weights_history[:, idx], color=color, linewidth=1.5)
        ax.fill_between(dates, 0, weights_history[:, idx], color=color, alpha=0.3)
        ax.set_ylabel(f'{model}', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(dates[0], dates[-1])

        # Add mean weight annotation
        mean_w = np.mean(weights_history[:, idx])
        ax.axhline(y=mean_w, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(
            dates[-1], mean_w,
            f' {mean_w:.2f}',
            va='center',
            fontsize=9,
            color='gray'
        )

    axes[-1].set_xlabel('Date', fontsize=11)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_alpha_sensitivity(
    alpha_results: pd.DataFrame,
    metric: str = 'rmse',
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance sensitivity to forgetting factor alpha.

    Parameters
    ----------
    alpha_results : pd.DataFrame
        Results from grid search with columns: alpha, method, rmse, mae
    metric : str
        Metric to plot ('rmse' or 'mae')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method in ['dma', 'dms']:
        data = alpha_results[alpha_results['method'] == method]
        ax.plot(
            data['alpha'],
            data[metric],
            marker='o',
            markersize=8,
            linewidth=2,
            label=method.upper()
        )

    ax.set_xlabel('Forgetting Factor (alpha)', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} Sensitivity to Forgetting Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark optimal alpha
    best_row = alpha_results.loc[alpha_results[metric].idxmin()]
    ax.axvline(
        x=best_row['alpha'],
        color='red',
        linestyle='--',
        alpha=0.5,
        label=f"Optimal: {best_row['alpha']}"
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_dma_vs_individual(
    dma_forecasts: np.ndarray,
    model_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str = "DMA vs Individual Model Forecasts",
    figsize: Tuple[int, int] = (14, 7),
    top_n_models: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare DMA forecasts with top individual models.

    Parameters
    ----------
    dma_forecasts : np.array
        DMA combined forecasts
    model_forecasts : dict
        Individual model forecasts
    actuals : np.array
        Realized values
    dates : pd.DatetimeIndex
        Date index
    title : str
        Plot title
    figsize : tuple
        Figure size
    top_n_models : int
        Number of individual models to show
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # Top panel: forecasts vs actual
    ax1 = axes[0]
    ax1.plot(dates, actuals, 'k-', linewidth=2, label='Actual', alpha=0.9)
    ax1.plot(dates, dma_forecasts, 'r-', linewidth=2, label='DMA', alpha=0.8)

    # Select top models by lowest RMSE
    model_rmse = {}
    for m, fc in model_forecasts.items():
        valid = ~np.isnan(fc) & ~np.isnan(actuals)
        if valid.sum() > 0:
            model_rmse[m] = np.sqrt(np.mean((fc[valid] - actuals[valid]) ** 2))

    top_models = sorted(model_rmse, key=model_rmse.get)[:top_n_models]

    for m in top_models:
        color = get_model_color(m)
        ax1.plot(dates, model_forecasts[m], linestyle='--', color=color, linewidth=1, label=m, alpha=0.7)

    ax1.set_ylabel('Reserves (USD millions)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Bottom panel: DMA forecast errors
    ax2 = axes[1]
    dma_errors = dma_forecasts - actuals
    colors = ['green' if e >= 0 else 'red' for e in dma_errors]
    ax2.bar(dates, dma_errors, color=colors, alpha=0.6, width=25)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('DMA Error', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_selection_frequency(
    selection_freq: pd.Series,
    title: str = "DMS: Model Selection Frequency",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart of model selection frequencies.

    Parameters
    ----------
    selection_freq : pd.Series
        Frequency of each model being selected
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [get_model_color(m) for m in selection_freq.index]
    bars = ax.bar(
        range(len(selection_freq)),
        selection_freq.values * 100,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_xticks(range(len(selection_freq)))
    ax.set_xticklabels(selection_freq.index, rotation=45, ha='right')
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add percentage labels on bars
    for bar, pct in zip(bars, selection_freq.values):
        height = bar.get_height()
        ax.annotate(
            f'{pct*100:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_ylim(0, max(selection_freq.values * 100) * 1.15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_weight_heatmap(
    weights_history: np.ndarray,
    model_names: List[str],
    dates: pd.DatetimeIndex,
    resample_freq: str = 'Q',
    title: str = "Model Weights Over Time",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Heatmap of model weights over time.

    Parameters
    ----------
    weights_history : np.array
        Weight matrix (T, K)
    model_names : list
        Names of models
    dates : pd.DatetimeIndex
        Date index
    resample_freq : str
        Frequency for resampling ('M', 'Q', 'Y')
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Create DataFrame and resample
    df = pd.DataFrame(weights_history, index=dates, columns=model_names)
    df_resampled = df.resample(resample_freq).mean()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        df_resampled.T,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=1
    )

    # Set axis labels
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # X-axis: dates
    n_ticks = min(12, len(df_resampled))
    tick_positions = np.linspace(0, len(df_resampled) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [df_resampled.index[i].strftime('%Y-%m') for i in tick_positions],
        rotation=45,
        ha='right'
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Weight')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_performance_comparison(
    metrics_df: pd.DataFrame,
    metric: str = 'rmse',
    title: str = "Forecast Performance Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing DMA/DMS with individual models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics with columns: model, rmse, mae, mape
    metric : str
        Metric to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric
    df_sorted = metrics_df.sort_values(metric)

    # Highlight DMA/DMS
    colors = []
    for m in df_sorted['model']:
        if m in ['DMA', 'DMS']:
            colors.append('#d62728')  # Red for DMA/DMS
        elif m == 'EqualWeight':
            colors.append('#ff7f0e')  # Orange for equal weight
        else:
            colors.append('#1f77b4')  # Blue for individual models

    bars = ax.barh(
        range(len(df_sorted)),
        df_sorted[metric],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['model'])
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, df_sorted[metric]):
        width = bar.get_width()
        ax.annotate(
            f'{val:.1f}',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=9
        )

    # Add legend
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='DMA/DMS'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Equal Weight'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Individual Models'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_dma_report_figures(
    dma_results,
    dms_results,
    model_forecasts: Dict[str, np.ndarray],
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    alpha_search_results: pd.DataFrame,
    output_dir: str,
    crisis_periods: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, str]:
    """
    Create all DMA report figures and save to output directory.

    Parameters
    ----------
    dma_results : DMAResults
        Results from DMA
    dms_results : DMAResults
        Results from DMS
    model_forecasts : dict
        Individual model forecasts
    actuals : np.array
        Realized values
    dates : pd.DatetimeIndex
        Date index
    alpha_search_results : pd.DataFrame
        Alpha grid search results
    output_dir : str
        Directory to save figures
    crisis_periods : list, optional
        Crisis period date ranges

    Returns
    -------
    figure_paths : dict
        Mapping of figure name to saved path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_paths = {}

    # 1. DMA Weight Evolution (stacked)
    fig_path = output_path / "dma_weight_evolution.png"
    plot_dma_weights_stacked(
        dma_results.weights_history,
        dma_results.model_names,
        dates,
        title="DMA: Time-Varying Model Weights",
        crisis_periods=crisis_periods,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['weight_evolution'] = str(fig_path)

    # 2. DMS Selection Path
    fig_path = output_path / "dms_selection_path.png"
    plot_dms_selection_path(
        dms_results.weights_history,
        dms_results.model_names,
        dates,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['selection_path'] = str(fig_path)

    # 3. Weight Evolution by Model
    fig_path = output_path / "weight_by_model.png"
    plot_weight_evolution_by_model(
        dma_results.weights_history,
        dma_results.model_names,
        dates,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['weight_by_model'] = str(fig_path)

    # 4. Alpha Sensitivity
    fig_path = output_path / "alpha_sensitivity.png"
    plot_alpha_sensitivity(
        alpha_search_results,
        metric='rmse',
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['alpha_sensitivity'] = str(fig_path)

    # 5. DMA vs Individual
    fig_path = output_path / "dma_vs_individual.png"
    plot_dma_vs_individual(
        dma_results.combined_forecasts,
        model_forecasts,
        actuals,
        dates,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['dma_vs_individual'] = str(fig_path)

    # 6. Selection Frequency
    selection_freq = dms_results.get_selection_frequency()
    fig_path = output_path / "selection_frequency.png"
    plot_selection_frequency(
        selection_freq,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['selection_frequency'] = str(fig_path)

    # 7. Weight Heatmap
    fig_path = output_path / "weight_heatmap.png"
    plot_weight_heatmap(
        dma_results.weights_history,
        dma_results.model_names,
        dates,
        save_path=str(fig_path)
    )
    plt.close()
    figure_paths['weight_heatmap'] = str(fig_path)

    return figure_paths

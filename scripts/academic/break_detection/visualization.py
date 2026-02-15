"""
Visualization Functions for Structural Break Analysis

Provides plotting utilities for Bai-Perron breaks, CUSUM tests,
and regime comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings


def setup_plot_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_series_with_breaks(
    series: pd.Series,
    break_dates: List[str],
    event_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None,
    show_regime_means: bool = True,
    confidence_intervals: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot time series with structural break dates marked.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    break_dates : list
        List of break date strings
    event_names : list, optional
        Names for each break event
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show_regime_means : bool
        Whether to show horizontal lines for regime means
    confidence_intervals : dict, optional
        Confidence intervals for break dates

    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the series
    ax.plot(series.index, series.values, 'b-', linewidth=1.5, alpha=0.8, label=series.name or 'Series')

    # Convert break dates to timestamps
    break_timestamps = [pd.Timestamp(d) for d in break_dates]

    # Define colors for different regimes
    colors = plt.cm.Set2(np.linspace(0, 1, len(break_dates) + 1))

    # Add shaded regions for regimes
    all_dates = [series.index[0]] + break_timestamps + [series.index[-1]]

    for i in range(len(all_dates) - 1):
        start = all_dates[i]
        end = all_dates[i + 1]
        ax.axvspan(start, end, alpha=0.15, color=colors[i], zorder=0)

        # Compute and show regime mean
        if show_regime_means:
            mask = (series.index >= start) & (series.index < end)
            if mask.any():
                regime_mean = series[mask].mean()
                ax.hlines(regime_mean, start, end, colors=colors[i],
                         linestyles='--', linewidth=2, alpha=0.7)

    # Add vertical lines for breaks
    for i, break_date in enumerate(break_timestamps):
        label = event_names[i] if event_names and i < len(event_names) else f"Break {i+1}"
        ax.axvline(break_date, color='red', linestyle='--', linewidth=2, alpha=0.8)

        # Add label
        y_pos = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(label, xy=(break_date, y_pos), xytext=(5, 0),
                   textcoords='offset points', fontsize=9, rotation=90,
                   va='top', ha='left', color='red', fontweight='bold')

        # Add confidence interval shading if provided
        if confidence_intervals and i in confidence_intervals:
            lower, upper = confidence_intervals[i]
            lower_date = series.index[lower] if lower < len(series) else series.index[-1]
            upper_date = series.index[upper] if upper < len(series) else series.index[-1]
            ax.axvspan(lower_date, upper_date, alpha=0.2, color='red', zorder=1)

    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel or series.name or 'Value')
    ax.set_title(title or f'{series.name or "Series"} with Structural Breaks')

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # Add legend
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_bic_selection(
    bic_values: Dict[int, float],
    lwz_values: Optional[Dict[int, float]] = None,
    optimal_n: int = None,
    title: str = "Information Criterion by Number of Breaks",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot BIC/LWZ values for break number selection.

    Parameters
    ----------
    bic_values : dict
        BIC values keyed by number of breaks
    lwz_values : dict, optional
        LWZ values keyed by number of breaks
    optimal_n : int, optional
        Optimal number of breaks to highlight
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by number of breaks
    n_breaks = sorted(bic_values.keys())
    bic_vals = [bic_values[n] for n in n_breaks]

    ax.plot(n_breaks, bic_vals, 'bo-', linewidth=2, markersize=8, label='BIC')

    if lwz_values:
        lwz_vals = [lwz_values.get(n, np.nan) for n in n_breaks]
        ax.plot(n_breaks, lwz_vals, 'gs--', linewidth=2, markersize=8, label='LWZ')

    # Highlight optimal
    if optimal_n is not None and optimal_n in bic_values:
        ax.scatter([optimal_n], [bic_values[optimal_n]], color='red', s=200,
                  zorder=5, marker='*', label=f'Optimal: {optimal_n} breaks')

    ax.set_xlabel('Number of Breaks')
    ax.set_ylabel('Information Criterion')
    ax.set_title(title)
    ax.set_xticks(n_breaks)
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_cusum(
    cusum_result: Dict[str, Any],
    dates: Optional[pd.DatetimeIndex] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot CUSUM test results with critical boundaries.

    Parameters
    ----------
    cusum_result : dict
        Results from cusum_test()
    dates : pd.DatetimeIndex, optional
        Dates for x-axis (if None, uses indices)
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    cusum = np.array(cusum_result.get("cusum", []))
    upper = np.array(cusum_result.get("upper_bound", []))
    lower = np.array(cusum_result.get("lower_bound", []))
    indices = cusum_result.get("indices", list(range(len(cusum))))

    if len(cusum) == 0:
        ax.text(0.5, 0.5, "No CUSUM data available", ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        return fig

    # Use dates if provided, otherwise use indices
    if dates is not None and len(indices) > 0:
        x_vals = [dates[i] for i in indices if i < len(dates)]
        if len(x_vals) < len(cusum):
            x_vals = indices
    else:
        x_vals = indices

    # Plot CUSUM path
    ax.plot(x_vals, cusum, 'b-', linewidth=2, label='CUSUM')

    # Plot critical boundaries
    ax.plot(x_vals, upper, 'r--', linewidth=1.5, label='Upper bound (5%)')
    ax.plot(x_vals, lower, 'r--', linewidth=1.5, label='Lower bound (5%)')

    # Fill between bounds
    ax.fill_between(x_vals, lower, upper, alpha=0.1, color='green')

    # Add zero line
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Mark crossing if any
    if cusum_result.get("first_crossing_idx") is not None:
        crossing_idx = cusum_result["first_crossing_idx"]
        if crossing_idx < len(x_vals):
            ax.axvline(x_vals[crossing_idx], color='red', linestyle=':',
                      linewidth=2, alpha=0.8, label='First crossing')

    # Formatting
    ax.set_xlabel('Observation' if dates is None else 'Date')
    ax.set_ylabel('CUSUM Statistic')

    stable = cusum_result.get("stable", None)
    stability_text = "STABLE" if stable else "UNSTABLE" if stable is not None else "Unknown"
    default_title = f"CUSUM Test - {stability_text}"
    ax.set_title(title or default_title)

    ax.legend(loc='upper left')

    if isinstance(x_vals[0], pd.Timestamp):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_cusumsq(
    cusumsq_result: Dict[str, Any],
    dates: Optional[pd.DatetimeIndex] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot CUSUMSQ test results.

    Parameters
    ----------
    cusumsq_result : dict
        Results from cusumsq_test()
    dates : pd.DatetimeIndex, optional
        Dates for x-axis
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    cusumsq = np.array(cusumsq_result.get("cusumsq", []))
    expected = np.array(cusumsq_result.get("expected_line", []))
    upper = np.array(cusumsq_result.get("upper_bound", []))
    lower = np.array(cusumsq_result.get("lower_bound", []))
    indices = cusumsq_result.get("indices", list(range(len(cusumsq))))

    if len(cusumsq) == 0:
        ax.text(0.5, 0.5, "No CUSUMSQ data available", ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        return fig

    # Use dates if provided
    if dates is not None and len(indices) > 0:
        x_vals = [dates[i] for i in indices if i < len(dates)]
        if len(x_vals) < len(cusumsq):
            x_vals = indices
    else:
        x_vals = indices

    # Plot CUSUMSQ path
    ax.plot(x_vals, cusumsq, 'b-', linewidth=2, label='CUSUMSQ')

    # Plot expected line (45-degree)
    ax.plot(x_vals, expected, 'k--', linewidth=1, alpha=0.5, label='Expected')

    # Plot bounds
    ax.plot(x_vals, upper, 'r--', linewidth=1.5, label='Bounds (5%)')
    ax.plot(x_vals, lower, 'r--', linewidth=1.5)

    # Fill between bounds
    ax.fill_between(x_vals, lower, upper, alpha=0.1, color='green')

    # Mark crossing if any
    if cusumsq_result.get("first_crossing_idx") is not None:
        crossing_idx = cusumsq_result["first_crossing_idx"]
        if crossing_idx < len(x_vals):
            ax.axvline(x_vals[crossing_idx], color='red', linestyle=':',
                      linewidth=2, alpha=0.8, label='First crossing')

    # Formatting
    ax.set_xlabel('Observation' if dates is None else 'Date')
    ax.set_ylabel('CUSUMSQ Statistic')
    ax.set_ylim(-0.1, 1.1)

    stable = cusumsq_result.get("stable", None)
    stability_text = "STABLE" if stable else "UNSTABLE" if stable is not None else "Unknown"
    default_title = f"CUSUM of Squares Test - {stability_text}"
    ax.set_title(title or default_title)

    ax.legend(loc='upper left')

    if isinstance(x_vals[0], pd.Timestamp):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_regime_comparison(
    series: pd.Series,
    break_dates: List[str],
    event_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create boxplot comparison of regimes defined by structural breaks.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    break_dates : list
        List of break date strings
    event_names : list, optional
        Names for regimes
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Convert break dates
    break_timestamps = [pd.Timestamp(d) for d in break_dates]
    all_dates = [series.index[0]] + break_timestamps + [series.index[-1]]

    # Extract regime data
    regime_data = []
    regime_labels = []

    for i in range(len(all_dates) - 1):
        start = all_dates[i]
        end = all_dates[i + 1]
        mask = (series.index >= start) & (series.index < end)
        regime_data.append(series[mask].values)

        if event_names and i < len(event_names):
            label = f"Before {event_names[i]}" if i == 0 else event_names[i-1] if i > 0 else ""
        else:
            label = f"Regime {i+1}"

        # Add date range
        label += f"\n({start.strftime('%Y-%m')} to {end.strftime('%Y-%m')})"
        regime_labels.append(label)

    # Boxplot
    ax1 = axes[0]
    bp = ax1.boxplot(regime_data, labels=[f"R{i+1}" for i in range(len(regime_data))],
                     patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(regime_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel('Regime')
    ax1.set_ylabel(series.name or 'Value')
    ax1.set_title('Distribution by Regime')

    # Add regime labels as text
    for i, label in enumerate(regime_labels):
        ax1.annotate(label, xy=(i + 1, ax1.get_ylim()[0]),
                    xytext=(0, -30), textcoords='offset points',
                    ha='center', va='top', fontsize=8, rotation=0)

    # Summary statistics table
    ax2 = axes[1]
    ax2.axis('off')

    stats_data = []
    for i, data in enumerate(regime_data):
        stats_data.append([
            f"Regime {i+1}",
            f"{len(data)}",
            f"{np.mean(data):.1f}",
            f"{np.std(data):.1f}",
            f"{np.min(data):.1f}",
            f"{np.max(data):.1f}",
            f"{np.mean(data) / np.mean(regime_data[0]) * 100:.1f}%" if i > 0 else "100%"
        ])

    columns = ['Regime', 'N', 'Mean', 'Std', 'Min', 'Max', 'Rel. to R1']

    table = ax2.table(cellText=stats_data, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax2.set_title('Regime Statistics', pad=20)

    plt.suptitle(title or f'Regime Comparison: {series.name or "Series"}', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_all_breaks_summary(
    results: Dict[str, Dict],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create summary plot showing breaks across all variables.

    Parameters
    ----------
    results : dict
        Dictionary of results keyed by variable name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    n_vars = len(results)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)

    if n_vars == 1:
        axes = [axes]

    for ax, (var_name, var_results) in zip(axes, results.items()):
        # Get series data if available
        if "series" in var_results:
            series = var_results["series"]
            ax.plot(series.index, series.values, 'b-', linewidth=1, alpha=0.7)
            dates = series.index
        else:
            dates = None

        # Get break dates
        break_dates = var_results.get("break_dates", [])
        for bd in break_dates:
            ax.axvline(pd.Timestamp(bd), color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_ylabel(var_name, fontsize=10)

        # Add n_breaks annotation
        n_breaks = var_results.get("n_breaks", len(break_dates))
        ax.annotate(f"n={n_breaks}", xy=(0.02, 0.85), xycoords='axes fraction',
                   fontsize=9, fontweight='bold')

    axes[-1].set_xlabel('Date')
    axes[0].set_title('Structural Breaks Across Variables', fontsize=14)

    if dates is not None:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_chow_test_results(
    chow_results: Dict[str, Any],
    series: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Chow test results for multiple break dates.

    Parameters
    ----------
    chow_results : dict
        Results from multiple_chow_tests()
    series : pd.Series, optional
        Original series to plot
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()

    tests = chow_results.get("tests", [])
    n_tests = len(tests)

    if n_tests == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No Chow test results available", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Series with break dates
    ax1 = axes[0]
    if series is not None:
        ax1.plot(series.index, series.values, 'b-', linewidth=1.5, alpha=0.8)

        for test in tests:
            if test.get("valid", False):
                break_date = pd.Timestamp(test["break_date"])
                color = 'red' if test.get("reject_null", False) else 'gray'
                ax1.axvline(break_date, color=color, linestyle='--', linewidth=2, alpha=0.8)

                label = test.get("event_name", "")
                ax1.annotate(label, xy=(break_date, ax1.get_ylim()[1]),
                           xytext=(5, -5), textcoords='offset points',
                           fontsize=9, rotation=90, va='top', color=color)

        ax1.set_ylabel(series.name or 'Value')
        ax1.set_title('Series with Tested Break Dates (Red = Significant)')

    # Bottom panel: F-statistics bar chart
    ax2 = axes[1]

    events = []
    f_stats = []
    colors = []

    for test in tests:
        events.append(test.get("event_name", "Unknown"))
        f_stats.append(test.get("f_statistic", 0) if test.get("valid", False) else 0)
        colors.append('red' if test.get("reject_null", False) else 'steelblue')

    bars = ax2.bar(events, f_stats, color=colors, alpha=0.7)

    # Add p-value labels
    for bar, test in zip(bars, tests):
        if test.get("valid", False):
            p_val = test.get("p_value", 1.0)
            ax2.annotate(f'p={p_val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax2.set_ylabel('F-statistic')
    ax2.set_xlabel('Event')
    ax2.set_title('Chow Test F-Statistics by Event')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig

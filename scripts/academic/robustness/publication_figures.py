"""
Publication Figure Generation
==============================

Generates high-quality figures for the academic paper.

Figures:
- Forecast comparison (top models vs actual)
- DMA weight evolution with crisis shading
- Subsample performance bar chart
- Model ranking heatmap

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# Crisis periods for shading
CRISIS_PERIODS = [
    ('COVID-19', '2020-03-01', '2021-06-30', '#ffcccc'),
    ('Forex Crisis', '2021-07-01', '2022-04-30', '#ffe0cc'),
    ('Default', '2022-04-01', '2022-12-31', '#ffcccc'),
]


class FigureGenerator:
    """Generates publication-quality figures."""

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 300,
        figsize: Tuple[float, float] = (10, 6)
    ):
        """
        Initialize figure generator.

        Parameters
        ----------
        output_dir : Path
            Output directory for figures
        dpi : int
            Resolution for saved figures
        figsize : tuple
            Default figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    def forecast_comparison(
        self,
        actual: pd.Series,
        forecasts: Dict[str, pd.Series],
        title: str = 'Forecast Comparison',
        top_n: int = 4,
        show_crisis: bool = True
    ) -> Path:
        """
        Create forecast comparison figure.

        Parameters
        ----------
        actual : pd.Series
            Actual values with datetime index
        forecasts : dict
            Model name -> forecast series
        title : str
            Figure title
        top_n : int
            Number of top models to show
        show_crisis : bool
            Whether to shade crisis periods

        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Add crisis shading
        if show_crisis:
            self._add_crisis_shading(ax, actual.index.min(), actual.index.max())

        # Plot actual
        ax.plot(actual.index, actual.values, 'k-', linewidth=2,
               label='Actual', zorder=10)

        # Colors for forecasts
        colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))

        # Rank models by RMSE if more than top_n
        if len(forecasts) > top_n:
            rmse = {}
            for name, fc in forecasts.items():
                aligned = fc.reindex(actual.index)
                valid = ~(actual.isna() | aligned.isna())
                if valid.sum() > 0:
                    rmse[name] = np.sqrt(np.mean((actual[valid] - aligned[valid])**2))
            sorted_models = sorted(rmse.keys(), key=lambda x: rmse[x])[:top_n]
        else:
            sorted_models = list(forecasts.keys())

        for i, model in enumerate(sorted_models):
            fc = forecasts[model]
            ax.plot(fc.index, fc.values, '--', linewidth=1.5,
                   color=colors[i], alpha=0.8, label=model)

        ax.set_xlabel('Date')
        ax.set_ylabel('Foreign Reserves (USD millions)')
        ax.set_title(title)

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Save
        filepath = self.output_dir / 'forecast_comparison.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'forecast_comparison.pdf',
                   bbox_inches='tight')
        plt.close()

        return filepath

    def dma_weight_evolution(
        self,
        weights_df: pd.DataFrame,
        date_col: str = 'date',
        top_n: int = 5,
        show_crisis: bool = True
    ) -> Path:
        """
        Create DMA weight evolution figure.

        Parameters
        ----------
        weights_df : pd.DataFrame
            DataFrame with date and weight columns
        date_col : str
            Name of date column
        top_n : int
            Number of top models to highlight
        show_crisis : bool
            Whether to shade crisis periods

        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get weight columns
        weight_cols = [c for c in weights_df.columns if c not in ['date', 'actual', 'forecast']]

        # Sort by average weight
        avg_weights = weights_df[weight_cols].mean().sort_values(ascending=False)
        top_models = avg_weights.head(top_n).index.tolist()

        # Ensure date is datetime
        df = weights_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        # Add crisis shading
        if show_crisis:
            self._add_crisis_shading(ax, df.index.min(), df.index.max())

        # Colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))

        # Stacked area plot for top models
        bottom = np.zeros(len(df))
        for i, model in enumerate(top_models):
            if model in df.columns:
                values = df[model].fillna(0).values
                ax.fill_between(df.index, bottom, bottom + values,
                               alpha=0.7, label=model, color=colors[i])
                bottom += values

        # Other models combined
        other_cols = [c for c in weight_cols if c not in top_models]
        if other_cols:
            other_sum = df[other_cols].sum(axis=1).fillna(0).values
            ax.fill_between(df.index, bottom, bottom + other_sum,
                           alpha=0.5, label='Other', color='gray')

        ax.set_xlabel('Date')
        ax.set_ylabel('DMA Weight')
        ax.set_title('Dynamic Model Averaging: Weight Evolution')
        ax.set_ylim(0, 1)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')

        filepath = self.output_dir / 'dma_weight_evolution.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'dma_weight_evolution.pdf',
                   bbox_inches='tight')
        plt.close()

        return filepath

    def subsample_bar_chart(
        self,
        subsample_results: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> Path:
        """
        Create subsample performance bar chart.

        Parameters
        ----------
        subsample_results : pd.DataFrame
            Subsample analysis results
        metric : str
            Metric to display

        Returns
        -------
        Path
            Path to saved figure
        """
        # Pivot data
        pivot = subsample_results.pivot(index='Model', columns='Period', values=metric)

        # Order periods
        period_order = ['Pre-Crisis', 'Crisis', 'COVID', 'Post-Default']
        available_periods = [p for p in period_order if p in pivot.columns]

        if not available_periods:
            available_periods = pivot.columns.tolist()

        pivot = pivot[available_periods]

        # Sort by mean RMSE
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar positions
        n_models = len(pivot)
        n_periods = len(available_periods)
        bar_width = 0.8 / n_periods
        x = np.arange(n_models)

        # Colors for periods
        colors = plt.cm.Set2(np.linspace(0, 1, n_periods))

        for i, period in enumerate(available_periods):
            offset = (i - n_periods/2 + 0.5) * bar_width
            ax.bar(x + offset, pivot[period].values, bar_width,
                  label=period, color=colors[i], edgecolor='white')

        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'Subsample Robustness: {metric} by Period')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend(title='Period', loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        filepath = self.output_dir / 'subsample_bar_chart.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'subsample_bar_chart.pdf',
                   bbox_inches='tight')
        plt.close()

        return filepath

    def model_ranking_heatmap(
        self,
        ranking_matrix: pd.DataFrame,
        title: str = 'Model Rankings'
    ) -> Path:
        """
        Create model ranking heatmap.

        Parameters
        ----------
        ranking_matrix : pd.DataFrame
            Rankings with models as rows, categories as columns
        title : str
            Figure title

        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Custom colormap (green=good, red=bad)
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        cmap = LinearSegmentedColormap.from_list('rank', colors)

        # Normalize to 0-1 based on number of models
        n_models = len(ranking_matrix)
        norm_matrix = (ranking_matrix - 1) / (n_models - 1)

        sns.heatmap(
            norm_matrix,
            annot=ranking_matrix.values,
            fmt='.0f',
            cmap=cmap,
            center=0.5,
            ax=ax,
            cbar_kws={'label': 'Rank (1=Best)'},
            linewidths=0.5,
            linecolor='white'
        )

        ax.set_title(title)
        ax.set_ylabel('Model')
        ax.set_xlabel('Dimension')

        filepath = self.output_dir / 'ranking_heatmap.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'ranking_heatmap.pdf',
                   bbox_inches='tight')
        plt.close()

        return filepath

    def horizon_deterioration(
        self,
        horizon_results: pd.DataFrame,
        metric: str = 'RMSE',
        top_n: int = 6
    ) -> Path:
        """
        Plot forecast deterioration across horizons.

        Parameters
        ----------
        horizon_results : pd.DataFrame
            Horizon analysis results
        metric : str
            Metric to plot
        top_n : int
            Number of models to show

        Returns
        -------
        Path
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get unique models
        models = horizon_results['Model'].unique()

        # Rank by h=1 performance
        h1_data = horizon_results[horizon_results['Horizon'] == 1]
        if len(h1_data) > 0:
            sorted_models = h1_data.sort_values(metric)['Model'].tolist()[:top_n]
        else:
            sorted_models = list(models)[:top_n]

        colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_models)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

        for i, model in enumerate(sorted_models):
            model_data = horizon_results[horizon_results['Model'] == model].sort_values('Horizon')
            ax.plot(model_data['Horizon'], model_data[metric],
                   marker=markers[i % len(markers)], linewidth=2,
                   color=colors[i], label=model, markersize=8)

        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel(f'{metric}')
        ax.set_title(f'Forecast Accuracy by Horizon')
        ax.set_xticks([1, 3, 6, 12])
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / 'horizon_deterioration.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(self.output_dir / 'horizon_deterioration.pdf',
                   bbox_inches='tight')
        plt.close()

        return filepath

    def _add_crisis_shading(
        self,
        ax,
        start_date,
        end_date
    ):
        """Add crisis period shading to axis."""
        for name, crisis_start, crisis_end, color in CRISIS_PERIODS:
            crisis_start = pd.Timestamp(crisis_start)
            crisis_end = pd.Timestamp(crisis_end)

            # Only shade if within plot range
            if crisis_end >= start_date and crisis_start <= end_date:
                ax.axvspan(
                    max(crisis_start, start_date),
                    min(crisis_end, end_date),
                    alpha=0.3, color=color, label=name
                )


def generate_all_figures(
    actual: pd.Series,
    forecasts: Dict[str, pd.Series],
    weights_df: pd.DataFrame,
    subsample_results: pd.DataFrame,
    horizon_results: pd.DataFrame,
    ranking_matrix: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Generate all publication figures.

    Returns
    -------
    dict
        Mapping of figure name to file path
    """
    generator = FigureGenerator(output_dir)
    figures = {}

    # Forecast comparison
    if actual is not None and forecasts:
        figures['forecast_comparison'] = generator.forecast_comparison(
            actual, forecasts,
            title='Forecast Comparison: Top Models vs Actual',
            top_n=4
        )

    # DMA weights
    if weights_df is not None and len(weights_df) > 0:
        figures['dma_weights'] = generator.dma_weight_evolution(
            weights_df, top_n=5
        )

    # Subsample bar chart
    if subsample_results is not None and len(subsample_results) > 0:
        figures['subsample'] = generator.subsample_bar_chart(subsample_results)

    # Ranking heatmap
    if ranking_matrix is not None and len(ranking_matrix) > 0:
        figures['ranking_heatmap'] = generator.model_ranking_heatmap(ranking_matrix)

    # Horizon deterioration
    if horizon_results is not None and len(horizon_results) > 0:
        figures['horizon'] = generator.horizon_deterioration(horizon_results)

    return figures

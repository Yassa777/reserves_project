"""
Appendix Tables for Academic Paper
====================================

Generates appendix tables A1-A6 with detailed results.

Tables:
- A1: Full DM test matrix
- A2: Individual model metrics by period
- A3: BVAR specification comparison
- A4: DMA weight summary
- A5: Density forecast evaluation
- A6: Rolling window stability

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class AppendixTableGenerator:
    """Generates appendix LaTeX tables."""

    def __init__(self, output_dir: Path):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_number(self, val, decimals=2, bold=False, stars=''):
        """Format number for LaTeX."""
        if pd.isna(val):
            return '--'
        s = f'{val:.{decimals}f}'
        if bold:
            s = f'\\textbf{{{s}}}'
        return s + stars

    def generate_table_a1_full_dm(
        self,
        dm_stats: pd.DataFrame,
        dm_pvalues: pd.DataFrame
    ) -> str:
        """Generate full DM test matrix (Table A1)."""
        latex = []

        models = dm_stats.columns.tolist()
        n = len(models)

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\footnotesize')
        latex.append(r'\caption{Full Diebold-Mariano Test Matrix}')
        latex.append(r'\label{tab:dm_full}')
        latex.append(r'\begin{tabular}{l' + 'c' * n + '}')
        latex.append(r'\toprule')

        # Abbreviate column headers
        abbrev = {name: name[:6] for name in models}
        header = ' & '.join([abbrev[m] for m in models])
        latex.append(f' & {header} \\\\')
        latex.append(r'\midrule')

        for row in models:
            row_abbrev = abbrev[row]
            cells = [row_abbrev]

            for col in models:
                if row == col:
                    cells.append('--')
                else:
                    stat = dm_stats.loc[row, col]
                    pval = dm_pvalues.loc[row, col]

                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.10:
                        stars = '*'
                    else:
                        stars = ''

                    cells.append(self.format_number(stat, 1, stars=stars))

            latex.append(' & '.join(cells) + ' \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\footnotesize')
        latex.append(r'\item Notes: DM statistics with HLN correction. ')
        latex.append(r'Positive = row model beats column. ')
        latex.append(r'*** p<0.01, ** p<0.05, * p<0.10.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table_a2_period_metrics(
        self,
        subsample_results: pd.DataFrame
    ) -> str:
        """Generate individual model metrics by period (Table A2)."""
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\footnotesize')
        latex.append(r'\caption{Forecast Accuracy by Sub-Period}')
        latex.append(r'\label{tab:period_metrics}')

        # Pivot for each metric
        for metric in ['RMSE', 'MAE']:
            if metric not in subsample_results.columns:
                continue

            pivot = subsample_results.pivot(
                index='Model', columns='Period', values=metric
            )

            periods = pivot.columns.tolist()
            n_periods = len(periods)

            latex.append(f'\\textbf{{{metric}:}}')
            latex.append(r'\begin{tabular}{l' + 'c' * n_periods + '}')
            latex.append(r'\toprule')
            latex.append('Model & ' + ' & '.join(periods) + ' \\\\')
            latex.append(r'\midrule')

            for model in pivot.index:
                cells = [model[:12]]
                for period in periods:
                    cells.append(self.format_number(pivot.loc[model, period], 1))
                latex.append(' & '.join(cells) + ' \\\\')

            latex.append(r'\bottomrule')
            latex.append(r'\end{tabular}')
            latex.append('')
            latex.append(r'\vspace{0.3cm}')

        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table_a3_bvar_specs(
        self,
        varset_results: pd.DataFrame
    ) -> str:
        """Generate BVAR specification comparison (Table A3)."""
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\small')
        latex.append(r'\caption{BVAR Specification Comparison}')
        latex.append(r'\label{tab:bvar_specs}')
        latex.append(r'\begin{tabular}{lcccc}')
        latex.append(r'\toprule')
        latex.append(r'Variable Set & N Obs & RMSE & MAE & MAPE (\%) \\')
        latex.append(r'\midrule')

        # Sort by RMSE
        df = varset_results.sort_values('RMSE')

        best_rmse = df['RMSE'].min()

        for _, row in df.iterrows():
            varset = row.get('Variable_Set', row.get('Variable_Set_Key', ''))
            n_obs = int(row.get('N_Obs', 0))
            rmse = self.format_number(row['RMSE'], 1, bold=(row['RMSE'] == best_rmse))
            mae = self.format_number(row['MAE'], 1)
            mape = self.format_number(row.get('MAPE', np.nan), 2)

            latex.append(f'{varset} & {n_obs} & {rmse} & {mae} & {mape} \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: All specifications use h=1 month horizon, test period only.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table_a4_dma_weights(
        self,
        weight_summary: pd.DataFrame
    ) -> str:
        """Generate DMA weight summary (Table A4)."""
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\small')
        latex.append(r'\caption{DMA Weight Summary Statistics}')
        latex.append(r'\label{tab:dma_weights}')
        latex.append(r'\begin{tabular}{lcccc}')
        latex.append(r'\toprule')
        latex.append(r'Model & Mean Weight & Std Dev & Max & Period of Max \\')
        latex.append(r'\midrule')

        for _, row in weight_summary.iterrows():
            model = row.get('Model', '')
            mean_w = self.format_number(row.get('Mean_Weight', np.nan), 3)
            std_w = self.format_number(row.get('Std_Weight', np.nan), 3)
            max_w = self.format_number(row.get('Max_Weight', np.nan), 3)
            period = row.get('Period_of_Max', '--')

            latex.append(f'{model} & {mean_w} & {std_w} & {max_w} & {period} \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table_a5_density(
        self,
        density_results: pd.DataFrame
    ) -> str:
        """Generate density forecast evaluation (Table A5)."""
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\small')
        latex.append(r'\caption{Density Forecast Evaluation}')
        latex.append(r'\label{tab:density}')
        latex.append(r'\begin{tabular}{lccccc}')
        latex.append(r'\toprule')
        latex.append(r'Model & CRPS & Log Score & Coverage 90\% & Coverage 95\% & PIT Uniform \\')
        latex.append(r'\midrule')

        for _, row in density_results.iterrows():
            model = row.get('Model', '')
            crps = self.format_number(row.get('Mean_CRPS', np.nan), 3)
            ls = self.format_number(row.get('Mean_LogScore', np.nan), 3)
            cov90 = self.format_number(row.get('Coverage_90', np.nan) * 100, 1) if pd.notna(row.get('Coverage_90')) else '--'
            cov95 = self.format_number(row.get('Coverage_95', np.nan) * 100, 1) if pd.notna(row.get('Coverage_95')) else '--'
            pit = 'Yes' if row.get('PIT_Chi2_pvalue', 0) > 0.05 else 'No'

            latex.append(f'{model} & {crps} & {ls} & {cov90} & {cov95} & {pit} \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: CRPS = Continuous Ranked Probability Score (lower is better). ')
        latex.append(r'Log Score = average log predictive density (higher is better).')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table_a6_stability(
        self,
        stability_results: pd.DataFrame
    ) -> str:
        """Generate rolling window stability (Table A6)."""
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(r'\small')
        latex.append(r'\caption{Model Ranking Stability}')
        latex.append(r'\label{tab:stability}')
        latex.append(r'\begin{tabular}{lcccc}')
        latex.append(r'\toprule')
        latex.append(r'Model & Mean Rank & Rank Std & Best Rank & Worst Rank \\')
        latex.append(r'\midrule')

        for _, row in stability_results.iterrows():
            model = row.get('Model', '')
            mean_r = self.format_number(row.get('Mean_Rank', np.nan), 1)
            std_r = self.format_number(row.get('Std_Rank', np.nan), 2)
            min_r = int(row.get('Min_Rank', 0)) if pd.notna(row.get('Min_Rank')) else '--'
            max_r = int(row.get('Max_Rank', 0)) if pd.notna(row.get('Max_Rank')) else '--'

            latex.append(f'{model} & {mean_r} & {std_r} & {min_r} & {max_r} \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: Rankings computed across all robustness dimensions ')
        latex.append(r'(periods, horizons, variable sets).')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def save_table(self, latex: str, filename: str) -> Path:
        """Save LaTeX table to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex)
        return filepath


def generate_all_appendix_tables(
    dm_stats: pd.DataFrame,
    dm_pvalues: pd.DataFrame,
    subsample_results: pd.DataFrame,
    varset_results: pd.DataFrame,
    density_results: pd.DataFrame,
    stability_results: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Path]:
    """Generate all appendix tables."""
    generator = AppendixTableGenerator(output_dir)
    tables = {}

    # A1: Full DM matrix
    if dm_stats is not None and dm_pvalues is not None:
        latex = generator.generate_table_a1_full_dm(dm_stats, dm_pvalues)
        tables['a1_dm_full'] = generator.save_table(latex, 'table_a1_dm_full.tex')

    # A2: Period metrics
    if subsample_results is not None and len(subsample_results) > 0:
        latex = generator.generate_table_a2_period_metrics(subsample_results)
        tables['a2_period_metrics'] = generator.save_table(latex, 'table_a2_period_metrics.tex')

    # A3: BVAR specs
    if varset_results is not None and len(varset_results) > 0:
        latex = generator.generate_table_a3_bvar_specs(varset_results)
        tables['a3_bvar_specs'] = generator.save_table(latex, 'table_a3_bvar_specs.tex')

    # A5: Density evaluation
    if density_results is not None and len(density_results) > 0:
        latex = generator.generate_table_a5_density(density_results)
        tables['a5_density'] = generator.save_table(latex, 'table_a5_density.tex')

    # A6: Stability
    if stability_results is not None and len(stability_results) > 0:
        latex = generator.generate_table_a6_stability(stability_results)
        tables['a6_stability'] = generator.save_table(latex, 'table_a6_stability.tex')

    return tables

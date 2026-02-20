"""
LaTeX Table Generation
=======================

Generates publication-ready LaTeX tables for the academic paper.

Tables:
- Table 1: Main Accuracy Comparison
- Table 2: Diebold-Mariano Test Summary
- Table 3: Model Confidence Set Membership
- Table 4: Subsample Robustness
- Table 5: Horizon Robustness
- Table 6: Variable Set Robustness
- Appendix Tables A1-A6

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path


class LaTeXTableGenerator:
    """Generates publication-quality LaTeX tables."""

    def __init__(
        self,
        output_dir: Path,
        table_style: str = 'booktabs',
        font_size: str = 'small'
    ):
        """
        Initialize table generator.

        Parameters
        ----------
        output_dir : Path
            Directory for output files
        table_style : str
            LaTeX table style ('booktabs', 'standard')
        font_size : str
            Font size for tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_style = table_style
        self.font_size = font_size

    def format_number(
        self,
        value: float,
        decimals: int = 2,
        bold: bool = False,
        stars: str = ''
    ) -> str:
        """Format a number for LaTeX output."""
        if pd.isna(value):
            return '--'

        formatted = f'{value:.{decimals}f}'

        if bold:
            formatted = f'\\textbf{{{formatted}}}'

        if stars:
            formatted = f'{formatted}{stars}'

        return formatted

    def generate_table1_accuracy(
        self,
        mcs_summary: pd.DataFrame,
        dm_formatted: pd.DataFrame
    ) -> str:
        """
        Generate Table 1: Main Forecast Accuracy Comparison.

        Parameters
        ----------
        mcs_summary : pd.DataFrame
            MCS summary with RMSE, MAE, and membership
        dm_formatted : pd.DataFrame
            Formatted DM test matrix

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        # Header
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Out-of-Sample Forecast Accuracy}')
        latex.append(r'\label{tab:accuracy}')
        include_policy = 'Policy_Loss' in mcs_summary.columns
        tabular_cols = 'lcccccc' if include_policy else 'lccccc'
        latex.append(rf'\begin{{tabular}}{{{tabular_cols}}}')
        latex.append(r'\toprule')
        if include_policy:
            latex.append(r'Model & RMSE & MAE & MAPE (\%) & Policy Loss & MCS & Rank \\')
        else:
            latex.append(r'Model & RMSE & MAE & MAPE (\%) & MCS & Rank \\')
        latex.append(r'\midrule')

        # Sort by rank
        if 'Rank' in mcs_summary.columns:
            df = mcs_summary.sort_values('Rank')
        else:
            df = mcs_summary.sort_values('RMSE')

        # Find best values for bolding
        best_rmse = df['RMSE'].min()
        best_mae = df['MAE'].min()
        best_mape = df['MAPE'].min() if 'MAPE' in df.columns else np.nan
        best_policy = df['Policy_Loss'].min() if include_policy else np.nan

        for _, row in df.iterrows():
            model = row['Model']

            # Bold best values
            rmse_str = self.format_number(
                row['RMSE'], decimals=1,
                bold=(row['RMSE'] == best_rmse)
            )
            mae_str = self.format_number(
                row['MAE'], decimals=1,
                bold=(row['MAE'] == best_mae)
            )

            if 'MAPE' in row and not pd.isna(row.get('MAPE')):
                mape_str = self.format_number(
                    row['MAPE'], decimals=2,
                    bold=(row['MAPE'] == best_mape)
                )
            else:
                mape_str = '--'

            if include_policy and 'Policy_Loss' in row and not pd.isna(row.get('Policy_Loss')):
                policy_str = self.format_number(
                    row['Policy_Loss'], decimals=1,
                    bold=(row['Policy_Loss'] == best_policy)
                )
            else:
                policy_str = '--'

            # MCS indicator
            if 'In_MCS' in row:
                mcs_str = r'$\checkmark$' if row['In_MCS'] else ''
            else:
                mcs_str = ''

            # Rank
            rank_str = str(int(row['Rank'])) if 'Rank' in row else '--'

            if include_policy:
                latex.append(
                    f'{model} & {rmse_str} & {mae_str} & {mape_str} & {policy_str} & {mcs_str} & {rank_str} \\\\'
                )
            else:
                latex.append(f'{model} & {rmse_str} & {mae_str} & {mape_str} & {mcs_str} & {rank_str} \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: RMSE = Root Mean Squared Error; MAE = Mean Absolute Error; ')
        if include_policy:
            latex.append(
                r'MAPE = Mean Absolute Percentage Error; Policy Loss = asymmetric loss penalizing reserve shortfalls; '
            )
        else:
            latex.append(r'MAPE = Mean Absolute Percentage Error; ')
        latex.append(r'MCS = 90\% Model Confidence Set membership. ')
        latex.append(r'Bold indicates best performance. Test period: 2023--2025.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table2_dm(
        self,
        dm_stats: pd.DataFrame,
        dm_pvalues: pd.DataFrame,
        benchmark: str = 'Naive'
    ) -> str:
        """
        Generate Table 2: Diebold-Mariano Test Summary.

        Parameters
        ----------
        dm_stats : pd.DataFrame
            DM test statistics matrix
        dm_pvalues : pd.DataFrame
            DM p-values matrix
        benchmark : str
            Benchmark model for comparison

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Diebold-Mariano Test Results}')
        latex.append(r'\label{tab:dm_tests}')

        # Extract benchmark column
        if benchmark in dm_stats.columns:
            models = [m for m in dm_stats.index if m != benchmark]

            latex.append(r'\begin{tabular}{lccc}')
            latex.append(r'\toprule')
            latex.append(f'Model & DM Statistic & p-value & Significance \\\\')
            latex.append(r'\midrule')

            for model in models:
                stat = dm_stats.loc[model, benchmark]
                pval = dm_pvalues.loc[model, benchmark]

                # Determine significance
                if pval < 0.01:
                    sig = '***'
                elif pval < 0.05:
                    sig = '**'
                elif pval < 0.10:
                    sig = '*'
                else:
                    sig = ''

                stat_str = self.format_number(stat, decimals=2)
                pval_str = self.format_number(pval, decimals=3)

                latex.append(f'{model} & {stat_str} & {pval_str} & {sig} \\\\')

            latex.append(r'\bottomrule')
            latex.append(r'\end{tabular}')
        else:
            # Full matrix
            latex.append(self._generate_dm_matrix(dm_stats, dm_pvalues))

        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: Diebold-Mariano test statistics with HAC standard errors. ')
        latex.append(r'Positive values indicate the row model outperforms the column model. ')
        latex.append(r'*** p < 0.01; ** p < 0.05; * p < 0.10.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def _generate_dm_matrix(
        self,
        dm_stats: pd.DataFrame,
        dm_pvalues: pd.DataFrame
    ) -> str:
        """Generate full DM matrix table body."""
        models = dm_stats.columns.tolist()
        n_models = len(models)

        # Abbreviated model names for compact display
        abbrev = {
            'EqualWeight': 'EqWt',
            'MSE-Weight': 'MSE-Wt',
            'GR-Convex': 'GR-C',
            'TrimmedMean': 'Trim',
        }

        cols = ' & '.join([abbrev.get(m, m[:6]) for m in models])
        lines = [
            r'\begin{tabular}{l' + 'c' * n_models + '}',
            r'\toprule',
            f'& {cols} \\\\',
            r'\midrule'
        ]

        for row_model in models:
            row_name = abbrev.get(row_model, row_model[:6])
            cells = [row_name]

            for col_model in models:
                if row_model == col_model:
                    cells.append('--')
                else:
                    stat = dm_stats.loc[row_model, col_model]
                    pval = dm_pvalues.loc[row_model, col_model]

                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.10:
                        stars = '*'
                    else:
                        stars = ''

                    cell = self.format_number(stat, decimals=1, stars=stars)
                    cells.append(cell)

            lines.append(' & '.join(cells) + ' \\\\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        return '\n'.join(lines)

    def generate_table3_mcs(
        self,
        mcs_results: Dict,
        mcs_summary: pd.DataFrame
    ) -> str:
        """
        Generate Table 3: Model Confidence Set Membership.

        Parameters
        ----------
        mcs_results : dict
            MCS results with mcs, eliminated, p_values
        mcs_summary : pd.DataFrame
            Summary with RMSE and ranks

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Model Confidence Set Results}')
        latex.append(r'\label{tab:mcs}')
        latex.append(r'\begin{tabular}{lccc}')
        latex.append(r'\toprule')
        latex.append(r'Model & RMSE & MCS p-value & Status \\')
        latex.append(r'\midrule')

        # Models in MCS
        latex.append(r'\multicolumn{4}{l}{\textit{Models in 90\% MCS:}} \\')

        for model in mcs_results.get('mcs', []):
            if model in mcs_summary['Model'].values:
                row = mcs_summary[mcs_summary['Model'] == model].iloc[0]
                rmse = self.format_number(row['RMSE'], decimals=1)
                latex.append(f'{model} & {rmse} & -- & Included \\\\')

        latex.append(r'\midrule')
        latex.append(r'\multicolumn{4}{l}{\textit{Eliminated models (order of elimination):}} \\')

        # Eliminated models
        eliminated = mcs_results.get('eliminated', [])
        p_values = mcs_results.get('p_values', [])

        for i, model in enumerate(eliminated):
            if model in mcs_summary['Model'].values:
                row = mcs_summary[mcs_summary['Model'] == model].iloc[0]
                rmse = self.format_number(row['RMSE'], decimals=1)
                pval = self.format_number(p_values[i], decimals=3) if i < len(p_values) else '--'
                latex.append(f'{model} & {rmse} & {pval} & Eliminated ({i+1}) \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(f'\\item Notes: MCS procedure with $\\alpha = {mcs_results.get("alpha", 0.10)}$. ')
        latex.append(r'Bootstrap replications: 1,000. Models in the MCS are not statistically ')
        latex.append(r'distinguishable from the best model at the 10\% level.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table4_subsample(
        self,
        subsample_results: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> str:
        """
        Generate Table 4: Subsample Robustness.

        Parameters
        ----------
        subsample_results : pd.DataFrame
            Subsample analysis results
        metric : str
            Metric to display

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        # Pivot to models x periods
        pivot = subsample_results.pivot(index='Model', columns='Period', values=metric)

        # Order periods
        period_order = ['Pre-Crisis', 'Crisis', 'COVID', 'Post-Default', 'Full Sample']
        available_periods = [p for p in period_order if p in pivot.columns]
        pivot = pivot[available_periods]

        # Add average rank
        ranks = pivot.rank(axis=0)
        pivot['Avg. Rank'] = ranks.mean(axis=1)
        pivot = pivot.sort_values('Avg. Rank')

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Subsample Robustness: RMSE by Period}')
        latex.append(r'\label{tab:subsample}')

        n_cols = len(available_periods) + 2  # model + periods + avg rank
        latex.append(r'\begin{tabular}{l' + 'c' * (n_cols - 1) + '}')
        latex.append(r'\toprule')

        header = 'Model & ' + ' & '.join(available_periods) + ' & Avg. Rank \\\\'
        latex.append(header)
        latex.append(r'\midrule')

        # Find best in each period
        best_values = {col: pivot[col].min() for col in available_periods}

        for model in pivot.index:
            row_data = [model]

            for period in available_periods:
                val = pivot.loc[model, period]
                is_best = val == best_values[period]
                row_data.append(self.format_number(val, decimals=1, bold=is_best))

            row_data.append(self.format_number(pivot.loc[model, 'Avg. Rank'], decimals=1))
            latex.append(' & '.join(row_data) + ' \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: Pre-Crisis: 2012--2018; Crisis: 2019--2022; ')
        latex.append(r'COVID: 2020--2021; Post-Default: 2023--2025. ')
        latex.append(r'Bold indicates best performance in each period.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table5_horizon(
        self,
        horizon_results: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> str:
        """
        Generate Table 5: Horizon Robustness.

        Parameters
        ----------
        horizon_results : pd.DataFrame
            Horizon analysis results
        metric : str
            Metric to display

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        # Pivot to models x horizons
        pivot = horizon_results.pivot(index='Model', columns='Horizon', values=metric)

        # Order horizons
        horizons = sorted([h for h in pivot.columns if isinstance(h, (int, float))])
        pivot = pivot[horizons]

        # Add deterioration ratio (h12/h1)
        if 1 in horizons and 12 in horizons:
            pivot['h12/h1'] = pivot[12] / pivot[1]

        # Sort by h1 performance
        if 1 in horizons:
            pivot = pivot.sort_values(1)

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Forecast Horizon Robustness: RMSE by Horizon}')
        latex.append(r'\label{tab:horizon}')

        n_cols = len(horizons) + 2  # model + horizons + ratio
        latex.append(r'\begin{tabular}{l' + 'c' * (n_cols - 1) + '}')
        latex.append(r'\toprule')

        horizon_labels = [f'h={h}' for h in horizons]
        header = 'Model & ' + ' & '.join(horizon_labels) + ' & $h_{12}/h_1$ \\\\'
        latex.append(header)
        latex.append(r'\midrule')

        # Find best in each horizon
        best_values = {h: pivot[h].min() for h in horizons}

        for model in pivot.index:
            row_data = [model]

            for h in horizons:
                val = pivot.loc[model, h]
                is_best = val == best_values[h]
                row_data.append(self.format_number(val, decimals=1, bold=is_best))

            if 'h12/h1' in pivot.columns:
                row_data.append(self.format_number(pivot.loc[model, 'h12/h1'], decimals=2))
            else:
                row_data.append('--')

            latex.append(' & '.join(row_data) + ' \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: Horizons in months. $h_{12}/h_1$ shows the deterioration ')
        latex.append(r'ratio as forecast horizon increases from 1 to 12 months. ')
        latex.append(r'Bold indicates best performance at each horizon.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def generate_table6_varset(
        self,
        varset_results: pd.DataFrame,
        metric: str = 'RMSE'
    ) -> str:
        """
        Generate Table 6: Variable Set Robustness.

        Parameters
        ----------
        varset_results : pd.DataFrame
            Variable set analysis results (models x varsets)
        metric : str
            Metric (for caption)

        Returns
        -------
        str
            LaTeX table code
        """
        latex = []

        # varset_results should already be pivoted
        if 'Model' in varset_results.columns:
            pivot = varset_results.set_index('Model')
        else:
            pivot = varset_results.copy()

        varsets = pivot.columns.tolist()

        # Add sensitivity measure (CV)
        pivot['CV'] = pivot.std(axis=1) / pivot.mean(axis=1)
        pivot = pivot.sort_values('CV')

        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(f'\\{self.font_size}')
        latex.append(r'\caption{Variable Set Robustness: RMSE by Specification}')
        latex.append(r'\label{tab:varset}')

        n_cols = len(varsets) + 2
        latex.append(r'\begin{tabular}{l' + 'c' * (n_cols - 1) + '}')
        latex.append(r'\toprule')

        header = 'Model & ' + ' & '.join(varsets) + ' & CV \\\\'
        latex.append(header)
        latex.append(r'\midrule')

        # Find best in each varset
        best_values = {v: pivot[v].min() for v in varsets if v != 'CV'}

        for model in pivot.index:
            row_data = [model]

            for varset in varsets:
                val = pivot.loc[model, varset]
                is_best = val == best_values.get(varset, np.inf)
                row_data.append(self.format_number(val, decimals=1, bold=is_best))

            row_data.append(self.format_number(pivot.loc[model, 'CV'], decimals=3))
            latex.append(' & '.join(row_data) + ' \\\\')

        latex.append(r'\bottomrule')
        latex.append(r'\end{tabular}')
        latex.append(r'\begin{tablenotes}')
        latex.append(r'\small')
        latex.append(r'\item Notes: CV = Coefficient of Variation (sensitivity to specification). ')
        latex.append(r'Lower CV indicates more stable performance across variable sets. ')
        latex.append(r'Bold indicates best performance for each specification.')
        latex.append(r'\end{tablenotes}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def save_table(self, latex_code: str, filename: str) -> Path:
        """Save LaTeX table to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex_code)
        return filepath


def generate_all_tables(
    mcs_summary: pd.DataFrame,
    mcs_results: Dict,
    dm_stats: pd.DataFrame,
    dm_pvalues: pd.DataFrame,
    subsample_results: pd.DataFrame,
    horizon_results: pd.DataFrame,
    varset_results: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Generate all LaTeX tables.

    Returns
    -------
    dict
        Mapping of table name to file path
    """
    generator = LaTeXTableGenerator(output_dir)

    tables = {}

    # Table 1: Main Accuracy
    if mcs_summary is not None and dm_stats is not None:
        latex = generator.generate_table1_accuracy(mcs_summary, dm_stats)
        tables['table1_accuracy'] = generator.save_table(latex, 'table1_accuracy.tex')

    # Table 2: DM Tests
    if dm_stats is not None and dm_pvalues is not None:
        latex = generator.generate_table2_dm(dm_stats, dm_pvalues)
        tables['table2_dm'] = generator.save_table(latex, 'table2_dm_tests.tex')

    # Table 3: MCS
    if mcs_results is not None and mcs_summary is not None:
        latex = generator.generate_table3_mcs(mcs_results, mcs_summary)
        tables['table3_mcs'] = generator.save_table(latex, 'table3_mcs.tex')

    # Table 4: Subsample
    if subsample_results is not None and len(subsample_results) > 0:
        latex = generator.generate_table4_subsample(subsample_results)
        tables['table4_subsample'] = generator.save_table(latex, 'table4_subsample.tex')

    # Table 5: Horizon
    if horizon_results is not None and len(horizon_results) > 0:
        latex = generator.generate_table5_horizon(horizon_results)
        tables['table5_horizon'] = generator.save_table(latex, 'table5_horizon.tex')

    # Table 6: Variable Set
    if varset_results is not None and len(varset_results) > 0:
        latex = generator.generate_table6_varset(varset_results)
        tables['table6_varset'] = generator.save_table(latex, 'table6_varset.tex')

    return tables


def format_for_latex(value: float, decimals: int = 2) -> str:
    """Simple formatting helper."""
    if pd.isna(value):
        return '--'
    return f'{value:.{decimals}f}'

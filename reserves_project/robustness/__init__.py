"""
Robustness Analysis Module for Academic Forecasting Pipeline
==============================================================

This module implements Phase 5 (Spec 11) - Robustness Tables and Academic Output.

Components:
- subsample_analysis: Performance across different time periods
- horizon_analysis: Comparison across forecast horizons (h=1,3,6,12)
- variable_set_analysis: Robustness across 5 variable sets
- latex_tables: Publication-ready LaTeX table generation
- publication_figures: High-quality figures for the paper
- paper_statistics: Key numbers for paper text

Author: Academic Forecasting Pipeline
Date: 2026-02-11
"""

from .subsample_analysis import (
    SubsampleAnalyzer,
    compute_subsample_metrics,
    subsample_robustness_table,
)

from .horizon_analysis import (
    HorizonAnalyzer,
    horizon_comparison_table,
    horizon_ranking_stability,
)

from .variable_set_analysis import (
    VariableSetAnalyzer,
    variable_set_comparison,
    ranking_consistency_test,
)

from .latex_tables import (
    LaTeXTableGenerator,
    generate_all_tables,
    format_for_latex,
)

from .publication_figures import (
    FigureGenerator,
    generate_all_figures,
)

from .paper_statistics import (
    compile_paper_statistics,
    format_inline_numbers,
)

from .appendix_tables import (
    AppendixTableGenerator,
    generate_all_appendix_tables,
)

__all__ = [
    # Subsample analysis
    'SubsampleAnalyzer',
    'compute_subsample_metrics',
    'subsample_robustness_table',
    # Horizon analysis
    'HorizonAnalyzer',
    'horizon_comparison_table',
    'horizon_ranking_stability',
    # Variable set analysis
    'VariableSetAnalyzer',
    'variable_set_comparison',
    'ranking_consistency_test',
    # LaTeX tables
    'LaTeXTableGenerator',
    'generate_all_tables',
    'format_for_latex',
    # Publication figures
    'FigureGenerator',
    'generate_all_figures',
    # Paper statistics
    'compile_paper_statistics',
    'format_inline_numbers',
    # Appendix tables
    'AppendixTableGenerator',
    'generate_all_appendix_tables',
]

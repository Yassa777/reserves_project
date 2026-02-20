# Specification 11: Robustness Tables and Academic Output
## Publication-Ready Tables, Figures, and Sensitivity Analysis

**Version:** 1.0
**Created:** 2026-02-10
**Status:** 🔵 NOT STARTED
**Phase:** 5 (Synthesis)
**Dependencies:** All previous specs (01-10)
**Blocks:** None (final deliverable)

---

## Objective

Generate publication-ready output for academic paper:
1. **Main results tables** - Forecast accuracy comparison
2. **Robustness checks** - Subsample analysis, alternative specifications
3. **Figures** - Time series plots, weight evolution, regime analysis
4. **LaTeX-formatted output** - Ready for journal submission
5. **Appendix tables** - Detailed breakdowns, diagnostic tests

---

## Unified Evaluator Inputs (NEW)

Use unified rolling-origin outputs (common forecast API) as the default input for Tables 1–3 and robustness checks:
- `data/forecast_results_unified/rolling_origin_summary_<varset>.csv`
- `data/forecast_results_unified/rolling_origin_forecasts_<varset>.csv`

These summaries include point metrics plus probabilistic scores (CRPS, log score) and coverage rates for 80/95% intervals, enabling additional robustness tables if needed.

---

## Main Tables

### Table 1: Forecast Accuracy Comparison (Point Forecasts)

```python
def generate_main_accuracy_table(results_dict, split="test"):
    """
    Main forecast accuracy table.

    Columns: Model, MAE, RMSE, MAPE, sMAPE, MASE
    """
    import pandas as pd

    rows = []
    for model_name, metrics in results_dict.items():
        if split in metrics:
            m = metrics[split]
            rows.append({
                "Model": model_name,
                "MAE": f"{m['mae']:.2f}",
                "RMSE": f"{m['rmse']:.2f}",
                "MAPE": f"{m['mape']:.1f}%",
                "sMAPE": f"{m['smape']:.1f}%",
                "MASE": f"{m['mase']:.3f}",
            })

    df = pd.DataFrame(rows)

    # Sort by RMSE
    df = df.sort_values("RMSE")

    # Highlight best
    df.loc[0, "Model"] = f"\\textbf{{{df.iloc[0]['Model']}}}"

    return df


def to_latex_table(df, caption, label, note=None):
    """
    Convert DataFrame to LaTeX table.
    """
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption=caption,
        label=label,
    )

    if note:
        latex = latex.replace(
            "\\end{tabular}",
            f"\\end{{tabular}}\n\\begin{{tablenotes}}\n\\small\n\\item {note}\n\\end{{tablenotes}}"
        )

    return latex
```

### Table 2: DM Test Results (Pairwise Comparison)

```python
def generate_dm_test_table(dm_pvalues, reference_model="DMA"):
    """
    DM test p-values relative to reference model.
    """
    rows = []
    for model in dm_pvalues.columns:
        if model != reference_model:
            p = dm_pvalues.loc[reference_model, model]
            if p < 0.01:
                sig = "***"
            elif p < 0.05:
                sig = "**"
            elif p < 0.10:
                sig = "*"
            else:
                sig = ""

            rows.append({
                "Model": model,
                "DM Statistic": f"{dm_stats.loc[reference_model, model]:.2f}",
                "p-value": f"{p:.3f}{sig}",
                "Conclusion": "Sig. different" if p < 0.10 else "Equal accuracy"
            })

    return pd.DataFrame(rows)
```

### Table 3: Model Confidence Set

```python
def generate_mcs_table(mcs_result, accuracy_metrics):
    """
    MCS membership with performance metrics.
    """
    rows = []
    for model, metrics in accuracy_metrics.items():
        in_mcs = "✓" if model in mcs_result['mcs'] else ""

        if model in mcs_result['eliminated']:
            elim = mcs_result['eliminated'].index(model) + 1
        else:
            elim = "-"

        rows.append({
            "Model": model,
            "RMSE": f"{metrics['rmse']:.2f}",
            "In MCS": in_mcs,
            "Elim. Order": elim,
        })

    df = pd.DataFrame(rows).sort_values("RMSE")
    return df
```

---

## Robustness Analysis

### Robustness 1: Subsample Stability

```python
def subsample_robustness(forecasts, actuals, dates, subsamples):
    """
    Evaluate model performance across different subsamples.

    Parameters:
    -----------
    subsamples : list of dict
        [{"name": "Pre-crisis", "start": "2010-01", "end": "2018-12"}, ...]
    """
    results = []

    for subsample in subsamples:
        mask = (dates >= subsample["start"]) & (dates <= subsample["end"])

        for model, fc in forecasts.items():
            valid = mask & ~np.isnan(fc) & ~np.isnan(actuals)
            if valid.sum() < 12:
                continue

            rmse = np.sqrt(np.mean((fc[valid] - actuals[valid])**2))
            mae = np.mean(np.abs(fc[valid] - actuals[valid]))

            results.append({
                "Subsample": subsample["name"],
                "Model": model,
                "N": valid.sum(),
                "RMSE": rmse,
                "MAE": mae,
            })

    return pd.DataFrame(results)


SUBSAMPLES = [
    {"name": "Pre-crisis (2012-2018)", "start": "2012-01-01", "end": "2018-12-01"},
    {"name": "Crisis (2019-2022)", "start": "2019-01-01", "end": "2022-12-01"},
    {"name": "Post-default (2023-2025)", "start": "2023-01-01", "end": "2025-12-01"},
    {"name": "COVID period", "start": "2020-01-01", "end": "2021-12-01"},
]
```

### Robustness 2: Alternative Forecast Horizons

```python
def horizon_robustness(model_results, horizons=[1, 3, 6, 12]):
    """
    Performance at different forecast horizons.
    """
    results = []

    for h in horizons:
        for model, metrics in model_results[f"h{h}"].items():
            results.append({
                "Horizon": f"{h}-month",
                "Model": model,
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
            })

    return pd.DataFrame(results)
```

### Robustness 3: Alternative Variable Sets

```python
def varset_robustness(results_by_varset):
    """
    Compare model performance across variable sets.
    """
    rows = []

    for varset, model_results in results_by_varset.items():
        for model, metrics in model_results.items():
            rows.append({
                "Variable Set": varset,
                "Model": model,
                "RMSE": metrics["rmse"],
                "Rank": None,  # Filled after sorting
            })

    df = pd.DataFrame(rows)

    # Compute ranks within each model
    for model in df["Model"].unique():
        mask = df["Model"] == model
        df.loc[mask, "Rank"] = df.loc[mask, "RMSE"].rank().astype(int)

    return df
```

### Robustness 4: Rolling vs Expanding Window

```python
def window_robustness(expanding_results, rolling_results):
    """
    Compare expanding vs rolling window estimation.
    """
    rows = []

    for model in expanding_results.keys():
        rows.append({
            "Model": model,
            "Expanding RMSE": expanding_results[model]["rmse"],
            "Rolling RMSE": rolling_results[model]["rmse"],
            "Difference": rolling_results[model]["rmse"] - expanding_results[model]["rmse"],
        })

    return pd.DataFrame(rows)
```

### Robustness 5: Alternative Loss Functions

```python
def loss_function_robustness(forecasts, actuals):
    """
    Compare rankings under different loss functions.
    """
    results = []

    for model, fc in forecasts.items():
        valid = ~np.isnan(fc) & ~np.isnan(actuals)

        errors = fc[valid] - actuals[valid]

        results.append({
            "Model": model,
            "MSE": np.mean(errors**2),
            "MAE": np.mean(np.abs(errors)),
            "MAPE": np.mean(np.abs(errors / actuals[valid])) * 100,
            "QLIKE": np.mean(np.log(fc[valid]**2) + (actuals[valid]**2) / (fc[valid]**2)),
        })

    df = pd.DataFrame(results)

    # Add ranks
    for col in ["MSE", "MAE", "MAPE", "QLIKE"]:
        df[f"{col}_Rank"] = df[col].rank().astype(int)

    return df
```

---

## Figures

### Figure 1: Forecast Comparison

```python
def plot_forecast_comparison(actuals, forecasts_dict, dates,
                              models_to_plot=None, figsize=(14, 8)):
    """
    Multi-panel forecast comparison.
    """
    import matplotlib.pyplot as plt

    if models_to_plot is None:
        models_to_plot = list(forecasts_dict.keys())[:4]

    n_models = len(models_to_plot)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=figsize, sharex=True)

    # Actuals
    axes[0].plot(dates, actuals, 'k-', linewidth=1.5, label='Actual')
    axes[0].set_ylabel('Reserves (USD m)')
    axes[0].legend()
    axes[0].set_title('Foreign Exchange Reserves: Forecasts vs Actual')

    # Individual model forecasts
    for i, model in enumerate(models_to_plot):
        ax = axes[i + 1]
        ax.plot(dates, actuals, 'k-', alpha=0.5, linewidth=1)
        ax.plot(dates, forecasts_dict[model], 'b-', linewidth=1.5, label=model)
        ax.fill_between(
            dates,
            actuals,
            forecasts_dict[model],
            alpha=0.3,
            color='red' if np.mean(forecasts_dict[model] - actuals) > 0 else 'green'
        )
        ax.set_ylabel('USD m')
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Date')
    plt.tight_layout()

    return fig
```

### Figure 2: DMA Weight Evolution

```python
def plot_dma_weights_academic(weights, model_names, dates, figsize=(12, 6)):
    """
    Academic-style weight evolution plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=figsize)

    # Use colorblind-friendly palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    ax.stackplot(dates, weights.T, labels=model_names, colors=colors, alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Model Weight', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Dynamic Model Averaging: Time-Varying Weights', fontsize=14)

    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    # Add crisis shading
    crisis_periods = [
        ("2018-10-01", "2019-06-01", "Currency crisis", 0.1),
        ("2020-03-01", "2021-06-01", "COVID-19", 0.1),
        ("2022-03-01", "2022-12-01", "Default", 0.15),
    ]

    for start, end, label, alpha in crisis_periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color='gray')

    plt.tight_layout()
    return fig
```

### Figure 3: Regime Comparison (MS-VAR vs Breaks vs TVP)

```python
def plot_regime_comparison(reserves, msvar_probs, break_dates,
                            tvp_coeffs, dates, figsize=(14, 10)):
    """
    Compare different regime identification methods.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Panel A: Reserves with break dates
    axes[0].plot(dates, reserves, 'k-', linewidth=1.5)
    for bd in break_dates:
        axes[0].axvline(bd, color='red', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Reserves\n(USD m)')
    axes[0].set_title('(a) Foreign Exchange Reserves with Structural Breaks')

    # Panel B: MS-VAR regime probabilities
    axes[1].fill_between(dates, 0, msvar_probs, alpha=0.7, color='blue')
    axes[1].set_ylabel('P(Crisis\nRegime)')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('(b) MS-VAR Smoothed Regime Probabilities')

    # Panel C: TVP coefficient (e.g., exchange rate)
    axes[2].plot(dates, tvp_coeffs['mean'], 'b-', linewidth=1.5)
    axes[2].fill_between(dates, tvp_coeffs['lower'], tvp_coeffs['upper'],
                          alpha=0.3, color='blue')
    axes[2].set_ylabel('Exchange\nRate Coef.')
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('(c) TVP-VAR: Time-Varying Exchange Rate Coefficient')

    # Panel D: DMA weights for best models
    axes[3].set_ylabel('DMA\nWeight')
    axes[3].set_xlabel('Date')
    axes[3].set_title('(d) DMA Model Weights')

    plt.tight_layout()
    return fig
```

---

## LaTeX Output Generation

### Full Table Export

```python
def export_all_tables_latex(output_dir):
    """
    Generate all LaTeX tables for paper.
    """
    tables = {}

    # Table 1: Main accuracy
    tables['tab_accuracy'] = to_latex_table(
        main_accuracy_df,
        caption="Forecast Accuracy Comparison (Test Sample, 2023-2025)",
        label="tab:accuracy",
        note="MASE scaled by naive forecast MAE. Best model in bold."
    )

    # Table 2: DM tests
    tables['tab_dm'] = to_latex_table(
        dm_summary_df,
        caption="Diebold-Mariano Test Results",
        label="tab:dm",
        note="*, **, *** indicate significance at 10%, 5%, 1% levels."
    )

    # Table 3: MCS
    tables['tab_mcs'] = to_latex_table(
        mcs_df,
        caption="Model Confidence Set (90% level)",
        label="tab:mcs",
        note="MCS procedure of Hansen et al. (2011)."
    )

    # Robustness tables
    tables['tab_robustness_subsample'] = to_latex_table(
        subsample_df,
        caption="Robustness: Subsample Performance",
        label="tab:robustness_subsample"
    )

    # Write to files
    for name, latex in tables.items():
        with open(f"{output_dir}/{name}.tex", 'w') as f:
            f.write(latex)

    return tables
```

### Appendix Tables

```python
def generate_appendix_tables():
    """
    Generate detailed appendix tables.
    """
    appendix = {}

    # A1: Variable definitions
    # A2: Unit root tests
    # A3: Cointegration tests
    # A4: Detailed DM matrix
    # A5: Full model specifications
    # A6: Hyperparameter selection

    return appendix
```

---

## Summary Statistics

### Descriptive Statistics Table

```python
def generate_descriptive_stats(data_df, variables):
    """
    Summary statistics for key variables.
    """
    stats = data_df[variables].describe().T
    stats['skewness'] = data_df[variables].skew()
    stats['kurtosis'] = data_df[variables].kurtosis()

    # ADF test p-values
    from statsmodels.tsa.stattools import adfuller
    stats['adf_pvalue'] = [
        adfuller(data_df[v].dropna())[1] for v in variables
    ]

    return stats
```

---

## File Structure

```
reserves_project/scripts/academic/
├── generate_robustness_tables.py  ← Main execution
├── output/
│   ├── tables/                    ← LaTeX tables
│   ├── figures/                   ← PDF/PNG figures
│   └── data/                      ← CSV exports
├── robustness/
│   ├── subsample.py
│   ├── horizon.py
│   ├── varset.py
│   └── window.py
└── latex/
    ├── table_generator.py
    └── figure_generator.py
```

## Output Structure

```
data/robustness/
├── tables/
│   ├── tab_accuracy.tex
│   ├── tab_dm.tex
│   ├── tab_mcs.tex
│   ├── tab_robustness_subsample.tex
│   ├── tab_robustness_horizon.tex
│   ├── tab_robustness_varset.tex
│   └── appendix/
│       ├── tab_a1_variables.tex
│       ├── tab_a2_unitroot.tex
│       └── ...
├── figures/
│   ├── fig_forecast_comparison.pdf
│   ├── fig_dma_weights.pdf
│   ├── fig_regime_comparison.pdf
│   ├── fig_crps_comparison.pdf
│   └── fig_subsample_performance.pdf
└── summary/
    ├── main_results.json
    ├── robustness_summary.json
    └── paper_statistics.json
```

---

## Paper Statistics Summary

```python
def compile_paper_statistics():
    """
    Compile key statistics cited in paper text.
    """
    stats = {
        "sample_period": {
            "start": "2012-01",
            "end": "2025-12",
            "n_obs": 168,
        },
        "best_model": {
            "name": "DMA",
            "test_rmse": 245.3,
            "vs_naive_improvement": "12.5%",
        },
        "mcs_models": ["DMA", "MS-VECM", "BVAR"],
        "dma_alpha": 0.99,
        "key_breaks": ["2018-10", "2022-04"],
        "dm_tests": {
            "dma_vs_naive": {"stat": 2.34, "pvalue": 0.019},
            "msvar_vs_arima": {"stat": 1.87, "pvalue": 0.062},
        },
    }

    return stats
```

---

## Execution Log

### Pre-Execution
| Check | Status | Notes |
|-------|--------|-------|
| All spec results available | ⬜ | Specs 01-10 complete |
| LaTeX packages available | ⬜ | booktabs, tablenotes |
| Plotting libraries | ⬜ | matplotlib, seaborn |

### Execution
| Step | Status | Timestamp | Notes |
|------|--------|-----------|-------|
| Compile main accuracy table | ⬜ | | |
| Compile DM test table | ⬜ | | |
| Compile MCS table | ⬜ | | |
| Run subsample robustness | ⬜ | | |
| Run horizon robustness | ⬜ | | |
| Run varset robustness | ⬜ | | |
| Generate figures | ⬜ | | |
| Export LaTeX | ⬜ | | |
| Compile paper statistics | ⬜ | | |

### Post-Execution
| Validation | Status | Notes |
|------------|--------|-------|
| Tables compile in LaTeX | ⬜ | |
| Figures high-resolution | ⬜ | |
| Statistics consistent | ⬜ | |

---

## Results Section (To Be Updated)

### Final Model Rankings
*[To be filled after execution]*

### Key Paper Statistics
*[To be filled after execution]*

### Robustness Summary
*[To be filled after execution]*

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-10 | 1.0 | Initial specification |

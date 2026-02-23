# Jupyter Notebooks

Interactive notebooks demonstrating key results from the Sri Lanka Reserves Forecasting study.

## Notebooks

| Notebook | Description | Prerequisites |
|----------|-------------|---------------|
| **01_quick_start.ipynb** | Load data, explore reserves, run a simple MS-VAR forecast | None |
| **02_model_comparison.ipynb** | Compare all models, visualize MS-VAR dominance | Pre-computed results |
| **03_scenario_analysis.ipynb** | Policy scenario fan charts, stress testing | Pre-computed scenarios |

## Quick Start

```bash
# Install Jupyter
pip install jupyter

# Launch notebooks
cd reserves_project/notebooks
jupyter notebook
```

## Generating Results

If notebooks show "results not found", run:

```bash
# Generate model comparison results
slreserves evaluate --include-ms

# Generate scenario results
slreserves scenarios --scenarios all
```

## Data Requirements

Notebooks expect data in `../data/`:
- `merged/reserves_forecasting_panel.csv` - Main dataset
- `forecast_results_unified/` - Model evaluation results
- `scenario_analysis/` - Scenario forecast outputs

# Test Suite

Unit tests for the reserves_project package.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=reserves_project --cov-report=html

# Run excluding slow tests
pytest tests/ -m "not slow"
```

## Test Structure

| File | Module Tested | Tests |
|------|---------------|-------|
| `test_metrics.py` | `eval.metrics` | Forecast metrics (MAE, RMSE, MAPE, MASE, asymmetric loss) |
| `test_diebold_mariano.py` | `eval.diebold_mariano` | DM test, HLN correction, pairwise matrix |
| `test_scenarios.py` | `scenarios.definitions` | Scenario class, policy scenarios, shock multipliers |
| `test_varsets.py` | `config.varsets` | Variable set configuration, registry |
| `test_models.py` | `models.*` | Smoke tests for MS-VAR, BVAR |

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_time_series`: Simple time series for basic tests
- `sample_panel`: Panel dataset matching expected structure
- `forecast_data`: Actual/forecast pairs for metric testing
- `multi_model_forecasts`: Multiple model forecasts for pairwise tests

## Adding Tests

1. Create new test file: `tests/test_<module>.py`
2. Use pytest fixtures from `conftest.py`
3. Follow naming convention: `Test*` for classes, `test_*` for functions
4. Mark slow tests with `@pytest.mark.slow`

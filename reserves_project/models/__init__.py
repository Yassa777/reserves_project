"""Model adapters and model-specific implementations."""

from .bvar import BayesianVAR
from .ms_switching_var import MarkovSwitchingVAR
from .msvar_diagnostics import (
    smoothed_probabilities_df,
    transition_matrix_df,
    expected_durations_df,
    classification_certainty_df,
    fit_diagnostics_dict,
)
from .ml_models import create_lag_features

__all__ = [
    "BayesianVAR",
    "MarkovSwitchingVAR",
    "smoothed_probabilities_df",
    "transition_matrix_df",
    "expected_durations_df",
    "classification_certainty_df",
    "fit_diagnostics_dict",
    "create_lag_features",
]

"""Model adapters and model-specific implementations."""

from .bvar import BayesianVAR
from .ms_switching_var import MarkovSwitchingVAR

__all__ = ["BayesianVAR", "MarkovSwitchingVAR"]

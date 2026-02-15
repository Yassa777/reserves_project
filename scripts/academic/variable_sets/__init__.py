# Variable set definitions and builders
# Parsimonious, BoP, Monetary, PCA, Full

from .config import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    SOURCE_DATA_PATH,
    SUPPLEMENTARY_DATA_PATH,
    HISTORICAL_FX_PATH,
    OUTPUT_DIR,
    # Core config
    TARGET_VAR,
    TRAIN_END,
    VALID_END,
    MIN_OBS_ARIMA,
    MIN_OBS_VECM,
    MIN_OBS_VAR,
    MIN_OBS_PCA,
    MISSING_STRATEGY,
    # Variable sets
    VARIABLE_SETS,
    VARSET_ORDER,
    VARSET_PARSIMONIOUS,
    VARSET_BOP,
    VARSET_MONETARY,
    VARSET_PCA,
    VARSET_FULL,
    # Functions
    get_varset,
    get_all_varsets,
    get_output_dir,
    get_all_required_vars,
)

from .pca_builder import (
    build_pca_factors,
    interpret_loadings,
    generate_scree_data,
    kaiser_criterion,
    elbow_criterion,
)

from .validators import (
    validate_variable_set,
    validate_all_varsets,
    apply_missing_strategy,
    check_data_quality,
    validate_date_index,
    check_train_valid_test_split,
)

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "SOURCE_DATA_PATH",
    "SUPPLEMENTARY_DATA_PATH",
    "HISTORICAL_FX_PATH",
    "OUTPUT_DIR",
    # Core config
    "TARGET_VAR",
    "TRAIN_END",
    "VALID_END",
    "MIN_OBS_ARIMA",
    "MIN_OBS_VECM",
    "MIN_OBS_VAR",
    "MIN_OBS_PCA",
    "MISSING_STRATEGY",
    # Variable sets
    "VARIABLE_SETS",
    "VARSET_ORDER",
    "VARSET_PARSIMONIOUS",
    "VARSET_BOP",
    "VARSET_MONETARY",
    "VARSET_PCA",
    "VARSET_FULL",
    # Functions
    "get_varset",
    "get_all_varsets",
    "get_output_dir",
    "get_all_required_vars",
    # PCA
    "build_pca_factors",
    "interpret_loadings",
    "generate_scree_data",
    "kaiser_criterion",
    "elbow_criterion",
    # Validators
    "validate_variable_set",
    "validate_all_varsets",
    "apply_missing_strategy",
    "check_data_quality",
    "validate_date_index",
    "check_train_valid_test_split",
]

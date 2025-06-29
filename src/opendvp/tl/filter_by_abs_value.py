from typing import Literal

import anndata as ad
import pandas as pd

from opendvp.utils import logger


def filter_by_abs_value(
    adata: ad.AnnData,
    feature_name: str,
    lower_bound: float | int | None = None,
    upper_bound: float | int | None = None,
    mode: Literal['absolute', 'quantile'] = "absolute", # How to interpret bounds
) -> ad.AnnData:
    """Filter cells in an AnnData object by a range of values for a specified feature.

    This function creates a boolean mask for each cell based on a feature's value,
    using either absolute thresholds or quantile-based thresholds. The feature can
    be a marker from `adata.X` or a continuous variable from `adata.obs`.
    Cells with feature values within the specified lower and upper bounds (inclusive)
    will pass the filter.

    Parameters:
    ----------
    adata : ad.AnnData
        AnnData object containing the data matrix and metadata.
    feature_name : str
        Name of the feature to filter on. The function will automatically determine
        if this feature is a marker in `adata.var_names` or a continuous variable
        in `adata.obs`. If found in both, a ValueError will be raised.
    lower_bound : float, int, or None, default None
        The lower threshold for filtering. If None, no lower bound is applied.
        If `mode` is 'absolute', this is the direct value.
        If `mode` is 'quantile', this is the quantile (0 <= value <= 1).
    upper_bound : float, int, or None, default None
        The upper threshold for filtering. If None, no upper bound is applied.
        If `mode` is 'absolute', this is the direct value.
        If `mode` is 'quantile', this is the quantile (0 <= value <= 1).
    mode : {'absolute', 'quantile'}, default 'absolute'
        Determines how `lower_bound` and `upper_bound` are interpreted:
        'absolute': Bounds are direct numerical values.
        'quantile': Bounds are quantiles (e.g., 0.25 for 25th percentile).

    Returns:
    -------
    ad.AnnData
        A copy of the input AnnData with a new boolean column in `.obs` indicating which cells passed the filter.
        The new column name will be `f"{feature_name}_filtered_by_{mode}"`.

    Raises:
    ------
    ValueError
        If `feature_name` is not found in `adata.var_names` or `adata.obs.columns`,
        if `feature_name` is found in both, if bounds are invalid,
        or if the `adata.obs` column is not numeric when used for filtering.
    """
    logger.info(f"Starting filter_by_abs_value for feature '{feature_name}'...")

    adata_copy = adata.copy()

    # --- 1. Input Validation and Data Extraction ---
    is_in_var = feature_name in adata_copy.var_names
    is_in_obs = feature_name in adata_copy.obs.columns

    source_determined: Literal['X', 'obs']

    if is_in_var and is_in_obs:
        raise ValueError(
            f"Feature '{feature_name}' found in both adata.var_names and adata.obs.columns. "
            "Please ensure the feature name is unique or specify its source if ambiguity is intended.")
    elif is_in_var:
        source_determined = 'X'
        data = adata_copy[:, feature_name].X
        data = data.to_array() if hasattr(data, 'to_array') else data
        data_series = pd.Series(data.flatten(), index=adata_copy.obs_names)
    elif is_in_obs:
        source_determined = 'obs'
        data_series = adata_copy.obs[feature_name]
        if not pd.api.types.is_numeric_dtype(data_series):
            raise ValueError(f"Feature '{feature_name}' in adata.obs is not numeric. "
                             "Filtering by value/quantile requires a numeric column.")
    else:
        raise ValueError(f"Feature '{feature_name}' not found in either adata.var_names or adata.obs.columns.")

    logger.info(f"Feature '{feature_name}' identified from adata.{source_determined}.")

    if lower_bound is None and upper_bound is None:
        raise ValueError("At least one of 'lower_bound' or 'upper_bound' must be provided.")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError("'lower_bound' cannot be greater than 'upper_bound'.")
    if mode not in ['absolute', 'quantile']:
        raise ValueError(f"Invalid 'mode' specified: '{mode}'. Must be 'absolute' or 'quantile'.")

    if mode == 'quantile':
        # Quantile bounds must be floats between 0 and 1
        if (lower_bound is not None and not (0 <= lower_bound <= 1)):
            raise ValueError("For 'quantile' mode, 'lower_bound' must be between 0 and 1 (inclusive).")
        if (upper_bound is not None and not (0 <= upper_bound <= 1)):
            raise ValueError("For 'quantile' mode, 'upper_bound' must be between 0 and 1 (inclusive).")
        # Convert to float if provided as int (e.g., 0 or 1) to ensure consistent quantile behavior
        if lower_bound is not None: 
            lower_bound = float(lower_bound)
        if upper_bound is not None: 
            upper_bound = float(upper_bound)


    # --- 2. Determine Actual Thresholds ---
    actual_lower_bound = None
    actual_upper_bound = None

    if mode == 'quantile':
        if lower_bound is not None:
            actual_lower_bound = data_series.quantile(lower_bound)
        if upper_bound is not None:
            actual_upper_bound = data_series.quantile(upper_bound)
    else: # mode == 'absolute'
        actual_lower_bound = lower_bound
        actual_upper_bound = upper_bound

    # --- 3. Apply Filtering Logic ---
    # Initialize mask to include all cells
    pass_filter = pd.Series(True, index=adata_copy.obs_names)

    if actual_lower_bound is not None:
        pass_filter = pass_filter & (data_series >= actual_lower_bound)
        logger.info(f"Keeping cells with '{feature_name}' >= {actual_lower_bound:.4f} "
                    f"(from {'quantile' if mode == 'quantile' else 'absolute'} bound: {lower_bound}).")
    if actual_upper_bound is not None:
        pass_filter = pass_filter & (data_series <= actual_upper_bound)
        logger.info(f"Keeping cells with '{feature_name}' <= {actual_upper_bound:.4f} "
                    f"(from {'quantile' if mode == 'quantile' else 'absolute'} bound: {upper_bound}).")

    # --- 4. Store Results and Return ---
    # Create a descriptive label for the new obs column
    new_obs_column_name = f"{feature_name}_filter"

    adata_copy.obs[new_obs_column_name] = pass_filter.to_numpy()

    num_cells_kept = pass_filter.sum()
    total_cells = len(pass_filter)
    logger.success(f"{num_cells_kept} of {total_cells} cells ({num_cells_kept/total_cells:.2%}) passed the filter.")
    logger.info(f"New boolean column '{new_obs_column_name}' added to adata.obs.")

    return adata_copy
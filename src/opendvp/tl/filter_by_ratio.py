import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


def filter_by_ratio(
    adata: ad.AnnData,
    end_cycle: str,
    start_cycle: str,
    label: str = "DAPI",
    min_ratio: float = 0.5,
    max_ratio: float = 1.05,
    add_detailed_pass_fail: bool = False
) -> ad.AnnData:
    """Filter cells by the ratio of two markers in an AnnData object.

    This function computes the ratio between two markers (columns) for each cell, and flags cells
    whose ratio falls within the specified range. The results are stored as new columns in `.obs`.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the data matrix and metadata.
    end_cycle : str
        Name of the marker/column for the numerator of the ratio.
    start_cycle : str
        Name of the marker/column for the denominator of the ratio.
    label : str, default 'DAPI'
        Label prefix for the new columns in `.obs`.
    min_ratio : float, default 0.5
        Minimum allowed ratio (exclusive).
    max_ratio : float, default 1.05
        Maximum allowed ratio (exclusive).
    add_detailed_pass_fail : bool, default False
        If True, adds intermediate boolean columns indicating which cells passed the
        lower bound (`_pass_nottoolow`) and upper bound (`_pass_nottoohigh`).

    Returns:
    -------
    ad.AnnData
        A new AnnData object with new columns in `.obs` for the ratio and pass/fail flags.

    Raises:
    ------
    ValueError
        If marker names are not found or if min_ratio >= max_ratio.
    """
    # --- 1. Validation and Setup ---
    logger.info("Starting filter_by_ratio...")
    if end_cycle not in adata.var_names:
        raise ValueError(f"end_cycle marker '{end_cycle}' not found in adata.var_names")
    if start_cycle not in adata.var_names:
        raise ValueError(f"start_cycle marker '{start_cycle}' not found in adata.var_names")
    if min_ratio >= max_ratio:
        raise ValueError("min_ratio must be less than max_ratio")

    adata_copy = adata.copy()

    # --- 2. Data Extraction and Calculation ---
    # Extract only the necessary data to avoid creating a large dense DataFrame.
    # Using .toarray() on a small slice is efficient.
    end_cycle_values = adata_copy[:, end_cycle].X.toarray().flatten()
    start_cycle_values = adata_copy[:, start_cycle].X.toarray().flatten()

    # Calculate the ratio, handling potential division by zero by replacing with NaN.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = end_cycle_values / start_cycle_values
    ratio[start_cycle_values == 0] = np.nan # Explicitly set ratio to NaN where denominator is 0

    # --- 3. Create Boolean Masks ---
    pass_nottoolow = ratio > min_ratio
    pass_nottoohigh = ratio < max_ratio
    # For the final pass, NaNs in the ratio (from division by zero) should be treated as False.
    pass_final = pd.Series(pass_nottoolow & pass_nottoohigh).fillna(False).to_numpy()

    # --- 4. Store Results in adata.obs ---
    adata_copy.obs[f'{label}_ratio'] = ratio
    adata_copy.obs[f'{label}_ratio_pass'] = pass_final
    if add_detailed_pass_fail:
        adata_copy.obs[f'{label}_ratio_pass_nottoolow'] = pass_nottoolow
        adata_copy.obs[f'{label}_ratio_pass_nottoohigh'] = pass_nottoohigh

    # --- 5. Logging ---
    # Use the calculated boolean masks for logging to be consistent.
    num_too_low = np.sum(ratio < min_ratio)
    num_too_high = np.sum(ratio > max_ratio)
    num_passed = np.sum(pass_final)
    total_cells = adata_copy.n_obs
    
    logger.info(f"Number of cells with {label} ratio < {min_ratio}: {num_too_low}")
    logger.info(f"Number of cells with {label} ratio > {max_ratio}: {num_too_high}")
    logger.info(f"Cells with {label} ratio between {min_ratio} and {max_ratio}: {num_passed}")
    logger.info(f"Cells filtered: {round(100 - (num_passed / total_cells) * 100, 2)}%")
    logger.success("filter_by_ratio complete.")

    return adata_copy
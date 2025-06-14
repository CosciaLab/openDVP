import time

import anndata as ad
import pandas as pd

from opendvp.utils import logger


def filter_by_ratio(
    adata: ad.AnnData,
    end_cycle: str,
    start_cycle: str,
    label: str = "DAPI",
    min_ratio: float = 0.5,
    max_ratio: float = 1.05
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

    Returns:
    -------
    ad.AnnData
        The input AnnData with new columns in `.obs` for the ratio and pass/fail flags.

    Raises:
    ------
    ValueError
        If marker names are not found or if min_ratio >= max_ratio.
    """
    if end_cycle not in adata.var_names:
        raise ValueError(f"end_cycle marker '{end_cycle}' not found in adata.var_names")
    if start_cycle not in adata.var_names:
        raise ValueError(f"start_cycle marker '{start_cycle}' not found in adata.var_names")
    if min_ratio >= max_ratio:
        raise ValueError("min_ratio must be less than max_ratio")

    data_matrix = adata.X.toarray()
    marker_df = pd.DataFrame(data=data_matrix, columns=adata.var_names)
    marker_df[f'{label}_ratio'] = marker_df[end_cycle] / marker_df[start_cycle]
    marker_df[f'{label}_ratio_pass_nottoolow'] = marker_df[f'{label}_ratio'] > min_ratio
    marker_df[f'{label}_ratio_pass_nottoohigh'] = marker_df[f'{label}_ratio'] < max_ratio
    marker_df[f'{label}_ratio_pass'] = marker_df[f'{label}_ratio_pass_nottoolow'] & marker_df[f'{label}_ratio_pass_nottoohigh']

    adata.obs[f'{label}_ratio'] = marker_df[f'{label}_ratio'].to_numpy()
    adata.obs[f'{label}_ratio_pass_nottoolow'] = marker_df[f'{label}_ratio_pass_nottoolow'].to_numpy()
    adata.obs[f'{label}_ratio_pass_nottoohigh'] = marker_df[f'{label}_ratio_pass_nottoohigh'].to_numpy()
    adata.obs[f'{label}_ratio_pass'] = adata.obs[f'{label}_ratio_pass_nottoolow'] & adata.obs[f'{label}_ratio_pass_nottoohigh']

    logger.info(f"Number of cells with {label} ratio < {min_ratio}: {sum(marker_df[f'{label}_ratio'] < min_ratio)}")
    logger.info(f"Number of cells with {label} ratio > {max_ratio}: {sum(marker_df[f'{label}_ratio'] > max_ratio)}")
    logger.info(f"Number of cells with {label} ratio between {min_ratio} and {max_ratio}: {sum(marker_df[f'{label}_ratio_pass'])}")
    logger.info(f"Percentage of cells filtered out: {round(100 - sum(marker_df[f'{label}_ratio_pass'])/len(marker_df)*100,2)}%")

    return adata
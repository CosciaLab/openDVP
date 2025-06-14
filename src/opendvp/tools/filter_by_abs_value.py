from typing import Literal

import anndata as ad
import pandas as pd

from opendvp.utils import logger


def filter_by_abs_value(
    adata: ad.AnnData,
    marker: str,
    mode: Literal['absolute', 'quantile'] = "quantile",
    value: float | int = 0.5,
    keep: Literal["below", "above"] = 'above',
) -> ad.AnnData:
    """Filter cells in an AnnData object by absolute marker value or quantile threshold.

    This function creates a boolean mask for each cell based on a marker's value, using either an absolute threshold or a quantile.
    The result is stored as a new column in `adata.obs` and the filtered AnnData is returned.

    Parameters:
    ----------
    adata : ad.AnnData
        AnnData object containing the data matrix and metadata.
    marker : str
        Name of the marker to filter on (must be present in `adata.var_names`).
    mode : {'absolute', 'quantile'}, default 'quantile'
        Whether to use an absolute value or quantile as the threshold.
    value : float or int, default 0.5
        The threshold value. If mode is 'absolute', this is the absolute threshold. If 'quantile', this is the quantile (0 < value < 1).
    keep : {'above', 'below'}, default 'above'
        Whether to keep cells 'above' or 'below' the threshold.

    Returns:
    -------
    ad.AnnData
        A copy of the input AnnData with a new boolean column in `.obs` indicating which cells passed the filter.

    Raises:
    ------
    ValueError
        If marker is not found, mutually exclusive arguments are violated, or parameters are invalid.
    """
    if marker not in adata.var_names:
        raise ValueError(f"Marker {marker} not found in adata.var_names")
    if mode == "absolute":
        if not isinstance(value, int | float):
            raise ValueError("For mode 'absolute', value must be a number (int or float)")
    elif mode == "quantile":
        if not isinstance(value, float):
            raise ValueError("For mode 'quantile', value must be a float between 0 and 1")
        if not (0 < value < 1):
            raise ValueError("For mode 'quantile', value must be between 0 and 1")
    
    adata_copy = adata.copy()
    data_matrix = adata_copy.X.toarray()
    marker_df = pd.DataFrame(data=data_matrix, columns=adata_copy.var_names)

    threshold = marker_df[marker].quantile(value) if mode == "quantile" else value

    label = f"{marker}_{keep}_{threshold}"
    operator = '>' if keep == 'above' else '<'
    marker_df[label] = marker_df.eval(f"{marker} {operator} @threshold")
    adata_copy.obs[label] = marker_df[label].to_numpy()
    logger.info(f"Number of cells with {marker} {keep} {threshold}: {sum(marker_df[label])}")
    return adata_copy
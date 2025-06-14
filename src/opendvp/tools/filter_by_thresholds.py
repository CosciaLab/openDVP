import time

import anndata as ad
import pandas as pd

from opendvp.utils import logger


def filter_adata_by_gates(adata: ad.AnnData, gates: pd.DataFrame, sample_id=None) -> ad.AnnData:
    """Filter an AnnData object to retain only markers specified in the `gates` DataFrame.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix to be filtered (cells x markers).
    gates : DataFrame
        A DataFrame containing a 'marker_id' column with marker names to retain. 
        Optionally contains a 'sample_id' column for filtering markers specific to a sample.
    sample_id : str, optional
        If provided, filters the `gates` DataFrame to only include rows with this sample_id.

    Returns:
    -------
    AnnData
        A new AnnData object with the same cells but filtered to include only the gated markers.
    
    Raises:
    ------
    AssertionError
        If marker IDs in `gates` are not all found in `adata.var.index`,
        or if `sample_id` is provided but not present in `gates`.
    """
    logger.info(" ---- filter_adata_by_gates : version number 1.0.1 ----")
    time_start = time.time()

    # Validate marker_id column
    if 'marker_id' not in gates.columns:
        raise ValueError("The `gates` DataFrame must contain a 'marker_id' column.")

    if sample_id is not None:
        if 'sample_id' not in gates.columns:
            raise ValueError("`sample_id` column is missing in `gates` but `sample_id` was provided.")
        gates = gates[gates['sample_id'] == sample_id]
        if gates.empty:
            raise ValueError(f"No markers found in gates for sample_id: {sample_id}")

    # Ensure markers are in adata
    marker_ids = gates['marker_id'].astype(str).unique()
    missing_markers = set(marker_ids) - set(adata.var_names)
    if missing_markers:
        raise ValueError(f"Markers not found in adata.var_names: {missing_markers}")

    # Robust slicing: use .loc to avoid surprises
    selected_markers = [m for m in marker_ids if m in adata.var_names]
    adata_filtered = adata[:, adata.var_names.isin(selected_markers)].copy()

    logger.info(f" ---- filter_adata_by_gates selected {len(selected_markers)} markers. Took {int(time.time() - time_start)}s ----")
    return adata_filtered
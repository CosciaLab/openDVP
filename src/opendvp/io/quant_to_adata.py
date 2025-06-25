import anndata as ad
import pandas as pd

from opendvp.utils import logger

#TODO not general enough, exemplar001 fails

def quant_to_adata(
        path: str,
        index_into_1_based : str | None = "CellID",
        meta_columns : list | None = None,
        ) -> ad.AnnData:
    """Convert cell quantification CSV data to an AnnData object for downstream analysis.

    This module provides a function to read a CSV file containing single-cell quantification data, extract metadata and marker intensities, and return an AnnData object suitable for spatial omics workflows. The function expects specific metadata columns and parses marker columns by splitting their names into mathematical operation and marker name.

    Parameters
    ----------
    path : str
        Path to the CSV file containing cell quantification data.
    index_into_1_based : str | None
        Column name to which to check if 0 exists, and if so add 1 to all values
        This is done so that cell index matches segmentation mask values
        If None, no modifications will be performed

    Returns:
    -------
    ad.AnnData
        AnnData object with cell metadata in `.obs` and marker intensities in `.X` and `.var`.

    Examples:
    --------
    >>> from opendvp.io import quant_to_adata
    >>> adata = quant_to_adata('my_quantification.csv')
    >>> print(adata)
    AnnData object with n_obs * n_vars = ...
    >>> adata.obs.head()
    >>> adata.var.head()

    Notes:
    ------
    - The CSV file must contain the following metadata columns: 'CellID', 'Y_centroid', 'X_centroid', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Orientation', 'Extent', 'Solidity'.
    - All other columns are treated as marker intensities and are split into 'math' and 'marker' components for AnnData.var.
    - Raises ValueError if required metadata columns are missing or if the file is not a CSV.
    - The function logs the number of cells and variables loaded, and the time taken for the operation.
    """
    if not path.endswith('.csv'):
        raise ValueError("The file should be a csv file")
    quant_data = pd.read_csv(path)
    quant_data.index = quant_data.index.astype(str)

    if not meta_columns:
        meta_columns = ['CellID', 'Y_centroid', 'X_centroid', 'Area', 'MajorAxisLength', 
                        'MinorAxisLength', 'Eccentricity', 'Orientation', 'Extent', 'Solidity']
    if not all([column in quant_data.columns for column in meta_columns]):
        raise ValueError("Not all metadata columns are present in the csv file")

    if index_into_1_based:
        quant_data[index_into_1_based] = quant_data[index_into_1_based].astype(int)
        if (quant_data[index_into_1_based] == 0).any():
            logger.info(f"Detected 0 in '{index_into_1_based}' — shifting all values by +1 for 1-based indexing.")
            quant_data[index_into_1_based] = quant_data[index_into_1_based] + 1

    metadata = quant_data[meta_columns].copy()
    data = quant_data.drop(columns=meta_columns).copy()
    variables = pd.DataFrame(index=data.columns)

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    return adata
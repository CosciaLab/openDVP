import time

import anndata as ad
import pandas as pd

from opendvp.utils import logger

#TODO not general enough, exemplar001 fails
#TODO let users pass list of metadata columns, everything else is data
#TODO automatically change CellID to CellID+1 (?) to match segmentation mask


def quant_to_adata(csv_data_path: str) -> ad.AnnData:
    """Convert cell quantification CSV data to an AnnData object for downstream analysis.

    This module provides a function to read a CSV file containing single-cell quantification data, extract metadata and marker intensities, and return an AnnData object suitable for spatial omics workflows. The function expects specific metadata columns and parses marker columns by splitting their names into mathematical operation and marker name.

    Parameters
    ----------
    csv_data_path : str
        Path to the CSV file containing cell quantification data.

    Returns
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
    time_start = time.time()

    if not csv_data_path.endswith('.csv'):
        raise ValueError("The file should be a csv file")
    quant_data = pd.read_csv(csv_data_path)
    quant_data.index = quant_data.index.astype(str)

    meta_columns = ['CellID', 'Y_centroid', 'X_centroid',
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'Orientation', 'Extent', 'Solidity']
    if not all([column in quant_data.columns for column in meta_columns]):
        raise ValueError("The metadata columns are not present in the csv file")

    metadata = quant_data[meta_columns]
    data = quant_data.drop(columns=meta_columns)
    variables = pd.DataFrame(
        index=data.columns,
        data={
            "math": [column_name.split("_")[0] for column_name in data.columns],
            "marker": ["_".join(column_name.split("_")[1:]) for column_name in data.columns],
        },
    )

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    logger.info(f" ---- read_quant is done, took {int(time.time() - time_start)}s  ----")
    return adata
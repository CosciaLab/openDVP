import time

import anndata as ad
import pandas as pd

from opendvp.utils import logger


def quant_to_adata(csv_data_path: str) -> ad.AnnData:
    """Read the quantification data from a csv file and return an anndata object.

    :param csv_data_path: path to the csv file
    :return: an anndata object.
    """
    #TODO not general enough, exemplar001 fails
    #TODO let users pass list of metadata columns, everything else is data

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
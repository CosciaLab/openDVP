import pandas as pd
import anndata as ad
from loguru import logger
import time, os

def read_quant(csv_data_path) -> ad.AnnData:
    """
    Read the quantification data from a csv file and return an anndata object
    :param csv_data_path: path to the csv file
    :return: an anndata object
    """

    #TODO not general enough, exemplar001 fails

    logger.info(" ---- read_quant : version number 1.1.0 ----")
    time_start = time.time()

    assert csv_data_path.endswith('.csv'), "The file should be a csv file"
    df = pd.read_csv(csv_data_path)
    df.index = df.index.astype(str)

    meta_columns = ['CellID', 'Y_centroid', 'X_centroid',
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'Orientation', 'Extent', 'Solidity']
    assert all([column in df.columns for column in meta_columns]), "The metadata columns are not present in the csv file"

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)
    variables = pd.DataFrame(
        index=data.columns,
        data={"math": [column_name.split("_")[0] for column_name in data.columns],
            "marker": ["_".join(column_name.split("_")[1:]) for column_name in data.columns]})

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    logger.info(f" ---- read_quant is done, took {int(time.time() - time_start)}s  ----")
    return adata

def switch_adat_var_index(adata, new_index):
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-11-16

    Description:
    Switch the index of adata.var to a new index. Useful for switching between gene names and protein names.

    Arg:
        adata: anndata object
        new_index: pandas series, new index to switch to
    Returns:
        adata: anndata object, with the new index
    """
    adata_copy = adata.copy()

    adata_copy.var[adata_copy.var.index.name] = adata_copy.var.index
    adata_copy.var.set_index(new_index, inplace=True)
    adata_copy.var.index.name = new_index
    
    return adata_copy

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def save_adata_checkpoint(adata, path_to_dir, checkpoint_name):
    try:    
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_dir,checkpoint_name), exist_ok=True)
        basename = f"{os.path.join(path_to_dir,checkpoint_name)}/{get_datetime()}_{checkpoint_name}_adata"
        
        # Save h5ad file
        try:
            logger.info("Writing h5ad")
            adata.write_h5ad(filename = basename + ".h5ad")
            logger.success("Wrote h5ad file")
        except (OSError, IOError, ValueError) as e:
                logger.error(f"Could not write h5ad file: {e}")
                return
        
        # Save CSV file
        try:
            logger.info("Writing parquet")
            adata.to_df().to_parquet(path=basename + ".parquet")
            logger.success("Wrote parquet file")
        except (OSError, IOError, ValueError) as e:
            logger.error(f"Could not write parquet file: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in save_adata_checkpoint: {e}")
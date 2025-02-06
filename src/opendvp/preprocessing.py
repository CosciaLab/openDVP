from loguru import logger
import time
import pandas as pd
import numpy as np


def process_gates_for_sm(gates:pd.DataFrame, sample_id:int) -> pd.DataFrame:
    """ Process gates dataframe to be in log1p scale """
    logger.info(" ---- process_gates_for_sm : version number 1.2.0 ----")
    time_start = time.time()

    df = gates.copy()

    df['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = df[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'marker', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap


def negate_var_by_ann(adata, path_to_geojson, marker_column, value_to_impute, label) -> ad.AnnData:

    # first label adata
    adata = filter_by_annotation(adata, path_to_geojson, column_name=label)

    # create array of data to correct
    array = adata[:,marker_column].X.toarray()

    logger.info( f" how many zeroes: {np.count_nonzero(array==value_to_impute)}")
    logger.info( f"array shape {array.shape}")
    logger.info( f"array mean {array.mean()}")
    logger.info( f"data type {type(array)}")

    # impute values
    array[~adata.obs[label].values] = value_to_impute

    logger.info( f" how many zeroes after: {np.count_nonzero(array==value_to_impute)}")
    logger.info( f"array shape {array.shape}")
    logger.info( f"array mean {array.mean()}")

    # replace array in adata
    adata[:,marker_column].X = array

    return adata
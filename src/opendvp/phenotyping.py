from loguru import logger
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
# import scimap as sm


def read_gates(gates_csv_path, sample_id=None) -> pd.DataFrame:
    """ Read the gates data from a csv file and return a dataframe """
    logger.info(" ---- read_gates : version number 1.1.0 ----")
    time_start = time.time()

    assert gates_csv_path.endswith('.csv'), "The file should be a csv file"
    gates = pd.read_csv(gates_csv_path)
    
    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    assert "gate_value" in gates.columns, "The column gate_value is not present in the csv file"
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")
    logger.info(f"  Markers found: {gates.marker_id.unique()}")
    logger.info(f"  Samples found: {gates.sample_id.unique()}")

    if sample_id is not None:
        assert "sample_id" in gates.columns, "The column sample_id is not present in the csv file"
        gates = gates[gates['sample_id']==sample_id]
        logger.info(f"  Found {gates.shape[0]} valid gates for sample {sample_id}")

    logger.info(f" ---- read_gates is done, took {int(time.time() - time_start)}s  ----")
    return gates
    
def process_gates_for_sm(gates:pd.DataFrame, sample_id) -> pd.DataFrame:
    """ Process gates dataframe to be in log1p scale """
    logger.info(" ---- process_gates_for_sm : version number 1.2.0 ----")
    time_start = time.time()

    df = gates.copy()

    df['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = df[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'markers', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap

def negate_var_by_ann(adata, target_variable, target_annotation_column , quantile_for_imputation=0.05) -> ad.AnnData:

    assert quantile_for_imputation >= 0 and quantile_for_imputation <= 1, "Quantile should be between 0 and 1"
    assert target_variable in adata.var_names, f"Variable {target_variable} not found in adata.var_names"
    assert target_annotation_column in adata.obs.columns, f"Annotation column {target_annotation_column} not found in adata.obs.columns"

    adata_copy = adata.copy()

    target_var_idx = adata_copy.var_names.get_loc(target_variable)
    target_rows = adata_copy.obs[target_annotation_column].values
    value_to_impute = np.quantile(adata_copy[:, target_var_idx].X.toarray(), quantile_for_imputation)
    logger.info(f"Imputing with {quantile_for_imputation}% percentile value = {value_to_impute}")

    adata_copy.X[target_rows, target_var_idx] = value_to_impute
    return adata_copy
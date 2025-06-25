import numpy as np
import pandas as pd

from opendvp.utils import logger


def import_thresholds(
    gates_csv_path :str, 
    sample_id : str | int | None = None, 
    log1p: bool = True
) -> pd.DataFrame:
    """Read gate thresholds from a CSV file, filter, and optionally log1p-transform for scimap compatibility.

    This function loads a CSV file containing gate thresholds, validates required columns, filters out rows
    with gate values of 0.0 (assumed not gated), optionally selects gates for a specific sample, and can
    log1p-transform gate values for downstream analysis (e.g., scimap). Logs progress and summary information.

    Args:
        gates_csv_path (str):
            Path to the CSV file containing gate thresholds. Must end with '.csv'.
        sample_id (str or int, optional):
            If provided, only gates for this sample will be returned and the output column will be named accordingly.
        log1p (bool, default True):
            If True, applies log1p transformation to gate values and formats output for scimap.

    Returns:
        pd.DataFrame: Filtered DataFrame containing valid gates, with columns including 'markers' and the sample_id
        (if log1p=True), or the original columns if log1p=False.

    Raises:
        ValueError: If the file extension is not '.csv', or required columns are missing.

    Example:
        >>> gates = read_and_process_gates('gates.csv', sample_id='sample1', log1p=True)
        >>> print(gates.head())
    """
    if not gates_csv_path.endswith('.csv'):
        raise ValueError("The file should be a csv file")
    gates = pd.read_csv(gates_csv_path)
    if "gate_value" not in gates.columns:
        raise ValueError("gate_value is not present")
    if "sample_id" not in gates.columns:
         raise ValueError("sample_id is not present")

    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")
    logger.info(f"  Markers found: {gates.marker_id.unique()}")
    logger.info(f"  Samples found: {gates.sample_id.unique()}")

    if sample_id is not None:
        if "sample_id" not in gates.columns:
            raise ValueError("The column sample_id is not present in the csv file")
        gates = gates[gates['sample_id'] == sample_id]
        logger.info(f"  Found {gates.shape[0]} valid gates for sample {sample_id}")

    if log1p:
        logger.info("   Applying log1p transformation to gate values and formatting for scimap.")
        gates_copy = gates.copy()
        gates_copy['log1p_gate_value'] = np.log1p(gates.gate_value)
        gates_for_scimap = gates_copy[['marker_id', 'log1p_gate_value']]
        gates_for_scimap = gates_for_scimap.rename(columns={'marker_id': 'markers', 'log1p_gate_value': sample_id if sample_id is not None else 'log1p_gate_value'})
        logger.info(f"   Output DataFrame columns: {gates_for_scimap.columns.tolist()}")
        return gates_for_scimap
    else:
        return gates
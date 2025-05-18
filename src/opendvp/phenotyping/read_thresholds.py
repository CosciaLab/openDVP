from opendvp.logger import logger
import time
import pandas as pd

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
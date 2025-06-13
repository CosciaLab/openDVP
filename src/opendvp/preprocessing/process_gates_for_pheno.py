import time

import pandas as pd

from opendvp.logger import logger


def process_gates_for_sm(gates:pd.DataFrame, sample_id) -> pd.DataFrame:
    """Process gates dataframe to be in log1p scale"""
    logger.info(" ---- process_gates_for_sm : version number 1.2.0 ----")
    time_start = time.time()

    df = gates.copy()

    df['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = df[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'markers', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap
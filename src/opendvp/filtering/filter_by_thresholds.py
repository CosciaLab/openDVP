import anndata as ad
import pandas as pd
from opendvp import logger
import time

def filter_adata_by_gates(adata: ad.AnnData, gates: pd.DataFrame, sample_id=None) -> ad.AnnData:
    """ Filter the adata object by the gates """
    logger.info(" ---- filter_adata_by_gates : version number 1.0.0 ----")
    time_start = time.time()
    assert gates.marker_id.isin(adata.var.index).all(), "Some markers in the gates are not present in the adata object"
    
    if sample_id is not None:
        assert sample_id in gates.columns, "The sample_id is not present in the gates"
        gates = gates[gates['sample_id']==sample_id]
    
    adata = adata[:, gates.marker_id]
    logger.info(f" ---- filter_adata_by_gates is done, took {int(time.time() - time_start)}s  ----")
    return adata
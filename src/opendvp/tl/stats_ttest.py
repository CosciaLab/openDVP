from datetime import datetime

import anndata as ad
import numpy as np
import pingouin as pg
import statsmodels.stats.multitest as smm

from opendvp.utils import logger

date = datetime.now().strftime("%Y%m%d")

def stats_ttest(
    adata: ad.AnnData,
    grouping: str,
    group1: str,
    group2: str,
    FDR_threshold: float = 0.05
) -> ad.AnnData:
    """Perform a t-test for all columns of an AnnData object between two groups.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    grouping : str
        Column header in adata.obs, categorizing different groups to test.
    group1 : str
        Value in grouping column to be tested against.
    group2 : str
        Value in grouping column to be tested against group 1.
    FDR_threshold : float, default 0.05
        The threshold for the FDR correction.

    Returns:
    ------- 
    None
        Results are saved to adata.var in-place.
    """
    if group1 not in adata.obs[grouping].unique() or group2 not in adata.obs[grouping].unique():
        raise ValueError(f"Given groups not found in {grouping}")
    
    adata_copy = adata.copy()
    t_values = []
    p_values = []
    diffs = []

    X = np.asarray(adata_copy.X)
    for column in adata_copy.var.index:
        mask1 = (adata_copy.obs[grouping] == group1).to_numpy(dtype=bool)
        mask2 = (adata_copy.obs[grouping] == group2).to_numpy(dtype=bool)
        col_idx = adata_copy.var.index.get_loc(column)
        array_1 = X[mask1][:, col_idx].flatten()
        array_2 = X[mask2][:, col_idx].flatten()
        result = pg.ttest(x=array_1, y=array_2, paired=False, alternative="two-sided")
        t_values.append(result.iloc[0, 0])
        p_values.append(result.iloc[0,3])
        diffs.append(np.mean(array_1) - np.mean(array_2))

    # Add results to adata object
    adata_copy.var["t_val"] = t_values
    adata_copy.var["p_val"] = p_values
    adata_copy.var["mean_diff"] = diffs
    # Correct for multiple testing
    result_BH = smm.multipletests(adata_copy.var["p_val"].values, alpha=FDR_threshold, method='fdr_bh')
    adata_copy.var["sig"] = result_BH[0]
    adata_copy.var["p_corr"] = result_BH[1]
    adata_copy.var['-log10_p_corr'] = -np.log10(adata_copy.var['p_corr'])

    logger.info(f"Using pingouin.ttest to perform unpaired two-sided t-test between {group1} and {group2}")
    logger.info(f"Using Benjamini-Hochberg for FDR correction, with a threshold of {FDR_threshold}")
    logger.info(f"The test found {np.sum(adata_copy.var['sig'])} proteins to be significantly")

    return adata_copy
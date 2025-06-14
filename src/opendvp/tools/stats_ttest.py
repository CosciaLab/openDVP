from datetime import datetime

import anndata as ad
import numpy as np
import pingouin as pg
import statsmodels.stats.multitest as smm

from opendvp.utils import logger

date = datetime.now().strftime("%Y%m%d")

def ttest_adata(
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
    adata_copy = adata.copy()
    t_values = []
    p_values = []
    FC = []
    
    X = np.asarray(adata_copy.X)
    for column in adata_copy.var.index:
        mask1 = (adata_copy.obs[grouping] == group1).to_numpy(dtype=bool)
        mask2 = (adata_copy.obs[grouping] == group2).to_numpy(dtype=bool)
        col_idx = adata_copy.var.index.get_loc(column)
        array_1 = X[mask1][:, col_idx].flatten()
        array_2 = X[mask2][:, col_idx].flatten()
        result = pg.ttest(x=array_1, y=array_2, paired=False, alternative="two-sided", correction=False, r=0.707)
        t_values.append(result.iloc[0,0])
        p_values.append(result.iloc[0,3])
        FC.append(np.mean(array_1) - np.mean(array_2))

    # Add results to adata object
    adata_copy.var["t_val"] = t_values
    adata_copy.var["p_val"] = p_values
    adata_copy.var["log2_FC"] = FC
    # Correct for multiple testing
    result_BH = smm.multipletests(adata_copy.var["p_val"].values, alpha=FDR_threshold, method='fdr_bh')
    adata_copy.var["significant_BH"] = result_BH[0]
    adata_copy.var["p_val_corr_BH"] = result_BH[1]
    adata_copy.var['-log10(p_val_corr)_BH'] = -np.log10(adata_copy.var['p_val_corr_BH'])

    logger.info(f"Testing for differential expression between {group1} and {group2}")
    logger.info("Using pingouin.ttest to perform t-test, two-sided, not paired")
    logger.info("Using statsmodels.stats.multitest.multipletests to correct for multiple testing")
    logger.info(f"Using Benjamini-Hochberg for FDR correction, with a threshold of {FDR_threshold}")
    logger.info(f"The test found {np.sum(adata_copy.var['significant_BH'])} proteins to be significantly")

    return adata_copy
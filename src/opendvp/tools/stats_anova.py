from datetime import datetime

date = datetime.now().strftime("%Y%m%d")

from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.stats.multitest as smm

date = datetime.now().strftime("%Y%m%d")


#TODO tukey of only significant from anova
#TODO average per sample as default (test assumes independence)

def anova_adata(
    adata: ad.AnnData,
    grouping: str,
    FDR_threshold: float = 0.05,
    posthoc = "pairwise_tukey",
) -> ad.AnnData:
    """Perform one-way ANOVA for all columns of an AnnData object across all groups in a categorical column.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    grouping : str
        Column header in adata.obs, categorizing different groups to test.
    FDR_threshold : float, default 0.05
        The threshold for the FDR correction.

    Returns:
    -------
    None
        Results are saved to adata.var in-place.
    """
    adata_copy = adata.copy()
    F_vals = []
    p_vals = []
    posthoc_results = []

    X = np.asarray(adata_copy.X)
    group_labels = adata_copy.obs[grouping].astype(str)

    for column in adata_copy.var.index:
        col_idx = adata_copy.var.index.get_loc(column)
        values = X[:, col_idx].flatten()
        df_feature = pd.DataFrame({'group': group_labels.values, 'value': values})
        result = pg.anova(data = df_feature, dv = "value", between="group", detailed=False)
        F_vals.append(result['F'].values[0])
        p_vals.append(result['p-unc'].values[0])
        
        if posthoc:
            results_posthoc = pg.pairwise_tukey(data=df_feature, dv="value", between="group", effsize="hedges")
            results_posthoc = results_posthoc.copy()
            results_posthoc.insert(0, "feature", column)
            posthoc_results.append(results_posthoc)
        

    adata_copy.var['anova_F'] = F_vals
    adata_copy.var['anova_p-unc'] = p_vals
    # Multiple testing correction
    result_BH = smm.multipletests(adata_copy.var['anova_p-unc'].values, alpha=FDR_threshold, method='fdr_bh')
    adata_copy.var['anova_significant_BH'] = result_BH[0]
    adata_copy.var['anova_p_corr_BH'] = result_BH[1]
    adata_copy.var['-log10(anova_p_corr)_BH'] = -np.log10(adata_copy.var['anova_p_corr_BH'])
    print(f"ANOVA across groups in '{grouping}' completed. {np.sum(adata_copy.var['anova_significant_BH'])} features significant at FDR < {FDR_threshold}.")
    if posthoc and posthoc_results:
        posthoc_df = pd.concat(posthoc_results, ignore_index=True)
        adata_copy.uns['anova_posthoc'] = posthoc_df
    return adata_copy
    
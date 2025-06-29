import anndata as ad
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.stats.multitest as smm

from opendvp.utils import logger


def stats_anova(
    adata: ad.AnnData,
    grouping: str,
    FDR_threshold: float = 0.05,
    posthoc: str | None = "pairwise_tukey",
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
    posthoc : str or None, default "pairwise_tukey"
        Post-hoc test to perform on significant features. Currently only 'pairwise_tukey' is supported.
        If None, no post-hoc test is run.

    Returns:
    -------
    ad.AnnData
        A new AnnData object with ANOVA results in `.var` and post-hoc results for significant
        features in `.uns['anova_posthoc']`.
    """
    if grouping not in adata.obs.columns:
        raise ValueError(f"Grouping column '{grouping}' not found in adata.obs.columns")

    logger.info(f"Starting one-way ANOVA across groups in '{grouping}'.")
    adata_copy = adata.copy()
    F_vals = []
    p_vals = []

    X = np.asarray(adata_copy.X)
    group_labels = adata_copy.obs[grouping].astype(str)

    # --- 1. Perform ANOVA for all features ---
    for column in adata_copy.var.index:
        col_idx = adata_copy.var.index.get_loc(column)
        values = X[:, col_idx].flatten()
        df_feature = pd.DataFrame({'group': group_labels.to_numpy(), 'value': values})

        # Check for conditions that would cause ANOVA to fail
        if df_feature['value'].isna().all() or df_feature.groupby('group')['value'].nunique().le(1).any():
            F_vals.append(np.nan)
            p_vals.append(np.nan)
            continue

        try:
            result = pg.anova(data=df_feature, dv="value", between="group", detailed=False)
            F_vals.append(result['F'].to_numpy()[0])
            p_vals.append(result['p-unc'].to_numpy()[0])
        except Exception:
            F_vals.append(np.nan)
            p_vals.append(np.nan)

    # --- 2. Add ANOVA results and perform multiple testing correction ---
    adata_copy.var['anova_F'] = F_vals
    adata_copy.var['anova_p-unc'] = p_vals

    p_vals_array = np.array(p_vals)
    valid_p_mask = ~np.isnan(p_vals_array)
    significant_bh = np.full(adata_copy.n_vars, False, dtype=bool)
    p_corr_bh = np.full(adata_copy.n_vars, np.nan)

    if np.any(valid_p_mask):
        pvals_to_correct = p_vals_array[valid_p_mask]
        result_BH = smm.multipletests(pvals_to_correct, alpha=FDR_threshold, method='fdr_bh')
        significant_bh[valid_p_mask] = result_BH[0]
        p_corr_bh[valid_p_mask] = result_BH[1]

    adata_copy.var['anova_sig_BH'] = significant_bh
    adata_copy.var['anova_p_corr'] = p_corr_bh
    adata_copy.var['-log10_anova_p_corr'] = -np.log10(p_corr_bh)

    n_significant = np.sum(significant_bh)
    logger.info(f"ANOVA found {n_significant} significant features at FDR < {FDR_threshold}.")

    # --- 3. Perform post-hoc test ONLY on significant features ---
    if posthoc == "pairwise_tukey" and n_significant > 0:
        logger.info(f"Performing post-hoc Tukey test on {n_significant} significant features.")
        posthoc_results = []
        significant_features = adata_copy.var.index[significant_bh]

        for feature_name in significant_features:
            col_idx = adata_copy.var.index.get_loc(feature_name)
            values = X[:, col_idx].flatten()
            df_feature = pd.DataFrame({'group': group_labels.to_numpy(), 'value': values})
            results_posthoc = pg.pairwise_tukey(data=df_feature, dv="value", between="group", effsize="hedges")
            results_posthoc.insert(0, "feature", feature_name)
            posthoc_results.append(results_posthoc)

        if posthoc_results:
            posthoc_df = pd.concat(posthoc_results, ignore_index=True)
            adata_copy.uns['anova_posthoc'] = posthoc_df
            logger.success("Post-hoc analysis complete. Results stored in `adata.uns['anova_posthoc']`.")

    return adata_copy
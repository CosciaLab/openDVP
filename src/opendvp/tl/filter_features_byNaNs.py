import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


def filter_features_byNaNs(
    adata: ad.AnnData,
    threshold: float = 0.7,
    grouping: str | None = None,
    valid_in_ANY_or_ALL_groups : Literal["ANY", "ALL"] = 'ANY'
) -> ad.AnnData:
    """Filter out proteins that have a NaN proportion above the threshold, for each group in the grouping variable.

    Parameters
    ----------
    adata : AnnData
        AnnData object to filter.
    threshold : float, default 0.7
        Proportion of valid values above which a protein is considered valid (between 0 and 1).
    grouping : Optional[str], default None
        Name of the column in adata.obs to discriminate the groups by. 
        If provided, counting of NaNs and validity is done per group.
    valid_in_ANY_or_ALL_groups : {'ANY', 'ALL'}, default 'ANY'
        'ANY' means that if a protein passes the threshold in any group it will be kept.
        'ALL' means that a protein must pass validity threshold for all groups to be kept (more stringent).

    Returns:
    -------
    AnnData
        Filtered AnnData object.
        The quality control metrics (e.g., NaN counts, valid proportions) are added to `adata.var`.
        A complete QC matrix for all initial features is stored in `adata.uns['filter_features_byNaNs_qc_metrics']`.
        The `adata.var` of the returned object will contain its original columns, plus 'mean' and 'nan_proportions'
        (derived from 'overall_mean' and 'overall_nan_proportions').
    """
    # TODO let users decide on an absolute number of valid values
    logger.info(f"Filtering protein with at least {threshold*100}% valid values in {valid_in_ANY_or_ALL_groups} group")
    warnings.simplefilter("ignore", category=RuntimeWarning)

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    adata_copy = adata.copy()
    initial_protein_count = adata_copy.shape[1]
    
    # Store original var columns before adding any QC metrics
    original_var_columns = adata_copy.var.columns.tolist()
    
    # --- Step 1: Calculate overall QC metrics for all features ---
    logger.info("Calculating overall QC metrics for all features.")
    X_all = np.asarray(adata_copy.X).astype('float64')
    adata_copy.var["overall_mean"] = np.nanmean(X_all, axis=0).round(3)
    adata_copy.var['overall_nan_count'] = np.isnan(X_all).sum(axis=0)
    adata_copy.var['overall_valid_count'] = (~np.isnan(X_all)).sum(axis=0)
    adata_copy.var['overall_nan_proportions'] = np.isnan(X_all).mean(axis=0).round(3)
    adata_copy.var['overall_valid'] = adata_copy.var['overall_nan_proportions'] < (1.0 - threshold)

    # Default mask if no grouping
    proteins_to_keep_mask = adata_copy.var.overall_valid.to_numpy().astype(bool) 

    # --- Step 2: Calculate group-specific QC metrics if grouping is provided ---
    if grouping:
        unique_groups = adata.obs[grouping].unique().tolist()
        logger.info(f"Filtering by groups, {grouping}: {unique_groups}")

        group_valid_cols = []
        for group in unique_groups:
            adata_group = adata[adata.obs[grouping] == group]
            logger.info(f" {group} has {adata_group.shape[0]} samples") 
            X_group = np.asarray(adata_group.X).astype('float64')
            
            # Calculate metrics for the current group and add directly to adata_copy.var
            adata_copy.var[f"{group}_mean"] = np.nanmean(X_group, axis=0).round(3)    
            adata_copy.var[f'{group}_nan_count'] = np.isnan(X_group).sum(axis=0)
            adata_copy.var[f'{group}_valid_count'] = (~np.isnan(X_group)).sum(axis=0)
            adata_copy.var[f'{group}_nan_proportions'] = np.isnan(X_group).mean(axis=0).round(3)
            
            # Determine validity for the current group
            current_group_valid_col = f'{group}_valid'
            adata_copy.var[current_group_valid_col] = adata_copy.var[f'{group}_nan_proportions'] < (1.0 - threshold)
            group_valid_cols.append(current_group_valid_col)
    
        # Calculate overall validity columns based on groups
        adata_copy.var['valid_in_all_groups'] = adata_copy.var[group_valid_cols].all(axis=1)
        adata_copy.var['valid_in_any_group'] = adata_copy.var[group_valid_cols].any(axis=1)
        adata_copy.var['not_valid_in_any_group'] = ~adata_copy.var['valid_in_any_group']

        if valid_in_ANY_or_ALL_groups == 'ALL':
            proteins_to_keep_mask = adata_copy.var.valid_in_all_groups.to_numpy().astype(bool)
            logger.info("Keeping proteins that pass 'ALL' groups criteria.")
        elif valid_in_ANY_or_ALL_groups == 'ANY':
            proteins_to_keep_mask = adata_copy.var.valid_in_any_group.to_numpy().astype(bool)
            logger.info("Keeping proteins that pass 'ANY' group criteria.")

    else:
        logger.info("No grouping variable was provided, filtering across all samples using overall validity.")
        logger.debug(f"adata has {adata_copy.shape[0]} samples")
        # For non-grouping, 'valid' and 'not_valid' are equivalent to 'overall_valid' and '~overall_valid'
        # These are kept for consistency with previous versions that didn't have 'overall_' prefix
        adata_copy.var['valid'] = adata_copy.var['overall_valid']
        adata_copy.var['not_valid'] = ~adata_copy.var['overall_valid']
    
    # --- Step 3: Store full QC matrix and perform filtering ---
    # Store the complete adata_copy.var (with all calculated metrics) into adata.uns
    adata_copy.uns['filter_features_byNaNs_qc_metrics'] = adata_copy.var.copy()
    logger.info("Complete QC metrics for all initial features stored in `adata.uns['filter_features_byNaNs_qc_metrics']`.")

    # Apply the filtering mask
    adata_copy = adata_copy[:, proteins_to_keep_mask]

    # --- Step 4: Clean up adata.var for the returned object ---
    # Create a new DataFrame for adata_copy.var with only the desired columns
    final_var_df = adata_copy.var[original_var_columns].copy()

    # Add the renamed overall mean and nan proportions if they exist in the filtered var
    if 'overall_mean' in adata_copy.var.columns:
        final_var_df['mean'] = adata_copy.var['overall_mean']
    if 'overall_nan_proportions' in adata_copy.var.columns:
        final_var_df['nan_proportions'] = adata_copy.var['overall_nan_proportions']
    
    # Assign the new var DataFrame back to adata_copy
    adata_copy.var = final_var_df

    logger.info(f"{adata_copy.shape[1]} proteins were kept.")
    logger.info(f"{initial_protein_count - adata_copy.shape[1]} proteins were removed.")

    logger.success("filter_features_byNaNs complete.")
    return adata_copy
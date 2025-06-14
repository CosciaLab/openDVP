import anndata as ad
import numpy as np

from opendvp.utils import logger


def impute_marker_with_annotation(
    adata : ad.AnnData, 
    target_variable : str,
    target_annotation_column : str, 
    quantile_for_imputation : float = 0.05
) -> ad.AnnData:
    """Change value of a feature in an AnnData object for rows matching a specific annotation.
    
    Using a specified quantile value from the variable's distribution.

    Parameters:
    ----------
    adata : ad.AnnData
        The annotated data matrix.
    target_variable : str
        The variable (gene/feature) to impute.
    target_annotation_column : str
        The column in adata.obs to use for selecting rows to impute.
    quantile_for_imputation : float, optional
        The quantile to use for imputation (default is 0.05).

    Returns:
    -------
    ad.AnnData
        A copy of the AnnData object with imputed values.
    """
    if not (0 <= quantile_for_imputation <= 1):
        raise ValueError("Quantile should be between 0 and 1")
    if target_variable not in adata.var_names:
        raise ValueError(f"Variable {target_variable} not found in adata.var_names")
    if target_annotation_column not in adata.obs.columns:
        raise ValueError(f"Annotation column {target_annotation_column} not found in adata.obs.columns")

    adata_copy = adata.copy()

    target_var_idx = adata_copy.var_names.get_loc(target_variable)
    # Get boolean mask for rows to impute
    target_rows_mask = adata_copy.obs[target_annotation_column].values
    # Convert X to dense numpy array for assignment
    X_dense = adata_copy.X.toarray() if hasattr(adata_copy.X, "toarray") else np.array(adata_copy.X)
    value_to_impute = np.quantile(
        X_dense[:, target_var_idx], quantile_for_imputation
    )
    logger.info(
        f"Imputing with {quantile_for_imputation}% percentile value = {value_to_impute}"
    )

    X_dense[target_rows_mask, target_var_idx] = value_to_impute
    adata_copy.X = X_dense
    return adata_copy
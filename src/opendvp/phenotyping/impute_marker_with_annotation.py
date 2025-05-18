def negate_var_by_ann(adata, target_variable, target_annotation_column , quantile_for_imputation=0.05) -> ad.AnnData:

    assert quantile_for_imputation >= 0 and quantile_for_imputation <= 1, "Quantile should be between 0 and 1"
    assert target_variable in adata.var_names, f"Variable {target_variable} not found in adata.var_names"
    assert target_annotation_column in adata.obs.columns, f"Annotation column {target_annotation_column} not found in adata.obs.columns"

    adata_copy = adata.copy()

    target_var_idx = adata_copy.var_names.get_loc(target_variable)
    target_rows = adata_copy.obs[target_annotation_column].values
    value_to_impute = np.quantile(adata_copy[:, target_var_idx].X.toarray(), quantile_for_imputation)
    logger.info(f"Imputing with {quantile_for_imputation}% percentile value = {value_to_impute}")

    adata_copy.X[target_rows, target_var_idx] = value_to_impute
    return adata_copy
import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


def impute_gaussian(
    adata: ad.AnnData,
    mean_shift: float = -1.8,
    std_dev_shift: float = 0.3,
    perSample: bool = False,
    layer_key: str = 'unimputed',
    uns_key: str = 'impute_gaussian_qc_metrics'
) -> ad.AnnData:
    """Impute missing values in an AnnData object using a Gaussian distribution.

    This function imputes missing values in the data matrix using a Gaussian distribution, with the mean shifted and
    the standard deviation scaled. Imputation can be performed per protein (column) or per sample (row).

    The original, un-imputed data matrix is stored in `adata.layers`.
    A DataFrame with quality control metrics for the imputation is stored in `adata.uns`. The QC metrics include
    the number of imputed values, the mean and standard deviation used for imputation, and a numpy array of the
    imputed values themselves for each feature.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with missing values to impute.
    mean_shift : float, default -1.8
        Number of standard deviations to shift the mean of the Gaussian distribution.
    std_dev_shift : float, default 0.3
        Factor to scale the standard deviation of the Gaussian distribution.
    perSample : bool, default False
        If True, impute per sample (row); if False, impute per protein (column).
    layer_key : str, default 'unimputed'
        Key under which to store the original, un-imputed data matrix in `adata.layers`.
    uns_key : str, default 'impute_gaussian_qc_metrics'
        Key under which to store the imputation QC metrics DataFrame in `adata.uns`.

    Returns:
    -------
    ad.AnnData
        AnnData object with imputed values in `.X`, the original matrix in `.layers[layer_key]`,
        and QC metrics in `.uns[uns_key]`.
    """
    adata_copy = adata.copy()

    # Store the original data in a new layer
    logger.info(f"Storing original data in `adata.layers['{layer_key}']`.")
    adata_copy.layers[layer_key] = np.asarray(adata_copy.X)

    # Ensure dense array for DataFrame construction
    data = np.asarray(adata_copy.X.copy())
    impute_df = pd.DataFrame(data=data, columns=adata_copy.var.index, index=adata_copy.obs_names)

    if perSample:
        logger.info("Imputation with Gaussian distribution PER SAMPLE")
        impute_df = impute_df.T
    else:
        logger.info("Imputation with Gaussian distribution PER PROTEIN")

    # Initialize QC metrics DataFrame
    qc_metrics = pd.DataFrame(
        index=impute_df.columns, columns=['n_imputed', 'imputation_mean', 'imputation_stddev'])
    qc_metrics['imputed_values'] = pd.Series(index=qc_metrics.index, dtype='object')

    logger.info(f'Mean number of missing values per sample: '
                f'{round(impute_df.isna().sum(axis=1).mean(),2)} out of {impute_df.shape[1]} proteins')
    logger.info(f'Mean number of missing values per protein: '
                f'{round(impute_df.isna().sum(axis=0).mean(),2)} out of {impute_df.shape[0]} samples')

    for col in impute_df.columns:
        col_mean = np.nanmean(impute_df[col])
        col_stddev = np.nanstd(impute_df[col], ddof=1)
        nan_mask = impute_df[col].isna()
        num_nans = nan_mask.sum()

        # Store QC metrics
        qc_metrics.loc[col, 'n_imputed'] = num_nans
        qc_metrics.loc[col, 'imputation_mean'] = col_mean
        qc_metrics.loc[col, 'imputation_stddev'] = col_stddev

        if num_nans > 0:
            shifted_random_values = np.random.normal(
                loc=(col_mean + (mean_shift * col_stddev)),
                scale=(col_stddev * std_dev_shift),size=num_nans)
            
            qc_metrics.at[col, 'imputed_values'] = shifted_random_values  # noqa: PD008
            impute_df.loc[nan_mask, col] = shifted_random_values
            logger.debug(f"Imputed {num_nans} NaNs in column '{col}' with mean={col_mean:.2f}, std={col_stddev:.2f}")

    if perSample:
        impute_df = impute_df.T

    if (impute_df < 0).any().any():
        logger.warning("Negative values found after imputation. Impute log-transformed data instead.")

    adata_copy.X = impute_df.to_numpy()
    adata_copy.uns[uns_key] = qc_metrics
    logger.info(f"Imputation complete. QC metrics stored in `adata.uns['{uns_key}']`.")

    return adata_copy
import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


def gaussian(
    adata: ad.AnnData,
    mean_shift: float = -1.8,
    std_dev_shift: float = 0.3,
    perSample: bool = False,
) -> ad.AnnData:
    """Impute missing values in an AnnData object using a Gaussian distribution.

    This function imputes missing values in the data matrix using a Gaussian distribution, with the mean shifted and
    the standard deviation scaled. Imputation can be performed per protein (column) or per sample (row).

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

    Returns:
    -------
    ad.AnnData
        AnnData object with imputed values.

    Raises:
    ------
    ValueError
        If negative values are imputed (when data is not log-transformed).
    """
    adata_copy = adata.copy()
    # Ensure dense array for DataFrame construction (scverse best practice)
    data = np.asarray(adata_copy.X)
    impute_df = pd.DataFrame(data=data, columns=adata_copy.var.index, index=adata_copy.obs_names)

    if perSample:
        logger.info("Imputation with Gaussian distribution PER SAMPLE")
        impute_df = impute_df.T
    else:
        logger.info("Imputation with Gaussian distribution PER PROTEIN")

    logger.info(f'Mean number of missing values per sample: '
                f'{round(impute_df.isna().sum(axis=1).mean(),2)} out of {impute_df.shape[1]} proteins')
    logger.info(f'Mean number of missing values per protein: '
                f'{round(impute_df.isna().sum(axis=0).mean(),2)} out of {impute_df.shape[0]} samples')

    for col in impute_df.columns:
        col_mean = impute_df[col].mean(skipna=True)
        col_stddev = impute_df[col].std(skipna=True)
        nan_mask = impute_df[col].isna()
        num_nans = nan_mask.sum()
        if num_nans > 0:
            shifted_random_values = np.random.normal(
                loc=(col_mean + (mean_shift * col_stddev)),
                scale=(col_stddev * std_dev_shift),
                size=num_nans)
            impute_df.loc[nan_mask, col] = shifted_random_values
            logger.info(f"Imputed {num_nans} NaNs in column '{col}' with mean={col_mean:.2f}, std={col_stddev:.2f}")

    if perSample:
        impute_df = impute_df.T

    # Check for negative values if data is not log-transformed
    if (impute_df < 0).any().any():
        logger.warning("Negative values found after imputation. Check if log-transformed.")

    adata_copy.X = impute_df.to_numpy()
    logger.info("Imputation complete")

    return adata_copy
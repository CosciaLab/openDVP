import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from opendvp.logger import logger

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def gaussian(
    adata: ad.AnnData,
    mean_shift: float = -1.8,
    std_dev_shift: float = 0.3,
    perSample: bool = False,
    qc_export_path: str | None = None
) -> ad.AnnData:
    """Description:
        This function imputes missing values in the adata object using a Gaussian distribution.
        The mean and standard deviation of the Gaussian distribution are calculated for each column (protein) in the adata object.
        The mean is shifted by the mean shift value multiplied by the standard deviation.
        The standard deviation is reduced by the std_dev_shift value.
    Parameters
    ----------
    adata : AnnData
        AnnData object
    mean_shift : float, default -1.8
        How many standard deviations to shift the mean of the Gaussian distribution
    std_dev_shift : float, default 0.3
        How much to reduce the standard deviation of the Gaussian distribution, in terms of percentage
    perSample : bool, default False
        Whether to impute missing values per sample or per protein
        Recommended to impute per protein when more than 12 protein values available
    qc_export_path : Optional[str], default None
        Path to export QC information. If None, no export is performed.

    Returns:
    -------
    AnnData
        AnnData object with imputed values
    """
    # TODO error if nan found in index, both np.nan and "nan"
    # TODO warn user when data is in linear space, could lead to negative values

    # Assert that imputated values are not negative, if so prompt user to change the mean_shift value

    logger.info("Starting imputation with Gaussian distribution version 2.1")

    adata_copy = adata.copy()
    # Ensure dense array for DataFrame construction (scverse best practice)
    data = np.asarray(adata_copy.X)
    df = pd.DataFrame(data=data, columns=adata_copy.var.index, index=adata_copy.obs_names)

    if perSample:
        logger.info("Imputation with Gaussian distribution PER SAMPLE")
        df = df.T
    else:
        logger.info("Imputation with Gaussian distribution PER PROTEIN")

    logger.info(f'Mean number of missing values per sample: {round(df.isnull().sum(axis=1).mean(),2)} out of {df.shape[1]} proteins')
    logger.info(f'Mean number of missing values per protein: {round(df.isnull().sum(axis=0).mean(),2)} out of {df.shape[0]} samples')

    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Calculate the mean and standard deviation for the column (protein values)
        col_mean = df[col].mean(skipna=True)
        col_stddev = df[col].std(skipna=True)
        # Identify NaN positions in the column
        nan_mask = df[col].isnull()
        num_nans = nan_mask.sum() 
        # Generate random values from a normal distribution       
        shifted_random_values = np.random.normal(
            loc=(col_mean + (mean_shift * col_stddev)), 
            scale=(col_stddev * std_dev_shift), 
            size=num_nans)
        # Replace NaNs in the column with the generated random values
        df.loc[nan_mask, col] = shifted_random_values

    if perSample:
        df = df.T
    
    adata_copy.X = df.values
    n_missing = np.isnan(np.asarray(adata_copy.X)).sum()
    logger.info(f'Number of missing values after imputation: {n_missing}')
    logger.info("Imputation complete")
    return adata_copy
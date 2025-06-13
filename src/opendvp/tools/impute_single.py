import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from opendvp.logger import logger

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def impute_single(
    array: np.ndarray,
    mean_shift: float = -1.8,
    std_dev_shift: float = 0.3,
    report_stats: bool = True
) -> None:
    """Impute missing values.
     
    Impute for 1D array using a Gaussian distribution in log2 space and plot the result.

    Parameters
    ----------
    array : np.ndarray
        1D array of values to impute (can contain NaNs).
    mean_shift : float, default -1.8
        How many standard deviations to shift the mean of the Gaussian distribution.
    std_dev_shift : float, default 0.3
        How much to reduce the standard deviation of the Gaussian distribution, as a fraction.
    report_stats : bool, default True
        Whether to print statistics about the imputation process.

    Returns:
    -------
    None
        This function only plots and logs statistics; it does not return an imputed array.
    """
    array_log2 = np.log2(array)
    mean_log2 = np.nanmean(array_log2)
    stddev_log2 = np.nanstd(array_log2)
    nans = np.isnan(array_log2)
    num_nans = np.sum(nans)

    shifted_random_values_log2 = np.random.normal(
        loc=(mean_log2 + (mean_shift * stddev_log2)), 
        scale=(stddev_log2 * std_dev_shift), 
        size=num_nans)
    
    if report_stats:
        logger.debug(f"mean: {mean_log2}")
        logger.debug(f"stddev: {stddev_log2}")
        logger.debug(f"Coefficient of variation: {np.nanstd(array)/np.nanmean(array)}")
        logger.debug(f"Min  : {np.nanmin(array_log2)}")
        logger.debug(f"Max  : {np.nanmax(array_log2)}")

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    _,fixed_bins,_ = plt.hist(array_log2, bins=30)
    data = np.concatenate([array_log2, shifted_random_values_log2])
    groups = ['Raw'] * len(array_log2) + ['Imputed'] * len(shifted_random_values_log2)

    sns.boxplot(x=data, y=groups, ax=ax_box, palette=['b', 'r'], orient='h')
    sns.histplot(x=array_log2, bins=fixed_bins, kde=False, ax=ax_hist, color='b', alpha=0.8)
    sns.histplot(x=shifted_random_values_log2, bins=fixed_bins, kde=False, ax=ax_hist, color='r', alpha=0.5)

    ax_box.set(yticks=[])
    ax_box.set(xticks=[])
    ax_hist.set(yticks=[], ylabel="")

    plt.tight_layout()
    plt.show()
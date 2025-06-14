import time

import numpy as np

from opendvp.utils import logger

datetime = time.strftime("%Y%m%d_%H%M%S")

def nan_difference(
    array1: np.ndarray,
    array2: np.ndarray
) -> None:
    """Calculate how many NaNs do not match between two arrays.
    
    Good quality control, since this can happen.

    Parameters
    ----------
    array1 : np.ndarray
        First array to compare.
    array2 : np.ndarray
        Second array to compare. Must have the same shape as array1.

    Returns:
    -------
    None
        Prints the number and percentage of mismatched NaNs.
    """
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: array1.shape={array1.shape}, array2.shape={array2.shape}")
    total = array1.shape[0] * array1.shape[1]

    logger.info("how many nans are not matched between arrays?")
    nan_mask1 = np.isnan(array1)
    nan_mask2 = np.isnan(array2)

    #True only if True,False or False,True. True True, or False False will be False.
    mismatch = np.logical_xor(nan_mask1, nan_mask2) & np.logical_or(nan_mask1, nan_mask2)
    logger.info(f"Number of NaNs not matching: {np.sum(mismatch)}") 
    logger.info(f"{np.sum(mismatch)*100/total} % of entire table")
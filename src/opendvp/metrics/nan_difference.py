import os, sys
import pandas as pd
import numpy as np
import anndata as ad
import time
datetime = time.strftime("%Y%m%d_%H%M%S")

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def nan_difference(
    array1: np.ndarray,
    array2: np.ndarray
) -> None:
    """
    Calculate how many NaNs do not match between two arrays.
    Good quality control, since this can happen.

    Parameters
    ----------
    array1 : np.ndarray
        First array to compare.
    array2 : np.ndarray
        Second array to compare. Must have the same shape as array1.

    Returns
    -------
    None
        Prints the number and percentage of mismatched NaNs.
    """
    assert array1.shape == array2.shape
    total = array1.shape[0] * array1.shape[1]

    print("how many nans are not matched between arrays?")
    nan_mask1 = np.isnan(array1)
    nan_mask2 = np.isnan(array2)

    #True only if True,False or False,True. True True, or False False will be False.
    mismatch = np.logical_xor(nan_mask1, nan_mask2) & np.logical_or(nan_mask1, nan_mask2)
    print(f"Number of NaNs not matching: {np.sum(mismatch)}") 
    print(f"{np.sum(mismatch)*100/total} % of entire table")
import numpy as np


def nan_difference(
    array1: np.ndarray,
    array2: np.ndarray
) -> tuple[int, int]:
    """Calculate how many NaNs do not match between two arrays.
    
    Good quality control, since this can happen.

    Parameters:
    ----------
    array1 : np.ndarray
        First array to compare.
    array2 : np.ndarray
        Second array to compare. Must have the same shape as array1.

    Returns:
    -------
    tuple[int, int]
        A tuple containing:
        - The number of mismatched NaNs.
        - The total number of elements in an array.
    """
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: array1.shape={array1.shape}, array2.shape={array2.shape}")
    
    total_elements = array1.size
    nan_mask1 = np.isnan(array1)
    nan_mask2 = np.isnan(array2)

    # XOR is True only if inputs are different (one True, one False)
    mismatch_mask = np.logical_xor(nan_mask1, nan_mask2)
    mismatch_count = int(np.sum(mismatch_mask))
    
    return mismatch_count, total_elements
from typing import Union
import pandas as pd
import numpy as np

def coefficient_of_variation(
    df: pd.DataFrame,
    axis: int = 0,
    nan_policy: str = 'propagate'
) -> pd.Series:
    """
    Calculate the coefficient of variation (CV = std / mean) along a specified axis of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    axis : int, default 0
        0 for column-wise CV, 1 for row-wise CV.
    nan_policy : {'propagate', 'raise', 'omit'}, default 'propagate'
        - 'propagate': returns NaN if NaN is present
        - 'raise': raises ValueError if NaN is present
        - 'omit': ignores NaNs in the calculation

    Returns
    -------
    pandas.Series
        CV values for each row or column.

    Raises
    ------
    ValueError
        If nan_policy='raise' and NaNs are present
    """
    if nan_policy not in {'propagate', 'raise', 'omit'}:
        raise ValueError("nan_policy must be 'propagate', 'raise', or 'omit'")

    if nan_policy == 'raise' and df.isna().any().any():
        raise ValueError("NaN values found in DataFrame and nan_policy is set to 'raise'")

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    # Map axis to pandas-accepted values (0, 1, 'index', 'columns')
    axis_label = 'index' if axis == 0 else 'columns'

    if nan_policy == 'omit':
        mean = df.mean(axis=axis_label, skipna=True)
        std = df.std(axis=axis_label, skipna=True)
    else:  # 'propagate'
        mean = df.mean(axis=axis_label, skipna=False)
        std = df.std(axis=axis_label, skipna=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std / mean
        cv.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cv
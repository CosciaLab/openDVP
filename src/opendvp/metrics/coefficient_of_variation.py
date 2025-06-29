import numpy as np
import pandas as pd


def coefficient_of_variation(
    df: pd.DataFrame, 
    axis: int = 0, 
    nan_policy: str = "propagate", 
    ddof: int = 1
) -> pd.Series:
    """Calculate the coefficient of variation.
     
    (CV = std / mean) along a specified axis of a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    axis : int, default 0
        0 for column-wise CV, 1 for row-wise CV.
    nan_policy : {'propagate', 'raise', 'omit'}, default 'propagate'
        - 'propagate': returns NaN if NaN is present
        - 'raise': raises ValueError if NaN is present
        - 'omit': ignores NaNs in the calculation
    ddof : int, default 1
        Delta Degrees of Freedom used in the std calculation. 
        The divisor used in calculations is N - ddof,
        default is 1
        

    Returns:
    -------
    pandas.Series
        CV values for each row or column.

    Raises:
    ------
    ValueError
        If nan_policy='raise' and NaNs are present
    """
    if nan_policy not in {"propagate", "raise", "omit"}:
        raise ValueError("nan_policy must be 'propagate', 'raise', or 'omit'")
    if nan_policy == "raise" and df.isna().any().any():
        raise ValueError("NaN values found in DataFrame and nan_policy is set to 'raise'")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    if nan_policy == "omit":
        mean = df.mean(axis=axis, skipna=True)
        std = df.std(axis=axis, skipna=True, ddof=ddof)
    else:  # 'propagate'
        mean = df.mean(axis=axis, skipna=False)
        std = df.std(axis=axis, skipna=False, ddof=ddof)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean

    return cv

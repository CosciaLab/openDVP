import numpy as np
import pytest

from opendvp.metrics.nan_difference import nan_difference


def test_basic_mismatch():
    """Test with arrays that have mismatched NaNs."""
    array1 = np.array([[np.nan, 2.0], [3.0, 4.0]])
    array2 = np.array([[1.0, np.nan], [3.0, 4.0]])

    mismatches, total = nan_difference(array1, array2)

    assert mismatches == 2
    assert total == 4


def test_no_mismatch_with_nans():
    """Test with arrays where NaNs are in the same positions."""
    array1 = np.array([[np.nan, 2.0], [3.0, np.nan]])
    array2 = np.array([[np.nan, 5.0], [6.0, np.nan]])

    mismatches, total = nan_difference(array1, array2)

    assert mismatches == 0
    assert total == 4


def test_no_nans():
    """Test with arrays that have no NaNs."""
    array1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    array2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    mismatches, total = nan_difference(array1, array2)

    assert mismatches == 0
    assert total == 4


def test_all_nans():
    """Test with arrays that are entirely composed of NaNs."""
    array1 = np.full((2, 2), np.nan)
    array2 = np.full((2, 2), np.nan)

    mismatches, total = nan_difference(array1, array2)

    assert mismatches == 0
    assert total == 4


def test_one_all_nans_one_some_nans():
    """Test with one array of all NaNs and another with some NaNs."""
    array1 = np.full((2, 2), np.nan)
    array2 = np.array([[1.0, np.nan], [np.nan, 4.0]])

    mismatches, total = nan_difference(array1, array2)

    # Mismatches are at (0,0) and (1,1)
    assert mismatches == 2
    assert total == 4


def test_shape_mismatch_raises_error():
    """Test that a ValueError is raised for arrays with different shapes."""
    array1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    array2 = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        nan_difference(array1, array2)
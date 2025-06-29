import numpy as np
import pandas as pd
import pytest

from opendvp.metrics.coefficient_of_variation import coefficient_of_variation


@pytest.fixture
def sample_dataframe():
    """A sample DataFrame for testing CV calculation."""
    data = {
        'col1': [10, 20, 30, 40, 50, 60],
        'col2': [1, 2, 3, 4, 5, 6],
        'col3': [100, 100, 100, 100, 100, 100],  # Zero CV
        'col4': [0, 0, 0, 0, 0, 0],  # Zero mean, should result in NaN CV
        'col5': [1, 1, 1, 1, 1, 1]   # Zero CV
    }
    return pd.DataFrame(data)


@pytest.fixture
def dataframe_with_nans():
    """A DataFrame with NaN values for testing nan_policy."""
    data = {
        'colA': [10, 20, np.nan, 40, 50],
        'colB': [1, np.nan, 3, 4, 5],
        'colC': [np.nan, np.nan, np.nan, np.nan, np.nan],  # All NaNs
        'colD': [100, 100, 100, 100, np.nan]
    }
    return pd.DataFrame(data)


def test_basic_cv_calculation_axis_0(sample_dataframe):
    """Test CV calculation column-wise (axis=0)."""
    expected_cv = pd.Series(
        [
            sample_dataframe['col1'].std() / sample_dataframe['col1'].mean(),
            sample_dataframe['col2'].std() / sample_dataframe['col2'].mean(),
            0.0,  # std=0, mean=100
            np.nan,  # std=0, mean=0 -> 0/0 = NaN
            0.0   # std=0, mean=1
        ],
        index=['col1', 'col2', 'col3', 'col4', 'col5']
    )
    result_cv = coefficient_of_variation(sample_dataframe, axis=0)
    pd.testing.assert_series_equal(result_cv, expected_cv, check_dtype=False, check_exact=False, atol=1e-7)


def test_basic_cv_calculation_axis_1(sample_dataframe):
    """Test CV calculation row-wise (axis=1)."""
    result_cv = coefficient_of_variation(sample_dataframe, axis=1)

    # Test a few specific rows
    # Row 0: [10, 1, 100, 0, 1]
    assert result_cv.iloc[0] == pytest.approx(np.std([10, 1, 100, 0, 1], ddof=1) / np.mean([10, 1, 100, 0, 1]))
    # Row 2: [30, 3, 100, 0, 1]
    assert result_cv.iloc[2] == pytest.approx(np.std([30, 3, 100, 0, 1], ddof=1) / np.mean([30, 3, 100, 0, 1]))
    # Last row: [60, 6, 100, 0, 1] - should be a valid CV
    last_row_values = sample_dataframe.iloc[-1].to_numpy()
    expected_last_row_cv = np.std(last_row_values, ddof=1) / np.mean(last_row_values)
    assert result_cv.iloc[-1] == pytest.approx(expected_last_row_cv)


def test_nan_policy_propagate(dataframe_with_nans):
    """Test nan_policy='propagate' ensures NaNs in input propagate to output."""
    result_cv = coefficient_of_variation(dataframe_with_nans, nan_policy='propagate')
    # If any NaN is present in a column, mean/std will be NaN, thus CV will be NaN
    assert np.isnan(result_cv['colA'])
    assert np.isnan(result_cv['colB'])
    assert np.isnan(result_cv['colC'])
    assert np.isnan(result_cv['colD'])


def test_nan_policy_raise(dataframe_with_nans):
    """Test nan_policy='raise' raises ValueError when NaNs are present."""
    with pytest.raises(ValueError, match="NaN values found in DataFrame and nan_policy is set to 'raise'"):
        coefficient_of_variation(dataframe_with_nans, nan_policy='raise')


def test_nan_policy_raise_no_nans(sample_dataframe):
    """Test nan_policy='raise' works correctly when no NaNs are present."""
    # This should not raise an error and should produce the same result as 'propagate'
    expected_cv = pd.Series(
        [
            sample_dataframe['col1'].std() / sample_dataframe['col1'].mean(),
            sample_dataframe['col2'].std() / sample_dataframe['col2'].mean(),
            0.0,
            np.nan,
            0.0
        ],
        index=['col1', 'col2', 'col3', 'col4', 'col5']
    )
    result_cv = coefficient_of_variation(sample_dataframe, nan_policy='raise')
    pd.testing.assert_series_equal(result_cv, expected_cv, check_dtype=False, check_exact=False, atol=1e-7)


def test_nan_policy_omit(dataframe_with_nans):
    """Test nan_policy='omit' ignores NaNs in calculations."""
    expected_cv = pd.Series(
        [
            np.std([10, 20, 40, 50], ddof=1) / np.mean([10, 20, 40, 50]),  # colA
            np.std([1, 3, 4, 5], ddof=1) / np.mean([1, 3, 4, 5]),          # colB
            np.nan,  # colC is all NaNs, so mean/std will be NaN even with skipna=True
            0.0      # colD: std([100,100,100,100])=0, mean=100 -> 0.0
        ],
        index=['colA', 'colB', 'colC', 'colD']
    )
    result_cv = coefficient_of_variation(dataframe_with_nans, nan_policy='omit')
    pd.testing.assert_series_equal(result_cv, expected_cv, check_dtype=False, check_exact=False, atol=1e-7)


def test_invalid_nan_policy():
    """Test that an invalid nan_policy string raises ValueError."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(ValueError, match="nan_policy must be 'propagate', 'raise', or 'omit'"):
        coefficient_of_variation(df, nan_policy='invalid_policy')


def test_invalid_axis():
    """Test that an invalid axis value raises ValueError."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(ValueError, match="axis must be 0 \\(columns\\) or 1 \\(rows\\)"):
        coefficient_of_variation(df, axis=2)
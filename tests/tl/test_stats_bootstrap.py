import numpy as np
import pandas as pd
import pytest

from opendvp.tl.stats_bootstrap import stats_bootstrap


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Creates a sample DataFrame for bootstrap testing."""
    rng = np.random.default_rng(42)
    # 100 samples, 4 features
    data = {
        "low_variance": rng.normal(loc=100, scale=1, size=100),
        "high_variance": rng.normal(loc=10, scale=5, size=100),
        "with_nans": rng.normal(loc=50, scale=10, size=100),
        "constant": np.full(100, 50.0),
    }
    df = pd.DataFrame(data)
    df.loc[10:20, "with_nans"] = np.nan
    return df


def test_return_types_and_shapes(sample_df: pd.DataFrame) -> None:
    """Test default return behavior (summary only) and output shapes."""
    n_bootstrap = 10
    subset_sizes = [10, 20]
    summary_df = stats_bootstrap(
        sample_df, n_bootstrap=n_bootstrap, subset_sizes=subset_sizes, return_summary=True, return_raw=False, plot=False
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape[0] == len(subset_sizes) * sample_df.shape[1]
    assert list(summary_df.columns) == ["subset_size", "feature", "cv_summary"]


def test_return_both_dataframes(sample_df: pd.DataFrame) -> None:
    """Test returning both raw and summary dataframes."""
    n_bootstrap = 10
    subset_sizes = [10, 20]
    raw_df, summary_df = stats_bootstrap(
        sample_df, n_bootstrap=n_bootstrap, 
        subset_sizes=subset_sizes, return_raw=True, 
        return_summary=True, plot=False
    ) # type: ignore

    assert isinstance(raw_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert raw_df.shape[0] == len(subset_sizes) * n_bootstrap * sample_df.shape[1]
    assert list(raw_df.columns) == ["feature", "cv", "subset_size", "bootstrap_id"]
    assert summary_df.shape[0] == len(subset_sizes) * sample_df.shape[1]


def test_return_raw_only(sample_df: pd.DataFrame) -> None:
    """Test returning only the raw results."""
    raw_df = stats_bootstrap(sample_df, return_raw=True, return_summary=False, plot=False)
    assert isinstance(raw_df, pd.DataFrame)
    assert "cv_summary" not in raw_df.columns


def test_return_none(sample_df: pd.DataFrame) -> None:
    """Test returning None when both flags are False."""
    result = stats_bootstrap(sample_df, return_raw=False, return_summary=False, plot=False)
    assert result is None


def test_reproducibility_with_seed(sample_df: pd.DataFrame) -> None:
    """Test that the same random_seed produces identical results."""
    summary1 = stats_bootstrap(sample_df, random_seed=42, plot=False)
    summary2 = stats_bootstrap(sample_df, random_seed=42, plot=False)
    summary3 = stats_bootstrap(sample_df, random_seed=123, plot=False)

    pd.testing.assert_frame_equal(summary1, summary2)
    assert not summary1.equals(summary3)


def test_summary_func_count_above_threshold(sample_df: pd.DataFrame) -> None:
    """Test the 'count_above_threshold' summary function."""
    cv_thresh = 0.05
    summary_df = stats_bootstrap(
        sample_df, summary_func="count_above_threshold", cv_threshold=cv_thresh, plot=False
    )

    assert "cv_count_above_threshold" in summary_df.columns
    # The CV for 'high_variance' should be high, so the count should be > 0
    assert summary_df.query("feature == 'high_variance'")["cv_count_above_threshold"].iloc[0] > 0
    # The CV for 'low_variance' should be low, so the count should be 0
    assert summary_df.query("feature == 'low_variance'")["cv_count_above_threshold"].iloc[0] == 0


def test_error_on_missing_cv_threshold(sample_df: pd.DataFrame) -> None:
    """Test ValueError when 'count_above_threshold' is used without cv_threshold."""
    with pytest.raises(ValueError, match="cv_threshold must be set"):
        stats_bootstrap(sample_df, summary_func="count_above_threshold", cv_threshold=None, plot=False)


def test_subsampling_vs_bootstrapping(sample_df: pd.DataFrame) -> None:
    """Test that sampling with and without replacement produce different results."""
    summary_replace = stats_bootstrap(sample_df, replace=True, random_seed=42, plot=False)
    summary_no_replace = stats_bootstrap(sample_df, replace=False, random_seed=42, plot=False)

    assert not summary_replace.equals(summary_no_replace)


def test_error_on_invalid_subset_size_for_subsampling(sample_df: pd.DataFrame) -> None:
    """Test ValueError when subset size is too large for sampling without replacement."""
    invalid_size = sample_df.shape[0] + 1
    with pytest.raises(ValueError, match="A subset size is larger than the number of rows"):
        stats_bootstrap(sample_df, subset_sizes=[invalid_size], replace=False, plot=False)


def test_nan_policy_omit(sample_df: pd.DataFrame) -> None:
    """Test that nan_policy='omit' runs and produces valid output for columns with NaNs."""
    summary_df = stats_bootstrap(sample_df, nan_policy="omit", plot=False)
    # The 'with_nans' feature should have a valid, non-NaN CV summary
    cv_summary_with_nans = summary_df.query("feature == 'with_nans'")["cv_summary"].iloc[0]
    assert pd.notna(cv_summary_with_nans)


def test_nan_policy_raise(sample_df: pd.DataFrame) -> None:
    """Test that nan_policy='raise' raises an error for columns with NaNs."""
    with pytest.raises(ValueError): # Or the specific error raised by the CV function
        stats_bootstrap(sample_df, nan_policy="raise", plot=False)


def test_constant_column_cv_is_zero(sample_df: pd.DataFrame) -> None:
    """Test that a column with constant values has a CV of zero."""
    summary_df = stats_bootstrap(sample_df, plot=False)
    cv_summary_constant = summary_df.query("feature == 'constant'")["cv_summary"].iloc[0]
    assert np.isclose(cv_summary_constant, 0.0)
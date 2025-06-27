import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.filter_by_ratio import filter_by_ratio


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create a sample AnnData object for testing."""
    n_obs = 100
    var_names = ["marker_A", "marker_B", "other_marker"]
    X = np.random.rand(n_obs, len(var_names)) * 10 # scale to get ratios > 1 sometimes
    return ad.AnnData(X=X, var=pd.DataFrame(index=var_names))


def test_filter_default_params(sample_adata):
    """Test filter_by_ratio with default parameters."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B")

    assert "DAPI_ratio" in adata_filtered.obs.columns
    assert "DAPI_ratio_pass" in adata_filtered.obs.columns
    assert "DAPI_ratio_pass_nottoolow" not in adata_filtered.obs.columns
    assert "DAPI_ratio_pass_nottoohigh" not in adata_filtered.obs.columns

    # Basic check: some cells should pass the filter (not all True or False)
    assert 0 < adata_filtered.obs["DAPI_ratio_pass"].sum() < sample_adata.n_obs


def test_filter_custom_label(sample_adata):
    """Test filter_by_ratio with a custom label."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", label="CustomRatio")
    assert "CustomRatio_ratio" in adata_filtered.obs.columns
    assert "CustomRatio_ratio_pass" in adata_filtered.obs.columns


def test_filter_custom_ratio_range(sample_adata):
    """Test filter_by_ratio with a custom ratio range."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", min_ratio=0.2, max_ratio=0.8)
    # The number of cells passing the filter should be different from the default range
    assert 0 < adata_filtered.obs["DAPI_ratio_pass"].sum() < sample_adata.n_obs


def test_filter_add_detailed(sample_adata):
    """Test filter_by_ratio with add_detailed_pass_fail=True."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", add_detailed_pass_fail=True)
    assert "DAPI_ratio_pass_nottoolow" in adata_filtered.obs.columns
    assert "DAPI_ratio_pass_nottoohigh" in adata_filtered.obs.columns

    # Check that _pass is the logical AND of _pass_nottoolow and _pass_nottoohigh
    expected_pass = (
        adata_filtered.obs["DAPI_ratio_pass_nottoolow"] &
        adata_filtered.obs["DAPI_ratio_pass_nottoohigh"]
    )
    assert (adata_filtered.obs["DAPI_ratio_pass"] == expected_pass).all()


def test_filter_division_by_zero(sample_adata):
    """Test filter_by_ratio handling division by zero."""
    # Set some "marker_B" values to 0 to force division by zero
    zero_indices = np.random.choice(sample_adata.n_obs, size=10, replace=False)
    sample_adata.X[zero_indices, 1] = 0  # Assuming marker_B is at index 1
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", add_detailed_pass_fail=True)

    # Check that ratio is NaN where start_cycle is 0
    assert np.isnan(adata_filtered.obs["DAPI_ratio"][zero_indices]).all()

    # Where ratio is nan, pass should be False
    assert (adata_filtered.obs["DAPI_ratio_pass"][zero_indices] == False).all()


def test_error_invalid_end_cycle(sample_adata):
    """Test filter_by_ratio with an invalid end_cycle marker."""
    with pytest.raises(ValueError, match="end_cycle marker 'invalid_marker' not found in adata.var_names"):
        filter_by_ratio(sample_adata, "invalid_marker", "marker_B")


def test_error_invalid_start_cycle(sample_adata):
    """Test filter_by_ratio with an invalid start_cycle marker."""
    with pytest.raises(ValueError, match="start_cycle marker 'invalid_marker' not found in adata.var_names"):
        filter_by_ratio(sample_adata, "marker_A", "invalid_marker")


def test_error_min_ratio_greater_than_max(sample_adata):
    """Test ValueError when min_ratio >= max_ratio."""
    with pytest.raises(ValueError, match="min_ratio must be less than max_ratio"):
        filter_by_ratio(sample_adata, "marker_A", "marker_B", min_ratio=1.0, max_ratio=0.5)


def test_returns_copy_not_view(sample_adata):
    """Test that the function returns a copy, not a view, of the AnnData object."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B")
    assert adata_filtered is not sample_adata

    # Modify the copy and check if the original is unchanged
    adata_filtered.obs["DAPI_ratio"].iloc[0] = -999
    assert sample_adata.obs.get("DAPI_ratio", None) is None


def test_no_cells_pass_filter(sample_adata):
    """Test scenario where no cells pass the filter (very strict range)."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", min_ratio=1000, max_ratio=2000)
    assert (adata_filtered.obs["DAPI_ratio_pass"] == False).all()


def test_all_cells_pass_filter(sample_adata):
    """Test scenario where all cells pass the filter (very permissive range)."""
    adata_filtered = filter_by_ratio(sample_adata, "marker_A", "marker_B", min_ratio=-1000, max_ratio=1000)
    assert (adata_filtered.obs["DAPI_ratio_pass"] == True).all()
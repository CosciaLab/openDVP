import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.filter_by_abs_value import filter_by_abs_value


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create a sample AnnData object for testing filter_by_abs_value.

    Includes data in X and obs, with one feature name duplicated to test ambiguity.
    """
    n_obs = 100
    n_vars = 5
    rng = np.random.default_rng(42)  # For reproducibility

    # Create X data (markers)
    X = rng.normal(loc=10, scale=2, size=(n_obs, n_vars))
    var_names = [f"marker_{i}" for i in range(n_vars)]
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=var_names))

    # Add obs data (continuous variables)
    adata.obs["cell_size"] = rng.normal(loc=50, scale=10, size=n_obs)
    adata.obs["cell_density"] = rng.uniform(0.1, 1.0, size=n_obs)
    adata.obs["patient_id"] = [f"P{i // 50}" for i in range(n_obs)]  # Non-numeric for type check
    
    # Deliberate duplicate name to test ambiguity error
    # This 'marker_0' in obs will conflict with 'marker_0' in var_names
    adata.obs["marker_0"] = rng.normal(loc=10, scale=2, size=n_obs) 

    return adata


# --- Test cases for feature in adata.X ---

def test_filter_x_absolute_lower_bound(sample_adata):
    """Test filtering a feature from adata.X with an absolute lower bound."""
    feature_name = "marker_1"
    lower_bound = 10.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, mode='absolute')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col]
    
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    # All cells marked True must meet the condition
    assert (original_data[mask] >= lower_bound).all()
    # No cells marked False should meet the condition
    assert not (original_data[~mask] >= lower_bound).any()
    # Ensure some filtering happened (not all True or all False)
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_x_absolute_upper_bound(sample_adata):
    """Test filtering a feature from adata.X with an absolute upper bound."""
    feature_name = "marker_2"
    upper_bound = 9.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, upper_bound=upper_bound, mode='absolute')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col] # type: ignore
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    assert (original_data[mask] <= upper_bound).all()
    assert not (original_data[~mask] <= upper_bound).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_x_absolute_range(sample_adata):
    """Test filtering a feature from adata.X with an absolute range."""
    feature_name = "marker_3"
    lower_bound = 9.0
    upper_bound = 11.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, upper_bound=upper_bound, mode='absolute')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col] # type: ignore
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    assert ((original_data[mask] >= lower_bound) & (original_data[mask] <= upper_bound)).all()
    assert not ((original_data[~mask] >= lower_bound) & (original_data[~mask] <= upper_bound)).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_x_quantile_lower_bound(sample_adata):
    """Test filtering a feature from adata.X with a quantile lower bound."""
    feature_name = "marker_4"
    lower_bound = 0.25
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, mode='quantile')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col] # type: ignore
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    actual_lower_threshold = original_data.quantile(lower_bound)
    assert (original_data[mask] >= actual_lower_threshold).all()
    assert not (original_data[~mask] >= actual_lower_threshold).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_x_quantile_upper_bound(sample_adata):
    """Test filtering a feature from adata.X with a quantile upper bound."""
    feature_name = "marker_1" # Re-use marker_1
    upper_bound = 0.75
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, upper_bound=upper_bound, mode='quantile')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col] # type: ignore
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    actual_upper_threshold = original_data.quantile(upper_bound)
    assert (original_data[mask] <= actual_upper_threshold).all()
    assert not (original_data[~mask] <= actual_upper_threshold).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_x_quantile_range(sample_adata):
    """Test filtering a feature from adata.X with a quantile range."""
    feature_name = "marker_2" # Re-use marker_2
    lower_bound = 0.1
    upper_bound = 0.9
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, upper_bound=upper_bound, mode='quantile')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col] # type: ignore
    original_data = pd.Series(sample_adata[:, feature_name].X.flatten(), index=sample_adata.obs_names)
    
    actual_lower_threshold = original_data.quantile(lower_bound)
    actual_upper_threshold = original_data.quantile(upper_bound)
    
    assert ((original_data[mask] >= actual_lower_threshold) & (original_data[mask] <= actual_upper_threshold)).all()
    assert not ((original_data[~mask] >= actual_lower_threshold) & (original_data[~mask] <= actual_upper_threshold)).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs


# --- Test cases for feature in adata.obs ---

def test_filter_obs_absolute_lower_bound(sample_adata):
    """Test filtering a feature from adata.obs with an absolute lower bound."""
    feature_name = "cell_size"
    lower_bound = 55.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, mode='absolute')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col]
    original_data = sample_adata.obs[feature_name]
    
    assert (original_data[mask] >= lower_bound).all()
    assert not (original_data[~mask] >= lower_bound).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs

def test_filter_obs_quantile_range(sample_adata):
    """Test filtering a feature from adata.obs with a quantile range."""
    feature_name = "cell_density"
    lower_bound = 0.2
    upper_bound = 0.8
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, upper_bound=upper_bound, mode='quantile')
    
    filter_col = f"{feature_name}_filter"
    assert filter_col in adata_filtered.obs.columns
    mask = adata_filtered.obs[filter_col]
    original_data = sample_adata.obs[feature_name]
    
    actual_lower_threshold = original_data.quantile(lower_bound)
    actual_upper_threshold = original_data.quantile(upper_bound)
    
    assert ((original_data[mask] >= actual_lower_threshold) & (original_data[mask] <= actual_upper_threshold)).all()
    assert not ((original_data[~mask] >= actual_lower_threshold) & (original_data[~mask] <= actual_upper_threshold)).any()
    assert mask.sum() > 0 and mask.sum() < sample_adata.n_obs


# --- Error handling tests ---

def test_error_feature_not_found(sample_adata):
    """Test ValueError when feature_name is not found in X or obs."""
    with pytest.raises(ValueError, match="Feature 'non_existent_feature' not found in either adata.var_names or adata.obs.columns."):
        filter_by_abs_value(sample_adata, "non_existent_feature", lower_bound=10)

def test_error_feature_ambiguity(sample_adata):
    """Test ValueError when feature_name exists in both X and obs."""
    # 'marker_0' exists in both X and obs in the sample_adata fixture
    with pytest.raises(ValueError, match="Feature 'marker_0' found in both adata.var_names and adata.obs.columns."):
        filter_by_abs_value(sample_adata, "marker_0", lower_bound=10)

def test_error_no_bounds_provided(sample_adata):
    """Test ValueError when neither lower_bound nor upper_bound is provided."""
    with pytest.raises(ValueError, match="At least one of 'lower_bound' or 'upper_bound' must be provided."):
        filter_by_abs_value(sample_adata, "marker_1")

def test_error_lower_bound_greater_than_upper_bound(sample_adata):
    """Test ValueError when lower_bound is greater than upper_bound."""
    with pytest.raises(ValueError, match="'lower_bound' cannot be greater than 'upper_bound'."):
        filter_by_abs_value(sample_adata, "marker_1", lower_bound=10, upper_bound=5)

def test_error_obs_feature_not_numeric(sample_adata):
    """Test ValueError when an obs feature is used for filtering but is not numeric."""
    with pytest.raises(ValueError, match="Feature 'patient_id' in adata.obs is not numeric."):
        filter_by_abs_value(sample_adata, "patient_id", lower_bound=1, mode='absolute')

def test_error_quantile_bound_out_of_range_lower(sample_adata):
    """Test ValueError for quantile lower bound outside [0, 1]."""
    with pytest.raises(ValueError, match="For 'quantile' mode, 'lower_bound' must be between 0 and 1"):
        filter_by_abs_value(sample_adata, "marker_1", lower_bound=-0.1, mode='quantile')
    with pytest.raises(ValueError, match="For 'quantile' mode, 'lower_bound' must be between 0 and 1"):
        filter_by_abs_value(sample_adata, "marker_1", lower_bound=1.1, mode='quantile')

def test_error_quantile_bound_out_of_range_upper(sample_adata):
    """Test ValueError for quantile upper bound outside [0, 1]."""
    with pytest.raises(ValueError, match="For 'quantile' mode, 'upper_bound' must be between 0 and 1"):
        filter_by_abs_value(sample_adata, "marker_1", upper_bound=-0.1, mode='quantile')
    with pytest.raises(ValueError, match="For 'quantile' mode, 'upper_bound' must be between 0 and 1"):
        filter_by_abs_value(sample_adata, "marker_1", upper_bound=1.1, mode='quantile')

# --- Test return value properties and edge filtering results ---

def test_returns_copy_not_view(sample_adata):
    """Test that the function returns a copy of AnnData, not a view."""
    feature_name = "marker_1"
    lower_bound = 10.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, mode='absolute')
    
    assert adata_filtered is not sample_adata # Ensure it's a distinct object
    
    # Modify the copy and ensure original is unchanged
    filter_col = f"{feature_name}_filter"
    adata_filtered.obs[filter_col].iloc[0] = False
    # The original adata should not have the new column yet
    assert filter_col not in sample_adata.obs.columns

def test_new_obs_column_name_format(sample_adata):
    """Test that the new column in .obs has the expected name format."""
    feature_name = "marker_2"
    lower_bound = 10.0
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=lower_bound, mode='absolute')
    assert f"{feature_name}_filter" in adata_filtered.obs.columns

def test_all_cells_filtered_out(sample_adata):
    """Test scenario where all cells are filtered out."""
    feature_name = "marker_3"
    # Set bounds so tight that no cells pass (e.g., outside the data's typical range)
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=1000, upper_bound=1001, mode='absolute')
    mask = adata_filtered.obs[f"{feature_name}_filter"]
    assert mask.sum() == 0 # Expect 0 cells to pass

def test_no_cells_filtered_out(sample_adata):
    """Test scenario where no cells are filtered out (all cells pass)."""
    feature_name = "marker_4"
    # Set bounds so wide that all cells pass
    adata_filtered = filter_by_abs_value(sample_adata, feature_name, lower_bound=-1000, upper_bound=1000, mode='absolute')
    mask = adata_filtered.obs[f"{feature_name}_filter"]
    assert mask.sum() == sample_adata.n_obs # Expect all cells to pass
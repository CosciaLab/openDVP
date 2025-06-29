import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.filter_adata_by_gates import filter_adata_by_gates


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create a sample AnnData object for testing."""
    n_obs = 10
    var_names = [
        "mean_Vimentin", "mean_CD3e", "mean_panCK", "mean_CD8",
        "mean_COL1A1", "mean_CD20", "mean_CD68", "mean_Ki67",
        "extra_marker_1", "extra_marker_2"
    ]
    X = np.random.rand(n_obs, len(var_names))
    return ad.AnnData(X=X, var=pd.DataFrame(index=var_names))


@pytest.fixture
def gates_df() -> pd.DataFrame:
    """Load the sample gates DataFrame from the provided CSV data."""
    return pd.DataFrame({
        'sample_id': [991, 991, 991, 991, 991, 991, 991, 991],
        'marker_id': [
            'mean_Vimentin', 'mean_CD3e', 'mean_panCK', 'mean_CD8',
            'mean_COL1A1', 'mean_CD20', 'mean_CD68', 'mean_Ki67'
        ],
        'gate_value': [574.38, 350, 50, 1200, 1320, 800, 271.58, 18.10]
    })


def test_filter_without_sample_id(sample_adata, gates_df):
    """Test filtering without specifying a sample_id."""
    adata_filtered = filter_adata_by_gates(sample_adata, gates_df)
    
    assert adata_filtered.n_vars == 8
    assert adata_filtered.shape == (sample_adata.n_obs, 8)
    expected_markers = sorted(gates_df['marker_id'].unique())
    assert sorted(list(adata_filtered.var_names)) == expected_markers


def test_filter_with_valid_sample_id(sample_adata, gates_df):
    """Test filtering with a valid sample_id."""
    # Add another sample to the gates to ensure filtering is happening
    other_sample_gates = pd.DataFrame({
        'sample_id': [992, 992],
        'marker_id': ['extra_marker_1', 'extra_marker_2'],
        'gate_value': [100, 200]
    })
    multi_sample_gates = pd.concat([gates_df, other_sample_gates], ignore_index=True)

    adata_filtered = filter_adata_by_gates(sample_adata, multi_sample_gates, sample_id=991)

    assert adata_filtered.n_vars == 8
    expected_markers = sorted(gates_df['marker_id'].unique())
    assert sorted(list(adata_filtered.var_names)) == expected_markers


def test_error_on_missing_marker_id_column(sample_adata, gates_df):
    """Test that a ValueError is raised if 'marker_id' column is missing."""
    invalid_gates = gates_df.drop(columns=['marker_id'])
    with pytest.raises(ValueError, match="must contain a 'marker_id' column"):
        filter_adata_by_gates(sample_adata, invalid_gates)


def test_error_on_missing_sample_id_column_when_specified(sample_adata, gates_df):
    """Test ValueError if sample_id is provided but column is missing in gates."""
    gates_no_sample = gates_df.drop(columns=['sample_id'])
    with pytest.raises(ValueError, match="`sample_id` column is missing in `gates`"):
        filter_adata_by_gates(sample_adata, gates_no_sample, sample_id=991)


def test_error_on_nonexistent_sample_id(sample_adata, gates_df):
    """Test ValueError if the specified sample_id is not in the gates DataFrame."""
    with pytest.raises(ValueError, match="No markers found in gates for sample_id: 12345"):
        filter_adata_by_gates(sample_adata, gates_df, sample_id=12345)


def test_error_on_markers_not_in_adata(sample_adata, gates_df):
    """Test ValueError if gates contain markers not present in the AnnData object."""
    extra_marker_gate = pd.DataFrame([{'marker_id': 'non_existent_marker', 'sample_id': 991}])
    invalid_gates = pd.concat([gates_df, extra_marker_gate], ignore_index=True)
    
    with pytest.raises(ValueError, match="Markers not found in adata.var_names:.*'non_existent_marker'"):
        filter_adata_by_gates(sample_adata, invalid_gates, sample_id=991)


def test_handles_duplicate_markers_in_gates(sample_adata, gates_df):
    """Test that duplicate markers in the gates file are handled correctly."""
    duplicate_gate = pd.DataFrame([{'marker_id': 'mean_Vimentin', 'sample_id': 991}])
    gates_with_duplicates = pd.concat([gates_df, duplicate_gate], ignore_index=True)

    assert len(gates_with_duplicates) == len(gates_df) + 1

    adata_filtered = filter_adata_by_gates(sample_adata, gates_with_duplicates, sample_id=991)

    # The number of variables should be based on unique markers
    assert adata_filtered.n_vars == 8
    assert 'mean_Vimentin' in adata_filtered.var_names


def test_returns_copy(sample_adata, gates_df):
    """Test that the function returns a copy, not a view, of the AnnData object."""
    adata_filtered = filter_adata_by_gates(sample_adata, gates_df)
    assert adata_filtered is not sample_adata
    
    # Modify the copy and check if the original is unchanged
    adata_filtered.X[0, 0] = -999
    assert sample_adata.X[0, 0] != -999
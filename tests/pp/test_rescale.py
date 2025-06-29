import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.pp import rescale


@pytest.fixture
def rescale_adata() -> ad.AnnData:
    """Creates a sample AnnData object for rescale tests.
    
    - 100 cells, 50 from 'image1' and 50 from 'image2'.
    - 'marker_A': Bimodal distribution, clearly separable.
    - 'marker_B': Unimodal distribution.
    - 'marker_C': A failed marker candidate.
    """
    np.random.seed(0)
    # Image 1 data
    X1_A = np.concatenate([np.random.normal(2, 0.5, 25), np.random.normal(8, 0.5, 25)])
    X1_B = np.random.normal(5, 1, 50)
    X1_C = np.random.uniform(0, 1, 50)

    # Image 2 data
    X2_A = np.concatenate(
        [np.random.normal(2.5, 0.5, 25), np.random.normal(8.5, 0.5, 25)]
    )
    X2_B = np.random.normal(5.5, 1, 50)
    X2_C = np.zeros(50)  # All zero, like a failed marker

    X = np.vstack(
        [np.column_stack([X1_A, X1_B, X1_C]), np.column_stack([X2_A, X2_B, X2_C])]
    )

    obs = pd.DataFrame(
        {"imageid": ["image1"] * 50 + ["image2"] * 50},
        index=[f"cell_{i}" for i in range(100)],
    )
    var = pd.DataFrame(index=["marker_A", "marker_B", "marker_C"])

    return ad.AnnData(X, obs=obs, var=var)


def test_rescale_with_gmm(rescale_adata: ad.AnnData):
    """Test rescaling with automatic GMM gating."""
    adata = rescale_adata.copy()
    adata_original_X = adata.X.copy()

    adata_rescaled = rescale(adata, gate=None, log=False, verbose=False)
    
    # 1. Check if adata is modified in place and raw is created
    assert adata_rescaled is adata
    assert adata.raw is not None
    np.testing.assert_array_equal(adata.raw.X, adata_original_X)

    # 2. Check that X is modified and values are within [0, 1]
    assert not np.array_equal(adata.X, adata_original_X)
    assert np.all(adata.X >= 0) and np.all(adata.X <= 1)

    # 3. Check that gates were stored in uns
    assert "gates" in adata.uns
    gates_df = adata.uns["gates"]
    assert isinstance(gates_df, pd.DataFrame)
    
    # Check that gates were determined for marker_A and marker_B (not NaN)
    # marker_C might be all zeros, leading to a max gate of 0, which is fine.
    assert not pd.isna(gates_df.loc["marker_A", "image1"])
    assert not pd.isna(gates_df.loc["marker_B", "image1"])
    assert not pd.isna(gates_df.loc["marker_C", "image1"])


def test_rescale_with_manual_gates(rescale_adata: ad.AnnData):
    """Test rescaling with a provided manual gate DataFrame."""
    adata = rescale_adata.copy()
    manual_gates = pd.DataFrame({"markers": ["marker_A", "marker_B"], "gates": [5.0, 9.0]})

    adata_rescaled = rescale(adata, gate=manual_gates, log=False, verbose=False)

    gates_df = adata_rescaled.uns["gates"]
    assert gates_df.loc["marker_A", "image1"] == 5.0
    assert gates_df.loc["marker_B", "image1"] == 9.0
    
    # Check that X is modified and values are within [0, 1]
    assert not np.array_equal(adata.X, adata_rescaled.raw.X)
    assert np.all(adata.X >= 0) and np.all(adata.X <= 1)

    # Check that other markers (like C) also have gates (potentially GMM-derived)
    assert not pd.isna(gates_df.loc["marker_C", "image1"])
    

def test_rescale_with_failed_markers(rescale_adata: ad.AnnData):
    """Test handling of failed markers."""
    adata = rescale_adata.copy()
    failed_markers = {"image1": ["marker_C"]}
    adata_rescaled = rescale(adata, failed_markers=failed_markers, log=True, verbose=False)

    gates_df = adata_rescaled.uns["gates"]
    
    # Check that X is modified and values are within [0, 1]
    assert not np.array_equal(adata.X, adata_rescaled.raw.X)
    assert np.all(adata.X >= 0) and np.all(adata.X <= 1)

    # Check that marker_C for image1 has a gate (it should be the max value due to 'failed')
    assert not pd.isna(gates_df.loc["marker_C", "image1"])
    
    # Check that other markers (like A and B) also have gates (potentially GMM-derived)
    assert not pd.isna(gates_df.loc["marker_A", "image1"])
    assert not pd.isna(gates_df.loc["marker_B", "image1"])


def test_rescale_log_transformation(rescale_adata: ad.AnnData):
    """Test that log transformation is applied when log=True."""
    adata = rescale_adata.copy()
    original_X = adata.X.copy()

    # Run rescale with log=True
    adata_rescaled = rescale(adata, log=True, verbose=False)

    # Check that adata.raw.X is the original data
    np.testing.assert_array_equal(adata_rescaled.raw.X, original_X)
    
    # Check that adata.X is transformed and scaled to [0, 1]
    assert not np.array_equal(adata_rescaled.X, original_X) # Should be different due to log and scaling
    assert np.all(adata_rescaled.X >= 0) and np.all(adata_rescaled.X <= 1)


def test_rescale_no_raw_data_initially(rescale_adata: ad.AnnData):
    """Test that adata.raw is created if it's initially None."""
    adata = ad.AnnData(rescale_adata.X.copy(), obs=rescale_adata.obs.copy(), var=rescale_adata.var.copy())
    assert adata.raw is None # Ensure it starts as None
    rescale(adata, log=False, verbose=False)
    assert adata.raw is not None
    np.testing.assert_array_equal(adata.raw.X, rescale_adata.X)

def test_rescale_invalid_failed_markers_input(rescale_adata: ad.AnnData):
    """Test that non-dict input for failed_markers raises ValueError."""
    with pytest.raises(ValueError, match="`failed_markers` should be a python dictionary"):
        rescale(rescale_adata, failed_markers=["marker_C"])
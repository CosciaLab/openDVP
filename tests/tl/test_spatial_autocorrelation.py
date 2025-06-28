import anndata as ad
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from opendvp.tl import spatial_autocorrelation

# --- Fixtures ---

def create_adata_spatial(
        grid_size: int = 20, 
        spacing: int = 1, 
        checker_block_size: int = 1, 
        seed: int = 42) -> ad.AnnData:

    rng = np.random.default_rng(seed)
    
    # Generate grid coordinates
    x_indices, y_indices = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size)
    )
    x_coords = (x_indices * spacing).flatten()
    y_coords = (y_indices * spacing).flatten()
    n_obs = x_coords.size

    # Create obs with centroids
    obs = pd.DataFrame({
        "x_centroid": x_coords,
        "y_centroid": y_coords,
    }, index=[f"cell_{i}" for i in range(n_obs)])

    # In silico gene expression
    gene_random = rng.random(n_obs)
    gene_gradient_x = x_coords.astype(float)
    # Use indices before spacing for robust checkerboard calculation
    checker_x = x_indices.flatten() // checker_block_size
    checker_y = y_indices.flatten() // checker_block_size
    gene_checkerboard = ((checker_x + checker_y) % 2).astype(float)
    gene_constant = np.full(n_obs, 5.0)
    gene_with_nan = rng.random(n_obs)
    gene_with_nan[5:10] = np.nan

    # Create gene matrix (obs x genes)
    X = np.vstack([
        gene_random,
        gene_gradient_x,
        gene_checkerboard,
        gene_constant,
        gene_with_nan
    ]).T

    var = pd.DataFrame(index=[
        "gene_random",
        "gene_gradient_x",
        "gene_checkerboard",
        "gene_constant",
        "gene_with_nan"
    ])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def adata_spatial() -> ad.AnnData:
    """Provides a default spatial AnnData object for tests."""
    return create_adata_spatial()


# --- Test Cases ---

def test_moran_i_run(adata_spatial):
    """Test that Moran's I runs without errors and adds correct columns."""
    k = 5
    spatial_autocorrelation(adata_spatial, method="moran", k=k)

    # Check if columns are added
    expected_cols = [f'Moran_I_k{k}', f'Moran_p_sim_k{k}', f'Moran_Zscore_k{k}']
    for col in expected_cols:
        assert col in adata_spatial.var.columns
        assert not adata_spatial.var[col].isnull().all() # Ensure not all values are NaN

def test_geary_c_run(adata_spatial):
    """Test that Geary's C runs without errors and adds correct columns."""
    threshold = 20
    spatial_autocorrelation(adata_spatial, method="geary", threshold=threshold)

    # Check if columns are added
    expected_cols = [f'Geary_C_threshold{threshold}', f'Geary_p_sim_threshold{threshold}']
    for col in expected_cols:
        assert col in adata_spatial.var.columns
        assert not adata_spatial.var[col].isnull().all()

def test_invalid_method(adata_spatial):
    """Test that an invalid method raises a ValueError."""
    with pytest.raises(ValueError, match="Method must be 'moran' or 'geary'"):
        spatial_autocorrelation(adata_spatial, method="invalid_method")

def test_missing_coordinates(adata_spatial):
    """Test that missing coordinate columns raise an error."""
    adata_no_coords = adata_spatial.copy()
    adata_no_coords.obs.drop(columns=['x_centroid'], inplace=True)
    with pytest.raises(KeyError):
        spatial_autocorrelation(adata_no_coords, method="moran")

def test_geary_c_island_error(adata_spatial):
    """Test that Geary's C raises a RuntimeError if too many islands are detected."""
    # Use a very small threshold to guarantee islands
    with pytest.raises(RuntimeError, match="Too many islands"):
        spatial_autocorrelation(adata_spatial, method="geary", threshold=0.1, island_threshold=0.05)

def test_moran_i_values(adata_spatial):
    """Test the logical correctness of Moran's I values for patterned data."""
    k = 6
    spatial_autocorrelation(adata_spatial, method="moran", k=k)
    
    # Gene with positive spatial gradient should have high positive Moran's I
    moran_i_gradient = adata_spatial.var.loc['gene_gradient_x', f'Moran_I_k{k}']
    p_val_gradient = adata_spatial.var.loc['gene_gradient_x', f'Moran_p_sim_k{k}']
    assert moran_i_gradient > 0.3
    assert p_val_gradient < 0.05

    # Gene with random noise should have Moran's I close to 0 and a low Z-score
    moran_i_random = adata_spatial.var.loc['gene_random', f'Moran_I_k{k}']
    z_score_random = adata_spatial.var.loc['gene_random', f'Moran_Zscore_k{k}']
    assert -0.15 < moran_i_random < 0.15
    assert abs(z_score_random) < 3 # Check if it's within 3 std deviations of the mean

    # Gene with checkerboard pattern should have negative Moran's I
    moran_i_checker = adata_spatial.var.loc['gene_checkerboard', f'Moran_I_k{k}']
    p_val_checker = adata_spatial.var.loc['gene_checkerboard', f'Moran_p_sim_k{k}']
    assert moran_i_checker < -0.1
    assert p_val_checker < 0.05

def test_geary_c_values():
    """Test the logical correctness of Geary's C values for patterned data."""
    threshold = 2.0
    adata_spatial = create_adata_spatial(grid_size=25, spacing=2)
    spatial_autocorrelation(adata_spatial, method="geary", threshold=threshold)

    # Gene with positive spatial gradient should have Geary's C < 1
    geary_c_gradient = adata_spatial.var.loc['gene_gradient_x', f'Geary_C_threshold{threshold}']
    p_val_gradient = adata_spatial.var.loc['gene_gradient_x', f'Geary_p_sim_threshold{threshold}']
    assert geary_c_gradient < 0.7
    assert p_val_gradient < 0.05

    # Gene with random noise should have Geary's C close to 1
    geary_c_random = adata_spatial.var.loc['gene_random', f'Geary_C_threshold{threshold}']
    assert 0.85 < geary_c_random < 1.15

    # Gene with checkerboard pattern should have Geary's C > 1
    geary_c_checker = adata_spatial.var.loc['gene_checkerboard', f'Geary_C_threshold{threshold}']
    p_val_checker = adata_spatial.var.loc['gene_checkerboard', f'Geary_p_sim_threshold{threshold}']
    assert geary_c_checker > 1.1
    assert p_val_checker < 0.05

@patch('opendvp.tl.spatial_autocorrelation.logger')
def test_handling_problematic_genes(mock_logger, adata_spatial):
    """Test that genes with constant or NaN values are handled gracefully."""
    k = 5
    spatial_autocorrelation(adata_spatial, method="moran", k=k)
 
    # Check that the results are NaN as expected
    assert np.isnan(adata_spatial.var.loc['gene_constant', f'Moran_I_k{k}'])
    assert np.isnan(adata_spatial.var.loc['gene_with_nan', f'Moran_I_k{k}'])

    # Check that the logger was called with the correct warnings
    # Get all calls to the warning method
    warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
    
    # Check for specific messages
    assert any("Skipping gene gene_constant: has zero variance" in call for call in warning_calls)
    assert any("Skipping gene gene_with_nan: contains NaN or Inf values" in call for call in warning_calls)
    assert any("2 genes failed during MORAN calculation" in call for call in warning_calls)
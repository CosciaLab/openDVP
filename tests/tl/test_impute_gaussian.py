from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.impute_gaussian import impute_gaussian


# Helper function to create a dummy AnnData object for testing
def create_test_adata(
    n_obs=10,
    n_vars=5,
    nan_pattern : Literal['random', 'structured', 'all_valid', 'all_nan'] | None = None,  
    grouping_col=None,
    var_cols=None
):
    """Creates a dummy AnnData object with specified dimensions and NaN patterns."""
    X = np.random.rand(n_obs, n_vars) * 10

    if nan_pattern == 'random':
        nan_indices = np.random.choice(n_obs * n_vars, int(n_obs * n_vars * 0.2), replace=False)
        X.flat[nan_indices] = np.nan
    elif nan_pattern == 'structured':
        # Introduce structured NaNs for predictable testing
        X[:n_obs//2, 0] = np.nan # Protein 0 has NaNs in first half of samples
        X[n_obs//2:, 1] = np.nan # Protein 1 has NaNs in second half of samples
        X[:, 2] = np.nan # Protein 2 is all NaNs
        X[:, 3] = 1.0 # Protein 3 is all valid
        X[:, 4] = 1.0 # Protein 4 is all valid
    elif nan_pattern == 'all_valid':
        X = np.ones((n_obs, n_vars))
    elif nan_pattern == 'all_nan':
        X = np.full((n_obs, n_vars), np.nan)

    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)])
    if grouping_col:
        obs[grouping_col] = ['groupA'] * (n_obs // 2) + ['groupB'] * (n_obs - n_obs // 2)
    
    var_data = {'original_col_1': [f'gene_{i}' for i in range(n_vars)],
                'original_col_2': np.arange(n_vars)}
    if var_cols:
        var_data.update(var_cols)
    var = pd.DataFrame(var_data, index=[f'protein_{i}' for i in range(n_vars)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata

def test_impute_per_protein_default():
    """Test default imputation (per protein) and QC storage."""
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='structured')
    original_X = np.asarray(adata.X.copy())

    # protein_0 has 5 NaNs, protein_1 has 5 NaNs, protein_2 has 10 NaNs, p3 and p4 no NaNs

    imputed_adata = impute_gaussian(adata)

    # 1. Check that no NaNs remain in .X (except for all-NaN columns)
    # protein_2 was all NaNs, so it will be imputed with NaNs.
    assert not np.isnan(imputed_adata.X[:, :2]).any()
    assert np.isnan(imputed_adata.X[:, 2]).all()

    # 2. Check that the original data is stored in layers
    assert 'unimputed' in imputed_adata.layers
    assert np.allclose(imputed_adata.layers['unimputed'], original_X, equal_nan=True)

    # 3. Check that QC metrics are stored in uns
    assert 'impute_gaussian_qc_metrics' in imputed_adata.uns
    qc_df = imputed_adata.uns['impute_gaussian_qc_metrics']
    assert isinstance(qc_df, pd.DataFrame)
    assert qc_df.index.equals(adata.var_names)
    expected_cols = ['n_imputed', 'imputation_mean', 'imputation_stddev', 'imputed_values']
    assert all(col in qc_df.columns for col in expected_cols)

    # 4. Verify QC content for a specific protein (protein_0)
    assert qc_df.loc['protein_0', 'n_imputed'] == 5
    assert len(qc_df.loc['protein_0', 'imputed_values']) == 5
    assert isinstance(qc_df.loc['protein_0', 'imputed_values'], np.ndarray)

    # 5. Verify QC for all-NaN protein (protein_2)
    assert qc_df.loc['protein_2', 'n_imputed'] == 10
    assert np.isnan(qc_df.loc['protein_2', 'imputation_mean'])
    assert np.isnan(qc_df.loc['protein_2', 'imputation_stddev'])
    assert np.isnan(qc_df.loc['protein_2', 'imputed_values']).all()


def test_impute_per_sample():
    """Test imputation per sample."""
    # Create data where rows have structured NaNs
    X = np.random.rand(5, 10) * 10
    # row 0 has NaNs in first half of proteins
    X[0, :5] = np.nan
    # row 1 has NaNs in second half
    X[1, 5:] = np.nan
    # row 2 is all NaNs
    X[2, :] = np.nan

    adata = ad.AnnData(X)
    original_X = adata.X.copy()

    imputed_adata = impute_gaussian(adata, perSample=True)

    # 1. Check that no NaNs remain in .X (except for all-NaN rows)
    assert not np.isnan(imputed_adata.X[[0, 1, 3, 4], :]).any()
    assert np.isnan(imputed_adata.X[2, :]).all()

    # 2. Check layer and uns storage
    assert 'unimputed' in imputed_adata.layers
    assert np.allclose(imputed_adata.layers['unimputed'], original_X, equal_nan=True)
    assert 'impute_gaussian_qc_metrics' in imputed_adata.uns

    # 3. Check QC DataFrame
    qc_df = imputed_adata.uns['impute_gaussian_qc_metrics']
    assert qc_df.index.equals(adata.obs_names)  # Index should be samples

    # 4. Verify QC content for a specific sample
    assert qc_df.loc['0', 'n_imputed'] == 5
    assert len(qc_df.loc['0', 'imputed_values']) == 5


def test_no_nans_no_change():
    """Test that data with no NaNs is not changed."""
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_valid')
    original_X = adata.X.copy()

    imputed_adata = impute_gaussian(adata)

    # .X should be unchanged
    assert np.array_equal(imputed_adata.X, original_X)

    # QC metrics should reflect no imputation
    qc_df = imputed_adata.uns['impute_gaussian_qc_metrics']
    assert qc_df['n_imputed'].sum() == 0
    assert all((isinstance(arr, np.ndarray | list) and len(arr) == 0) or pd.isna(arr)
        for arr in qc_df['imputed_values']
    )


def test_custom_keys():
    """Test using custom keys for layer and uns storage."""
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='structured')

    imputed_adata = impute_gaussian(adata, layer_key='original_data', uns_key='my_imputation_stats')

    assert 'unimputed' not in imputed_adata.layers
    assert 'original_data' in imputed_adata.layers

    assert 'impute_gaussian_qc_metrics' not in imputed_adata.uns
    assert 'my_imputation_stats' in imputed_adata.uns


def test_ddof_calculation_is_correct():
    """Test that the standard deviation is calculated with ddof=1 (sample std)."""
    data = np.array([1, 2, 3, 4, 10, np.nan])
    adata = ad.AnnData(X=data.reshape(-1, 1))

    # Manually calculate sample standard deviation
    manual_std = np.nanstd(data, ddof=1)

    imputed_adata = impute_gaussian(adata)

    qc_df = imputed_adata.uns['impute_gaussian_qc_metrics']
    stored_std = qc_df.loc['0', 'imputation_stddev']  # var name is '0' by default

    assert np.isclose(manual_std, stored_std)
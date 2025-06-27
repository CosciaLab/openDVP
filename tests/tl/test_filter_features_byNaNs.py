from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.filter_features_byNaNs import filter_features_byNaNs


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

def test_basic_filtering_no_grouping():
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='structured')
    # Based on structured NaN pattern and threshold=0.7 (needs >=70% valid, i.e., <=30% NaN)
    # protein_0: 50% NaN -> removed
    # protein_1: 50% NaN -> removed
    # protein_2: 100% NaN -> removed
    # protein_3: 0% NaN -> kept
    # protein_4: 0% NaN -> kept
    expected_kept_proteins = ['protein_3', 'protein_4']

    filtered_adata = filter_features_byNaNs(adata, threshold=0.7, grouping=None)

    assert filtered_adata.shape[1] == len(expected_kept_proteins)
    assert all(p in filtered_adata.var_names for p in expected_kept_proteins)
    assert 'mean' in filtered_adata.var.columns
    assert 'nan_proportions' in filtered_adata.var.columns
    assert 'original_col_1' in filtered_adata.var.columns # Check original columns preserved

    # Check adata.uns content
    assert 'filter_features_byNaNs_qc_metrics' in filtered_adata.uns
    qc_df = filtered_adata.uns['filter_features_byNaNs_qc_metrics']
    assert isinstance(qc_df, pd.DataFrame)
    assert qc_df.shape[0] == adata.shape[1] # Should contain all initial proteins
    assert 'overall_mean' in qc_df.columns
    assert 'overall_nan_proportions' in qc_df.columns
    assert 'overall_valid' in qc_df.columns
    assert 'valid' in qc_df.columns # For non-grouping case

    # Verify specific QC values for protein_2 (all NaNs)
    assert qc_df.loc['protein_2', 'overall_nan_proportions'] == 1.0
    assert not qc_df.loc['protein_2', 'overall_valid']

    # Check that 'mean' and 'nan_proportions' in returned adata.var match 'overall_' in uns
    assert np.allclose(filtered_adata.var['mean'], qc_df.loc[filtered_adata.var_names, 'overall_mean'])
    assert np.allclose(filtered_adata.var['nan_proportions'], qc_df.loc[filtered_adata.var_names, 'overall_nan_proportions'])

def test_filtering_with_grouping_any():
    adata = create_test_adata(n_obs=20, n_vars=5, grouping_col='sample_group')
    # Make some proteins pass in ANY group (threshold=0.7, i.e., <=30% NaN allowed)
    adata.X[:5, 0] = np.nan # protein_0: groupA (50% NaN)
    adata.X[5:10, 0] = 1.0
    adata.X[10:, 0] = 1.0 # protein_0: groupB (0% NaN) -> valid in groupB
    adata.X[:10, 1] = 1.0 # protein_1: groupA (0% NaN) -> valid in groupA
    adata.X[10:15, 1] = np.nan # protein_1: groupB (50% NaN)
    adata.X[15:, 1] = 1.0
    adata.X[:5, 2] = np.nan # protein_2: groupA (50% NaN)
    adata.X[5:10, 2] = 1.0
    adata.X[10:15, 2] = np.nan # protein_2: groupB (50% NaN) -> not valid in any
    adata.X[15:, 2] = 1.0
    adata.X[:, 3] = 1.0 # protein_3: groupA (0% NaN), groupB (0% NaN) -> valid in all
    adata.X[:, 4] = np.nan # protein_4: groupA (100% NaN), groupB (100% NaN) -> not valid in any

    expected_kept_proteins = ['protein_0', 'protein_1', 'protein_3']
    filtered_adata = filter_features_byNaNs(adata, threshold=0.7, grouping='sample_group', valid_in_ANY_or_ALL_groups='ANY')
    assert filtered_adata.shape[1] == len(expected_kept_proteins)
    assert all(p in filtered_adata.var_names for p in expected_kept_proteins)
    qc_df = filtered_adata.uns['filter_features_byNaNs_qc_metrics']
    assert 'groupA_nan_proportions' in qc_df.columns
    assert 'groupB_nan_proportions' in qc_df.columns
    assert 'valid_in_any_group' in qc_df.columns

    # Verify specific QC values for protein_0
    assert qc_df.loc['protein_0', 'groupA_nan_proportions'] == 0.5
    assert not qc_df.loc['protein_0', 'groupA_valid'] # 0.5 > (1-0.7)=0.3, so False
    assert qc_df.loc['protein_0', 'groupB_nan_proportions'] == 0.0
    assert qc_df.loc['protein_0', 'groupB_valid']
    assert qc_df.loc['protein_0', 'valid_in_any_group']

    # Verify specific QC values for protein_1
    assert qc_df.loc['protein_1', 'groupA_nan_proportions'] == 0.0
    assert qc_df.loc['protein_1', 'groupA_valid']
    assert qc_df.loc['protein_1', 'groupB_nan_proportions'] == 0.5
    assert not qc_df.loc['protein_1', 'groupB_valid']
    assert qc_df.loc['protein_1', 'valid_in_any_group']

    # Verify specific QC values for protein_2
    assert qc_df.loc['protein_2', 'groupA_nan_proportions'] == 0.5
    assert not qc_df.loc['protein_2', 'groupA_valid']
    assert qc_df.loc['protein_2', 'groupB_nan_proportions'] == 0.5
    assert not qc_df.loc['protein_2', 'groupB_valid']
    assert not qc_df.loc['protein_2', 'valid_in_any_group']

def test_filtering_with_grouping_all():
    adata = create_test_adata(n_obs=20, n_vars=5, grouping_col='sample_group')
    # Same setup as above, but 'ALL' criteria
    # Re-use the same structured NaN pattern as test_filtering_with_grouping_any
    adata.X[:5, 0] = np.nan # protein_0: groupA (50% NaN)
    adata.X[5:10, 0] = 1.0
    adata.X[10:, 0] = 1.0 # protein_0: groupB (0% NaN)
    adata.X[:10, 1] = 1.0 # protein_1: groupA (0% NaN)
    adata.X[10:15, 1] = np.nan # protein_1: groupB (50% NaN)
    adata.X[15:, 1] = 1.0
    adata.X[:5, 2] = np.nan # protein_2: groupA (50% NaN)
    adata.X[5:10, 2] = 1.0
    adata.X[10:15, 2] = np.nan # protein_2: groupB (50% NaN)
    adata.X[15:, 2] = 1.0
    adata.X[:, 3] = 1.0 # protein_3: groupA (0% NaN), groupB (0% NaN)
    adata.X[:, 4] = np.nan # protein_4: groupA (100% NaN), groupB (100% NaN)

    # Only protein_3 passes in ALL groups
    expected_kept_proteins = ['protein_3']

    filtered_adata = filter_features_byNaNs(adata, threshold=0.7, grouping='sample_group', valid_in_ANY_or_ALL_groups='ALL')

    assert filtered_adata.shape[1] == len(expected_kept_proteins)
    assert all(p in filtered_adata.var_names for p in expected_kept_proteins)

    qc_df = filtered_adata.uns['filter_features_byNaNs_qc_metrics']
    assert 'valid_in_all_groups' in qc_df.columns
    assert not qc_df.loc['protein_0', 'valid_in_all_groups']
    assert qc_df.loc['protein_3', 'valid_in_all_groups']

def test_threshold_extremes():
    # Use n_vars=5 to match the hardcoded structured pattern
    adata_mixed = create_test_adata(n_obs=10, n_vars=5, nan_pattern='structured')

    # Scenario 1: threshold = 0.0
    # Keep if nan_proportion < (1.0 - 0.0) => nan_proportion < 1.0
    # This keeps all proteins except those that are 100% NaN.
    filtered_adata_0 = filter_features_byNaNs(adata_mixed, threshold=0.0)
    expected_kept_0 = ['protein_0', 'protein_1', 'protein_3', 'protein_4']
    assert filtered_adata_0.shape[1] == len(expected_kept_0)
    assert all(p in filtered_adata_0.var_names for p in expected_kept_0)
    assert 'protein_2' not in filtered_adata_0.var_names

    # Scenario 2: threshold = 1.0
    # Keep if nan_proportion < (1.0 - 1.0) => nan_proportion < 0.0
    # This is impossible, so all proteins are removed.
    filtered_adata_1 = filter_features_byNaNs(adata_mixed, threshold=1.0)
    assert filtered_adata_1.shape[1] == 0

    # Scenario 3: All NaNs with threshold=0.0 (should remove all)
    adata_all_nan = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_nan')
    filtered_adata_all_nan = filter_features_byNaNs(adata_all_nan, threshold=0.0)
    assert filtered_adata_all_nan.shape[1] == 0

    # Scenario 4: All valid with threshold=1.0 (should remove all, as 0.0 < 0.0 is False)
    adata_all_valid = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_valid')
    filtered_adata_all_valid = filter_features_byNaNs(adata_all_valid, threshold=1.0)
    assert filtered_adata_all_valid.shape[1] == 0

    # Scenario 5: All valid with threshold close to 1.0 (should keep all)
    adata_all_valid_2 = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_valid')
    filtered_adata_all_valid_2 = filter_features_byNaNs(adata_all_valid_2, threshold=0.99)
    assert filtered_adata_all_valid_2.shape[1] == 5

def test_invalid_threshold_raises_error():
    adata = create_test_adata()
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        filter_features_byNaNs(adata, threshold=-0.1)
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        filter_features_byNaNs(adata, threshold=1.1)

def test_grouping_column_not_found_raises_key_error():
    adata = create_test_adata(grouping_col=None) # No grouping column
    with pytest.raises(KeyError): # pandas will raise KeyError if column not found
        filter_features_byNaNs(adata, grouping='non_existent_group')
        

def test_no_nans_in_data_all_kept():
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_valid')
    filtered_adata = filter_features_byNaNs(adata, threshold=0.7)
    assert filtered_adata.shape[1] == 5 # All should be kept
    assert np.all(filtered_adata.var['nan_proportions'] == 0.0)

def test_all_nans_in_data_all_removed():
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='all_nan')
    filtered_adata = filter_features_byNaNs(adata, threshold=0.7)
    assert filtered_adata.shape[1] == 0 # All should be removed
    qc_df = filtered_adata.uns['filter_features_byNaNs_qc_metrics']
    assert np.all(qc_df['overall_nan_proportions'] == 1.0)

def test_original_var_columns_preserved_and_qc_cleaned():
    adata = create_test_adata(n_obs=10, n_vars=3, var_cols={'custom_col': ['a', 'b', 'c']})
    filtered_adata = filter_features_byNaNs(adata, threshold=0.5)
    assert 'original_col_1' in filtered_adata.var.columns
    assert 'custom_col' in filtered_adata.var.columns
    assert 'mean' in filtered_adata.var.columns
    assert 'nan_proportions' in filtered_adata.var.columns
    # Ensure no other QC columns are present in the final adata.var
    assert 'overall_mean' not in filtered_adata.var.columns
    assert 'overall_nan_count' not in filtered_adata.var.columns
    assert 'overall_valid' not in filtered_adata.var.columns
    assert 'valid' not in filtered_adata.var.columns
    assert 'not_valid' not in filtered_adata.var.columns

def test_qc_metrics_in_uns_completeness_with_grouping():
    adata = create_test_adata(n_obs=10, n_vars=5, nan_pattern='structured', grouping_col='group')
    filtered_adata = filter_features_byNaNs(adata, threshold=0.7, grouping='group', valid_in_ANY_or_ALL_groups='ANY')

    qc_df = filtered_adata.uns['filter_features_byNaNs_qc_metrics']
    assert isinstance(qc_df, pd.DataFrame)
    assert qc_df.shape[0] == adata.shape[1] # All original proteins
    
    # Check for overall metrics
    assert 'overall_mean' in qc_df.columns
    assert 'overall_nan_count' in qc_df.columns
    assert 'overall_valid_count' in qc_df.columns
    assert 'overall_nan_proportions' in qc_df.columns
    assert 'overall_valid' in qc_df.columns

    # Check for group-specific metrics (assuming 'groupA' and 'groupB' are present)
    assert 'groupA_mean' in qc_df.columns
    assert 'groupA_nan_count' in qc_df.columns
    assert 'groupA_valid' in qc_df.columns
    assert 'groupB_mean' in qc_df.columns

    # Check for combined group metrics
    assert 'valid_in_all_groups' in qc_df.columns
    assert 'valid_in_any_group' in qc_df.columns
    assert 'not_valid_in_any_group' in qc_df.columns

    # Check that original var columns are also in the QC dataframe
    assert 'original_col_1' in qc_df.columns
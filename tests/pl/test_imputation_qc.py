from unittest.mock import patch

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opendvp.plotting import imputation_qc


def test_imputation_qc():
    """
    Test the imputation_qc function to ensure it runs without errors and
    handles parameters correctly.
    """
    # Create a dummy AnnData object
    n_obs, n_vars = 100, 50
    X_imputed = np.random.rand(n_obs, n_vars)
    X_raw = X_imputed.copy()
    # Introduce some NaNs to simulate unimputed data
    X_raw[np.random.choice([True, False], size=X_raw.shape, p=[0.1, 0.9])] = np.nan
    adata = anndata.AnnData(X_imputed)
    adata.layers["unimputed"] = X_raw
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Test if the function runs without highlighting
    fig = imputation_qc(adata, return_fig=True)
    assert fig is not None, "Figure object should be returned"
    plt.close(fig)

    # Test with highlighting
    fig = imputation_qc(adata, highlight_genes=["Gene_5", "Gene_10"], return_fig=True)
    assert fig is not None, "Figure object should be returned with highlighted genes"
    plt.close(fig)

    # Test with a pre-existing axis
    fig, ax = plt.subplots()
    result_fig = imputation_qc(adata, ax=ax, return_fig=True)
    assert result_fig is fig, "Should return the same figure object when an axis is provided"
    plt.close(fig)

    # Test dataframe processing
    adata_copy = adata.copy()
    df_imputed = adata_copy.to_df()
    df_raw = pd.DataFrame(data=adata_copy.layers["unimputed"], index=adata_copy.obs_names, columns=adata_copy.var_names)
    imp_mean = df_imputed.mean(axis=0)
    raw_mean = df_raw.mean(skipna=True, axis=0)
    df_sns = pd.DataFrame({"imp_mean": imp_mean, "raw_mean": raw_mean})
    df_sns["diff"] = df_sns.raw_mean - df_sns.imp_mean

    assert "imp_mean" in df_sns.columns
    assert "raw_mean" in df_sns.columns
    assert "diff" in df_sns.columns


@patch("matplotlib.pyplot.show")
def test_imputation_qc_shows_plot_and_returns_none(mock_show):
    """
    Test that plt.show() is called and None is returned when return_fig is False
    and no ax is provided.
    """
    # Create a dummy AnnData object
    n_obs, n_vars = 10, 5
    X_imputed = np.random.rand(n_obs, n_vars)
    adata = anndata.AnnData(X_imputed)
    # Introduce some NaNs to simulate unimputed data and create variance
    X_raw = X_imputed.copy()
    X_raw[np.random.choice([True, False], size=X_raw.shape, p=[0.1, 0.9])] = np.nan
    adata.layers["unimputed"] = X_raw
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Call the function with parameters that trigger plt.show()
    result = imputation_qc(adata, return_fig=False, ax=None)

    # Assert that the function returns None
    assert result is None, "Function should return None when return_fig is False"

    # Assert that plt.show() was called
    mock_show.assert_called_once()

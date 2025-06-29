import numpy as np
import pandas as pd
import anndata as ad
import pytest
from opendvp.tl.spatial_hyperparameter_search import spatial_hyperparameter_search

@pytest.fixture
def simple_adata():
    np.random.seed(42)
    n = 10
    obs = pd.DataFrame({
        'x_centroid': np.random.uniform(0, 100, n),
        'y_centroid': np.random.uniform(0, 100, n)
    }, index=[f'cell{i}' for i in range(n)])
    var = pd.DataFrame(index=[f'gene{i}' for i in range(3)])
    X = np.random.randn(n, 3)
    return ad.AnnData(X=X, obs=obs, var=var)

def test_returns_dataframe_and_plot(simple_adata):
    df, (fig, ax) = spatial_hyperparameter_search(
        simple_adata,
        x_y=['x_centroid', 'y_centroid'],
        threshold_range=np.arange(1, 10, 2),
        return_df=True
    ) # type: ignore
    assert isinstance(df, pd.DataFrame)
    assert 'threshold' in df.columns
    assert fig is not None and ax is not None

def test_default_parameters(simple_adata):
    # Should not raise
    result = spatial_hyperparameter_search(simple_adata)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_invalid_column_raises(simple_adata):
    with pytest.raises(ValueError):
        spatial_hyperparameter_search(simple_adata, x_y=['not_a_col', 'y_centroid'])

# def test_plot_network_at_threshold(monkeypatch, simple_adata):
#     called = {}
#     def fake_plot_graph_network(w, coords, threshold):
#         called['called'] = True
#         called['threshold'] = threshold
#     monkeypatch.setattr('opendvp.plotting.plot_graph_network.plot_graph_network', fake_plot_graph_network)
#     spatial_hyperparameter_search(simple_adata, threshold_range=[5, 10], plot_network_at=5)
#     assert called.get('called') is True
#     assert called.get('threshold') == 5

# def test_plot_network_at_runs_without_error(simple_adata):
#     # This test just ensures the function runs and returns expected output when plot_network_at is set
#     result = spatial_hyperparameter_search(
#         simple_adata,
#         threshold_range=np.arange(1, 10, 2),
#         plot_network_at=5,
#         return_df=True
#     )
#     df, (fig, ax) = result
#     assert isinstance(df, pd.DataFrame)
#     assert fig is not None and ax is not None

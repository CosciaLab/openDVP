import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

matplotlib.use("Agg")  # Use non-interactive backend for tests
from opendvp.plotting import correlation_heatmap


@pytest.fixture
def simple_adata():
    np.random.seed(42)
    X = np.random.rand(5, 4)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(5)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(4)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def labeled_adata():
    np.random.seed(0)
    X = np.random.rand(6, 3)
    obs = pd.DataFrame({"sample": ["A", "B", "A", "B", "A", "B"]}, index=[f"cell{i}" for i in range(6)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(3)])
    return AnnData(X=X, obs=obs, var=var)


def test_returns_figure(simple_adata):
    fig = correlation_heatmap(simple_adata, return_fig=True)
    assert fig is not None
    assert hasattr(fig, "suptitle")


def test_shows_plot(simple_adata, monkeypatch):
    called = {}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    result = correlation_heatmap(simple_adata, return_fig=False)
    assert result is None
    assert called["show"]


def test_sample_label(labeled_adata):
    fig = correlation_heatmap(labeled_adata, sample_label="sample", return_fig=True)
    assert fig is not None


def test_different_methods(simple_adata):
    for method in ["pearson", "kendall", "spearman"]:
        fig = correlation_heatmap(simple_adata, correlation_method=method, return_fig=True)
        assert fig is not None


def test_colormap_and_limits(simple_adata):
    fig = correlation_heatmap(simple_adata, color_map="viridis", vmin=0.0, vmax=1.0, return_fig=True)
    assert fig is not None


def test_invalid_method(simple_adata):
    with pytest.raises(ValueError):
        correlation_heatmap(simple_adata, correlation_method="invalid", return_fig=True)


def test_empty_adata():
    X = np.empty((0, 0))
    obs = pd.DataFrame(index=[])
    var = pd.DataFrame(index=[])
    adata = AnnData(X=X, obs=obs, var=var)
    with pytest.raises(ValueError):
        correlation_heatmap(adata, return_fig=True)


# ValueError: Distance matrix 'X' must be symmetric.
def test_single_feature():
    X = np.random.rand(5, 1)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(5)])
    var = pd.DataFrame(index=["gene0"])
    adata = AnnData(X=X, obs=obs, var=var)
    with pytest.raises(ValueError, match="Distance matrix 'X' must be symmetric"):
        correlation_heatmap(adata, return_fig=True)


# ValueError: The number of observations cannot be determined on an empty distance matrix.
def test_single_sample():
    X = np.random.rand(1, 4)
    obs = pd.DataFrame(index=["cell0"])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(4)])
    adata = AnnData(X=X, obs=obs, var=var)
    with pytest.raises(ValueError, match="The number of observations cannot be determined on an empty distance matrix"):
        correlation_heatmap(adata, return_fig=True)

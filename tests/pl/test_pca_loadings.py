import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opendvp.pl.pca_loadings import pca_loadings


@pytest.fixture
def pca_adata() -> ad.AnnData:
    """An AnnData carrying the PCA output `pca_loadings` reads: varm, uns and a Genes column."""
    rng = np.random.default_rng(0)
    n_vars = 40
    adata = ad.AnnData(
        X=rng.normal(size=(10, n_vars)),
        var=pd.DataFrame({"Genes": [f"GENE{i}" for i in range(n_vars)]}, index=[f"P{i}" for i in range(n_vars)]),
    )
    adata.varm["PCs"] = rng.normal(size=(n_vars, 5))
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.25, 0.15, 0.1, 0.05])}
    return adata


def test_returns_figure(pca_adata: ad.AnnData):
    fig = pca_loadings(pca_adata, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_returns_none_without_return_fig(pca_adata: ad.AnnData):
    assert pca_loadings(pca_adata) is None


def test_uses_the_supplied_axes(pca_adata: ad.AnnData):
    _, ax = plt.subplots()
    fig = pca_loadings(pca_adata, ax=ax, return_fig=True)
    assert fig is ax.figure
    plt.close(fig)


def test_labels_are_gene_names(pca_adata: ad.AnnData):
    fig = pca_loadings(pca_adata, top=5, n_pcs=1, return_fig=True)
    drawn = {text.get_text() for text in fig.axes[0].texts}
    assert drawn <= set(pca_adata.var["Genes"])
    assert len(drawn) == 5
    plt.close(fig)


def test_axis_labels_report_explained_variance(pca_adata: ad.AnnData):
    fig = pca_loadings(pca_adata, return_fig=True)
    assert "40.0 %" in fig.axes[0].get_xlabel()
    assert "25.0 %" in fig.axes[0].get_ylabel()
    plt.close(fig)


def test_top_bounds_the_number_of_labels(pca_adata: ad.AnnData):
    fig = pca_loadings(pca_adata, top=3, n_pcs=2, return_fig=True)
    assert len(fig.axes[0].texts) <= 6
    plt.close(fig)


def test_missing_pca_raises(pca_adata: ad.AnnData):
    del pca_adata.varm["PCs"]
    with pytest.raises(KeyError):
        pca_loadings(pca_adata)

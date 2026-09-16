import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opendvp.pl.rankplot import rankplot


@pytest.fixture
def rank_adata() -> ad.AnnData:
    """Two groups of ten samples over thirty proteins, with scattered NaNs."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=20, scale=2, size=(20, 30))
    X[rng.random(X.shape) < 0.1] = np.nan
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"group": ["A"] * 10 + ["B"] * 10}, index=[f"s{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"P{i}" for i in range(30)]),
    )


def test_returns_figure(rank_adata: ad.AnnData):
    fig = rankplot(rank_adata, adata_obs_key="group", return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_runs_with_ax(rank_adata: ad.AnnData):
    _, ax = plt.subplots()
    assert rankplot(rank_adata, adata_obs_key="group", ax=ax) is None
    plt.close("all")


def test_plots_every_group(rank_adata: ad.AnnData):
    fig = rankplot(rank_adata, adata_obs_key="group", return_fig=True)
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert {"A", "B"} <= labels
    plt.close(fig)


def test_subsets_to_the_requested_groups(rank_adata: ad.AnnData):
    fig = rankplot(rank_adata, adata_obs_key="group", groups=["A"], return_fig=True)
    labels = {text.get_text() for text in fig.axes[0].get_legend().get_texts()}
    assert "B" not in labels
    plt.close(fig)


def test_labels_requested_proteins(rank_adata: ad.AnnData):
    fig = rankplot(rank_adata, adata_obs_key="group", proteins_to_label=["P0", "P1"], return_fig=True)
    drawn = {text.get_text() for text in fig.axes[0].texts}
    assert {"P0", "P1"} <= drawn
    plt.close(fig)


def test_unknown_group_raises(rank_adata: ad.AnnData):
    with pytest.raises(ValueError, match="Groups not present"):
        rankplot(rank_adata, adata_obs_key="group", groups=["C"])


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_out_of_range_presence_fraction_raises(rank_adata: ad.AnnData, fraction: float):
    with pytest.raises(ValueError, match="min_presence_fraction"):
        rankplot(rank_adata, adata_obs_key="group", min_presence_fraction=fraction)


def test_everything_filtered_out_raises(rank_adata: ad.AnnData):
    rank_adata.X = np.full_like(rank_adata.X, np.nan)
    with pytest.raises(ValueError, match="filtering too strict"):
        rankplot(rank_adata, adata_obs_key="group")


def test_does_not_warn(rank_adata: ad.AnnData):
    """rankplot used to emit a pandas FutureWarning and a matplotlib deprecation warning."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = rankplot(rank_adata, adata_obs_key="group", return_fig=True)
    from_rankplot = [w for w in caught if "rankplot.py" in str(w.filename)]
    assert from_rankplot == [], [str(w.message) for w in from_rankplot]
    plt.close(fig)

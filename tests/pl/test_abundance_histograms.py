import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opendvp.pl.abundance_histograms import abundance_histograms


@pytest.fixture
def abundance_adata() -> ad.AnnData:
    """Six samples of log2-scale intensities, which is the range the plot's bins assume."""
    rng = np.random.default_rng(7)
    return ad.AnnData(
        X=rng.normal(loc=15, scale=2, size=(6, 200)),
        obs=pd.DataFrame({"raw_file_id": [f"run_{i}" for i in range(6)]}, index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"P{i}" for i in range(200)]),
    )


def test_returns_figure(abundance_adata: ad.AnnData):
    fig = abundance_histograms(abundance_adata, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_returns_none_without_return_fig(abundance_adata: ad.AnnData):
    assert abundance_histograms(abundance_adata) is None


def test_grid_is_wide_enough_for_every_sample(abundance_adata: ad.AnnData):
    fig = abundance_histograms(abundance_adata, n_cols=4, return_fig=True)
    # 6 samples over 4 columns needs 2 rows, so 8 axes
    assert len(fig.axes) == 8
    plt.close(fig)


@pytest.mark.parametrize("n_cols", [2, 3, 6])
def test_column_count_is_honoured(abundance_adata: ad.AnnData, n_cols: int):
    fig = abundance_histograms(abundance_adata, n_cols=n_cols, return_fig=True)
    expected_rows = int(np.ceil(6 / n_cols))
    assert len(fig.axes) == expected_rows * n_cols
    plt.close(fig)


def test_each_panel_is_titled_with_its_file_id(abundance_adata: ad.AnnData):
    fig = abundance_histograms(abundance_adata, return_fig=True)
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert titles == {f"file_id: run_{i}" for i in range(6)}
    plt.close(fig)


def test_each_panel_reports_a_shapiro_p_value(abundance_adata: ad.AnnData):
    fig = abundance_histograms(abundance_adata, return_fig=True)
    annotated = [ax for ax in fig.axes if any("Schapiro p:" in t.get_text() for t in ax.texts)]
    assert len(annotated) == 6
    plt.close(fig)


def test_missing_raw_file_id_raises(abundance_adata: ad.AnnData):
    del abundance_adata.obs["raw_file_id"]
    with pytest.raises(KeyError, match="raw_file_id"):
        abundance_histograms(abundance_adata)


def test_titles_follow_row_order_not_index_labels():
    """`adata.obs.raw_file_id[i]` used to index by label, so an integer index gave wrong titles."""
    rng = np.random.default_rng(1)
    adata = ad.AnnData(
        X=rng.normal(loc=15, scale=2, size=(3, 50)),
        # integer labels in descending order: label-based lookup would mismatch row order
        obs=pd.DataFrame({"raw_file_id": ["first", "second", "third"]}, index=pd.Index([20, 10, 0])),
        var=pd.DataFrame(index=[f"P{i}" for i in range(50)]),
    )
    fig = abundance_histograms(adata, n_cols=3, return_fig=True)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["file_id: first", "file_id: second", "file_id: third"]
    plt.close(fig)

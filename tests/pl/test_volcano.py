import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from opendvp.plotting.volcano import volcano


@pytest.fixture
def volcano_adata() -> ad.AnnData:
    """Create a sample AnnData object for volcano plot testing."""
    n_vars = 100
    var_names = [f"gene_{i}" for i in range(n_vars)]

    rng = np.random.default_rng(42)

    # Simulate log2 fold change
    mean_diff = rng.normal(loc=0, scale=1.5, size=n_vars)

    # Simulate p-values, with some being very small (significant)
    p_corr = rng.uniform(0, 1, size=n_vars)
    p_corr[:10] = rng.uniform(0, 0.04, size=10)  # First 10 are significant at FDR 0.05

    # Calculate -log10 p-value
    neg_log10_p_corr = -np.log10(p_corr)

    var_df = pd.DataFrame(
        {"mean_diff": mean_diff, "-log10_p_corr": neg_log10_p_corr, "p_corr": p_corr}, index=var_names
    )

    # Create a dummy AnnData object. X and obs are not used by the function.
    adata = ad.AnnData(X=np.zeros((5, n_vars)), var=var_df)
    return adata


def test_returns_figure(volcano_adata: ad.AnnData):
    """Test that the function returns a matplotlib Figure when return_fig=True."""
    fig = volcano(volcano_adata, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_runs_with_ax(volcano_adata: ad.AnnData):
    """Test that the function runs with a pre-existing Axes object."""
    fig, ax = plt.subplots()
    returned_fig = volcano(volcano_adata, ax=ax, return_fig=True)
    assert returned_fig is fig
    # Check if something was plotted by looking at artists or labels
    assert len(ax.collections) > 0  # scatter creates a PathCollection
    assert ax.get_xlabel() == "mean_diff"
    plt.close(fig)


def test_no_return_fig(volcano_adata: ad.AnnData, monkeypatch):
    """Test that the function returns None when return_fig=False."""
    # Mock plt.show() to prevent it from blocking tests
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = volcano(volcano_adata, return_fig=False)
    assert fig is None


def test_significant_highlighting(volcano_adata: ad.AnnData):
    """Test that significant highlighting runs without error."""
    fig = volcano(volcano_adata, significant=True, FDR=0.05, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_significant_without_fdr_raises_error(volcano_adata: ad.AnnData):
    """Test that ValueError is raised if significant=True but FDR is not provided."""
    with pytest.raises(ValueError, match="FDR must be specified if significant=True"):
        volcano(volcano_adata, significant=True)


def test_tag_top_genes(volcano_adata: ad.AnnData):
    """Test that tagging top genes runs without error and adds text labels."""
    fig, ax = plt.subplots()
    volcano(volcano_adata, tag_top=5, ax=ax, return_fig=True)
    assert isinstance(fig, Figure)
    # adjust_text adds arrows, but the original texts are also there
    assert len(ax.texts) > 0
    plt.close(fig)


def test_highlight_genes(volcano_adata: ad.AnnData):
    """Test highlighting a specific list of genes."""
    genes_to_highlight = ["gene_1", "gene_5", "gene_99"]
    fig = volcano(volcano_adata, highlight_genes=genes_to_highlight, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_highlight_genes_not_found(volcano_adata: ad.AnnData):
    """Test that highlighting non-existent genes does not raise an error."""
    genes_to_highlight = ["gene_1", "non_existent_gene"]
    fig = volcano(volcano_adata, highlight_genes=genes_to_highlight, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_highlight_genes_no_names(volcano_adata: ad.AnnData):
    """Test highlighting genes without showing their names."""
    genes_to_highlight = ["gene_1", "gene_5"]
    fig, ax = plt.subplots()
    volcano(
        volcano_adata,
        highlight_genes=genes_to_highlight,
        show_highlighted_genes_names=False,
        ax=ax,
        return_fig=True,
    )
    # There should be no text artists for the highlighted genes
    assert len(ax.texts) == 0
    plt.close(fig)


def test_with_group_names(volcano_adata: ad.AnnData):
    """Test that providing group names sets the x-axis label correctly."""
    fig, ax = plt.subplots()
    volcano(volcano_adata, group1="Treated", group2="Control", ax=ax)
    expected_label = "Difference in mean protein expression (log2)\nTreated (right) vs Control (left)"
    assert ax.get_xlabel() == expected_label
    plt.close(fig)


def test_missing_columns(volcano_adata: ad.AnnData):
    """Test that a KeyError is raised if required columns are missing."""
    with pytest.raises(KeyError):
        volcano(volcano_adata, x="missing_x_col", return_fig=True)
    with pytest.raises(KeyError):
        volcano(volcano_adata, y="missing_y_col", return_fig=True)
    with pytest.raises(KeyError):
        volcano(volcano_adata, significant=True, FDR=0.05, significant_metric="missing_sig_col", return_fig=True)

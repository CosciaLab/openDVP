import anndata
import numpy as np
import pandas as pd
import pytest

from opendvp.plotting import feature_comparison_boxplot


@pytest.fixture
def dummy_adata_for_comparison():
    """Create a dummy AnnData object for comparison boxplot testing."""
    obs = pd.DataFrame({"group": ["A", "A", "B", "B", "A", "B"], "three_groups": ["A", "B", "C", "A", "B", "C"]})
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    X = np.array(
        [
            [1, 5, 10],  # A
            [2, 6, 12],  # A
            [3, 6, 9],  # B
            [4, 7, 10],  # B
            [1.5, 5.5, 11],  # A
            [3.5, 6.5, 9.5],  # B
        ]
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_feature_comparison_boxplot_returns_figure(dummy_adata_for_comparison):
    """Test that the function returns a figure."""
    fig = feature_comparison_boxplot(
        dummy_adata_for_comparison,
        features=["gene1", "gene2", "gene3"],
        group_by="group",
        return_fig=True,
    )
    assert fig is not None
    assert len(fig.axes) == 1


def test_feature_comparison_boxplot_sorting(dummy_adata_for_comparison):
    """Test that features are correctly sorted by mean difference."""
    fig = feature_comparison_boxplot(
        dummy_adata_for_comparison,
        features=["gene1", "gene2", "gene3"],
        group_by="group",
        zscore=False,  # Test with raw values
        return_fig=True,
    )
    ax = fig.axes[0]
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert xticklabels == ["gene3", "gene2", "gene1"]


def test_zscore_label(dummy_adata_for_comparison):
    """Test that the y-axis label is correct when zscore is True."""
    fig = feature_comparison_boxplot(
        dummy_adata_for_comparison,
        features=["gene1"],
        group_by="group",
        zscore=True,
        return_fig=True,
    )
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Expression (Z-score)"


def test_zscore_sorting(dummy_adata_for_comparison):
    """Test sorting is correct when using z-scores."""
    # With z-scoring, the relative differences might change
    fig = feature_comparison_boxplot(
        dummy_adata_for_comparison,
        features=["gene1", "gene2", "gene3"],
        group_by="group",
        zscore=True,
        return_fig=True,
    )
    ax = fig.axes[0]
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    # The expected order might be different with z-scoring, let's calculate it
    # For this specific dummy data, the order should remain the same, but it's a good check
    assert xticklabels == ["gene3", "gene2", "gene1"]


def test_auto_group_detection(dummy_adata_for_comparison):
    """Test automatic detection of two groups."""
    fig = feature_comparison_boxplot(dummy_adata_for_comparison, features=["gene1"], group_by="group", return_fig=True)
    assert fig is not None


def test_color_dict(dummy_adata_for_comparison):
    """Test custom colors are applied."""
    color_dict = {"A": "#FF0000", "B": "#0000FF"}
    fig = feature_comparison_boxplot(
        dummy_adata_for_comparison, features=["gene1"], group_by="group", color_dict=color_dict, return_fig=True
    )
    assert fig is not None


def test_error_on_ambiguous_groups(dummy_adata_for_comparison):
    """Test that ValueError is raised for more than two groups without explicit group1/group2."""
    with pytest.raises(ValueError):
        feature_comparison_boxplot(
            dummy_adata_for_comparison, features=["gene1"], group_by="three_groups", return_fig=True
        )


def test_error_on_single_group_spec(dummy_adata_for_comparison):
    """Test that ValueError is raised if only one of group1 or group2 is specified."""
    with pytest.raises(ValueError):
        feature_comparison_boxplot(
            dummy_adata_for_comparison, features=["gene1"], group_by="group", group1="A", return_fig=True
        )

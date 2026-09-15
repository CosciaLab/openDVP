import anndata
import pandas as pd
import pytest

from opendvp.plotting import dual_axis_boxplots


@pytest.fixture
def dummy_adata():
    """Create a dummy AnnData object for testing."""
    obs = pd.DataFrame({"group": ["A", "A", "B", "B"], "feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
    return anndata.AnnData(obs=obs)


def test_dual_axis_boxplots_with_group(dummy_adata):
    """Test plotting with a group_by key."""
    fig = dual_axis_boxplots(dummy_adata, feature_1="feature1", feature_2="feature2", group_by="group", return_fig=True)
    assert fig is not None
    assert len(fig.axes) == 2


def test_dual_axis_boxplots_no_group(dummy_adata):
    """Test plotting without a group_by key."""
    fig = dual_axis_boxplots(dummy_adata, feature_1="feature1", feature_2="feature2", return_fig=True)
    assert fig is not None
    assert len(fig.axes) == 2


def test_missing_feature_raises_error(dummy_adata):
    """Test that a ValueError is raised for a missing feature."""
    with pytest.raises(ValueError):
        dual_axis_boxplots(dummy_adata, feature_1="feature1", feature_2="non_existent_feature", return_fig=True)


def test_missing_group_by_raises_error(dummy_adata):
    """Test that a ValueError is raised for a missing group_by key."""
    with pytest.raises(ValueError):
        dual_axis_boxplots(
            dummy_adata, feature_1="feature1", feature_2="feature2", group_by="non_existent_group", return_fig=True
        )


def test_default_labels(dummy_adata):
    """Test that default y-axis labels are set correctly."""
    fig = dual_axis_boxplots(dummy_adata, feature_1="feature1", feature_2="feature2", return_fig=True)
    ax1, ax2 = fig.axes
    assert ax1.get_ylabel() == "feature1"
    assert ax2.get_ylabel() == "feature2"


def test_custom_labels(dummy_adata):
    """Test that custom y-axis labels are set correctly."""
    fig = dual_axis_boxplots(
        dummy_adata,
        feature_1="feature1",
        feature_2="feature2",
        ylabel1="Custom Label 1",
        ylabel2="Custom Label 2",
        return_fig=True,
    )
    ax1, ax2 = fig.axes
    assert ax1.get_ylabel() == "Custom Label 1"
    assert ax2.get_ylabel() == "Custom Label 2"

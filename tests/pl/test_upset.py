import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from opendvp.plotting.upset import upset


@pytest.fixture
def adata_simple():
    # 6 samples, 3 variables, 2 groups
    X = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [np.nan, 2, 3, 4, 5, 6],
            [1, np.nan, np.nan, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    obs = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]})
    var = pd.DataFrame(index=["v1", "v2", "v3", "v4", "v5", "v6"])
    return AnnData(X=X, obs=obs, var=var)


def test_upset_returns_figure(adata_simple):
    fig = upset(adata_simple, groupby="group", return_fig=True)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_upset_invalid_groupby(adata_simple):
    with pytest.raises(ValueError):
        upset(adata_simple, groupby="not_a_column")


def test_upset_min_presence_fraction_1(adata_simple):
    # Only variables present in all samples of a group are counted
    fig = upset(adata_simple, groupby="group", min_presence_fraction=1.0, return_fig=True)
    assert fig is not None


def test_upset_min_presence_fraction_0(adata_simple):
    # All variables with at least one value in a group are counted
    fig = upset(adata_simple, groupby="group", min_presence_fraction=0.0, return_fig=True)
    assert fig is not None


def test_upset_with_ax(adata_simple):
    fig, ax = plt.subplots()
    result = upset(adata_simple, groupby="group", ax=ax, return_fig=True)
    assert result is fig


def test_upset_without_ax(adata_simple):
    result = upset(adata_simple, groupby="group", return_fig=True)
    assert hasattr(result, "savefig")


def test_upset_kwargs_passed(adata_simple):
    # Should not raise error if extra kwargs are passed
    fig = upset(adata_simple, groupby="group", return_fig=True, show_counts="%d")
    assert fig is not None

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

matplotlib.use("Agg")  # For headless testing2
from opendvp.plotting.density import density


@pytest.fixture
def demo_anndata():
    np.random.seed(0)
    n_samples = 20
    n_proteins = 4
    X = np.random.normal(10, 2, size=(n_samples, n_proteins))
    obs = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=n_samples),
            "batch": np.random.choice(["Batch1", "Batch2"], size=n_samples),
        }
    )
    var = pd.DataFrame(index=[f"Prot{i + 1}" for i in range(n_proteins)])
    return AnnData(X=X, obs=obs, var=var)


def test_density_runs_and_returns_fig(demo_anndata):
    fig = density(demo_anndata, color_by="group", return_fig=True)
    assert fig is not None
    assert hasattr(fig, "savefig")


def test_density_default_palette(demo_anndata):
    # Should not raise and should use default palette
    fig = density(demo_anndata, color_by="batch", return_fig=True)
    assert fig is not None


def test_density_custom_palette(demo_anndata):
    color_dict = {"Batch1": "red", "Batch2": "blue"}
    fig = density(demo_anndata, color_by="batch", color_dict=color_dict, return_fig=True)
    assert fig is not None


def test_density_invalid_color_by(demo_anndata):
    with pytest.raises(ValueError):
        density(demo_anndata, color_by="not_a_column")


def test_density_color_dict_partial(demo_anndata):
    # Only one group in color_dict, should still work for that group
    color_dict = {"A": "red"}
    with pytest.raises(ValueError):
        density(demo_anndata, color_by="group", color_dict=color_dict, return_fig=True)


def test_density_handles_single_group(demo_anndata):
    demo_anndata.obs["single"] = "only"
    fig = density(demo_anndata, color_by="single", return_fig=True)
    assert fig is not None


def test_density_handles_single_sample():
    X = np.array([[1, 2, 3, 4]])
    obs = pd.DataFrame({"group": ["A"]})
    var = pd.DataFrame(index=[f"Prot{i + 1}" for i in range(4)])
    adata = AnnData(X=X, obs=obs, var=var)
    fig = density(adata, color_by="group", return_fig=True)
    assert fig is not None


def test_density_kwargs_passed_to_kdeplot(demo_anndata):
    # Should not error with additional kwargs
    fig = density(demo_anndata, color_by="group", fill=True, return_fig=True)
    assert fig is not None

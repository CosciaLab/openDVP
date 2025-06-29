import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.stats_average_samples import stats_average_samples


@pytest.fixture
def sample_adata_for_averaging() -> ad.AnnData:
    """Creates a sample AnnData object for testing averaging."""
    n_obs = 20
    n_vars = 3
    rng = np.random.default_rng(42)

    obs = pd.DataFrame(
        {
            "condition": ["A"] * (n_obs // 2) + ["B"] * (n_obs // 2),
            "patient": np.tile(["P1", "P2"], n_obs // 2),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    X = np.hstack(
        [
            np.arange(n_obs).reshape(-1, 1),  # Predictable values
            np.ones((n_obs, 1)) * 2,  # Constant values
            rng.random((n_obs, 1)),  # Random values
        ]
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_average_single_category(sample_adata_for_averaging: ad.AnnData) -> None:
    """Test averaging by a single category."""
    adata = sample_adata_for_averaging
    result_adata = stats_average_samples(adata, categories=["condition"])

    assert result_adata.n_obs == 2
    assert result_adata.n_vars == adata.n_vars
    assert sorted(result_adata.obs["condition"].unique()) == ["A", "B"]

    # Calculate expected mean for the predictable gene_0
    mean_A = adata[adata.obs["condition"] == "A", "gene_0"].X.mean()
    mean_B = adata[adata.obs["condition"] == "B", "gene_0"].X.mean()

    # Find the corresponding rows in the result
    result_A_val = result_adata[result_adata.obs["condition"] == "A", "gene_0"].X.item()
    result_B_val = result_adata[result_adata.obs["condition"] == "B", "gene_0"].X.item()

    assert np.isclose(result_A_val, mean_A)
    assert np.isclose(result_B_val, mean_B)


def test_average_multiple_categories(sample_adata_for_averaging: ad.AnnData) -> None:
    """Test averaging by multiple categories."""
    adata = sample_adata_for_averaging
    result_adata = stats_average_samples(adata, categories=["condition", "patient"])

    assert result_adata.n_obs == 4  # 2 conditions * 2 patients
    assert result_adata.n_vars == adata.n_vars

    # Calculate expected mean for a specific group (A, P1)
    mask = (adata.obs["condition"] == "A") & (adata.obs["patient"] == "P1")
    mean_A_P1 = adata[mask, "gene_0"].X.mean()

    # Find the corresponding row in the result
    result_mask = (result_adata.obs["condition"] == "A") & (result_adata.obs["patient"] == "P1")
    result_val = result_adata[result_mask, "gene_0"].X.item()

    assert np.isclose(result_val, mean_A_P1)


def test_provenance_storage(sample_adata_for_averaging: ad.AnnData) -> None:
    """Test that the original adata is stored in .uns."""
    result_adata = stats_average_samples(sample_adata_for_averaging, categories=["condition"])

    assert "pre_averaged_adata" in result_adata.uns
    pre_avg_adata = result_adata.uns["pre_averaged_adata"]
    assert isinstance(pre_avg_adata, ad.AnnData)
    assert pre_avg_adata.shape == sample_adata_for_averaging.shape


def test_returns_copy(sample_adata_for_averaging: ad.AnnData) -> None:
    """Test that the function returns a new AnnData object, not a view."""
    result_adata = stats_average_samples(sample_adata_for_averaging, categories=["condition"])
    assert result_adata is not sample_adata_for_averaging

    # Modify the result and check the original is unchanged
    result_adata.X[0, 0] = -999
    assert not np.isclose(sample_adata_for_averaging.X[0, 0], -999)


def test_invalid_category_raises_error(sample_adata_for_averaging: ad.AnnData) -> None:
    """Test that a ValueError is raised for a non-existent category."""
    with pytest.raises(ValueError, match="Categories not found in adata.obs: \\['non_existent'\\]"):
        stats_average_samples(sample_adata_for_averaging, categories=["condition", "non_existent"])
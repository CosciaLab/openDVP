import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.stats_ttest import stats_ttest


@pytest.fixture
def adata_factory():
    """A factory to create sample AnnData objects for t-test testing."""

    def _create_adata(include_zero_variance: bool = False) -> ad.AnnData:
        n_obs = 50
        obs = pd.DataFrame(
            {"group": ["A"] * (n_obs // 2) + ["B"] * (n_obs - n_obs // 2)},
            index=[f"cell_{i}" for i in range(n_obs)],
        )

        # Create data with predictable differences
        rng = np.random.default_rng(42)
        # Feature 0: Group A >> Group B
        feature0_A = rng.normal(loc=10, scale=1, size=n_obs // 2)
        feature0_B = rng.normal(loc=2, scale=1, size=n_obs - n_obs // 2)
        # Feature 1: No difference
        feature1_A = rng.normal(loc=5, scale=2, size=n_obs // 2)
        feature1_B = rng.normal(loc=5, scale=2, size=n_obs - n_obs // 2)
        # Feature 2: Group B >> Group A
        feature2_A = rng.normal(loc=3, scale=1, size=n_obs // 2)
        feature2_B = rng.normal(loc=12, scale=1, size=n_obs - n_obs // 2)

        features = [
            np.concatenate([feature0_A, feature0_B]).reshape(-1, 1),
            np.concatenate([feature1_A, feature1_B]).reshape(-1, 1),
            np.concatenate([feature2_A, feature2_B]).reshape(-1, 1),
        ]
        var_names = ["diff_A_gt_B", "no_diff", "diff_B_gt_A"]

        if include_zero_variance:
            # Feature 3: Zero variance
            feature3_A = np.full(n_obs // 2, 1.0)
            feature3_B = np.full(n_obs - n_obs // 2, 1.0)
            features.append(np.concatenate([feature3_A, feature3_B]).reshape(-1, 1))
            var_names.append("zero_var")

        X = np.hstack(features)
        var = pd.DataFrame(index=var_names)
        return ad.AnnData(X=X, obs=obs, var=var)

    return _create_adata


@pytest.fixture
def sample_adata(adata_factory) -> ad.AnnData:
    """Provides a standard sample AnnData object without zero-variance features."""
    return adata_factory(include_zero_variance=False)


@pytest.fixture
def adata_with_zero_var(adata_factory) -> ad.AnnData:
    """Provides a sample AnnData object that includes a zero-variance feature."""
    return adata_factory(include_zero_variance=True)


def test_stats_ttest_adds_columns(sample_adata: ad.AnnData) -> None:
    """Test that stats_ttest adds the correct columns to adata.var."""
    result_adata = stats_ttest(sample_adata, grouping="group", group1="A", group2="B")

    expected_cols = ["t_val", "p_val", "mean_diff", "sig", "p_corr", "-log10_p_corr"]

    for col in expected_cols:
        assert col in result_adata.var.columns

    assert result_adata.n_obs == sample_adata.n_obs
    assert result_adata.n_vars == sample_adata.n_vars


def test_stats_ttest_returns_copy(sample_adata: ad.AnnData) -> None:
    """Test that the function returns a copy, not a view."""
    result_adata = stats_ttest(sample_adata, grouping="group", group1="A", group2="B")
    assert result_adata is not sample_adata
    # Modify the result and check the original is unchanged
    result_adata.var["t_val"].iloc[0] = 999
    assert "t_val" not in sample_adata.var.columns


def test_stats_ttest_correctness(sample_adata: ad.AnnData) -> None:
    """Test the correctness of the t-test calculations."""
    result_adata = stats_ttest(sample_adata, grouping="group", group1="A", group2="B", FDR_threshold=0.05)

    # Feature 'diff_A_gt_B': A > B, expect significant p-value, positive t-val and FC
    res_A_gt_B = result_adata.var.loc["diff_A_gt_B"]
    assert res_A_gt_B["p_corr"] < 0.05
    assert res_A_gt_B["sig"]
    assert res_A_gt_B["t_val"] > 0
    assert res_A_gt_B["mean_diff"] > 0

    # Feature 'no_diff': A ≈ B, expect non-significant p-value
    res_no_diff = result_adata.var.loc["no_diff"]
    assert res_no_diff["p_corr"] > 0.05
    assert not res_no_diff["sig"]

    # Feature 'diff_B_gt_A': B > A, expect significant p-value, negative t-val and FC
    res_B_gt_A = result_adata.var.loc["diff_B_gt_A"]
    assert res_B_gt_A["p_corr"] < 0.05
    assert res_B_gt_A["sig"]
    assert res_B_gt_A["t_val"] < 0
    assert res_B_gt_A["mean_diff"] < 0


def test_stats_ttest_nan_output_on_zero_variance(adata_with_zero_var: ad.AnnData) -> None:
    """Test that features with zero variance are handled gracefully (produce NaNs)."""
    result_adata = stats_ttest(adata_with_zero_var, grouping="group", group1="A", group2="B")

    res_zero_var = result_adata.var.loc["zero_var"]
    # t-test on constant arrays results in NaN
    assert np.isnan(res_zero_var["t_val"])
    assert np.isnan(res_zero_var["p_val"])
    assert np.isnan(res_zero_var["p_corr"])
    assert res_zero_var["mean_diff"] == 0.0  # 1.0 - 1.0 = 0.0
    assert not res_zero_var["sig"]


def test_stats_ttest_invalid_grouping_col(sample_adata: ad.AnnData) -> None:
    """Test that a KeyError is raised for a non-existent grouping column."""
    with pytest.raises(KeyError):
        stats_ttest(sample_adata, grouping="non_existent_group", group1="A", group2="B")


def test_stats_ttest_invalid_group_name_produces_nans(sample_adata: ad.AnnData) -> None:
    """Test that a non-existent group name results in NaN values without error."""
    with pytest.raises(ValueError, match="Given groups not found in"):
        stats_ttest(sample_adata, grouping="group", group1="A", group2="C")
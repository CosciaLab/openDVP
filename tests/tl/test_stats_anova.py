import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl.stats_anova import stats_anova


@pytest.fixture
def anova_adata_factory():
    """A factory to create sample AnnData objects for ANOVA testing."""

    def _create_adata(n_obs_per_group=20, groups=("A", "B", "C")) -> ad.AnnData:
        n_groups = len(groups)
        n_obs = n_obs_per_group * n_groups

        obs = pd.DataFrame({"group": np.repeat(groups, n_obs_per_group)}, index=[f"cell_{i}" for i in range(n_obs)])

        rng = np.random.default_rng(42)

        # Feature 1: Clear difference (A vs B, A vs C)
        feat1_A = rng.normal(loc=2, scale=1, size=n_obs_per_group)
        feat1_B = rng.normal(loc=10, scale=1, size=n_obs_per_group)
        feat1_C = rng.normal(loc=10, scale=1, size=n_obs_per_group)
        feature_diff = np.concatenate([feat1_A, feat1_B, feat1_C])

        # Feature 2: No difference
        feature_nodiff = rng.normal(loc=5, scale=2, size=n_obs)

        # Feature 3: All NaNs
        feature_all_nan = np.full(n_obs, np.nan)

        # Feature 4: Zero variance across all groups
        feature_zero_var = np.full(n_obs, 1.0)

        # Feature 5: Zero variance in one group
        feat5_A = rng.normal(loc=5, scale=1, size=n_obs_per_group)
        feat5_B = np.full(n_obs_per_group, 1.0)  # Zero variance in group B
        feat5_C = rng.normal(loc=5, scale=1, size=n_obs_per_group)
        feature_zero_var_in_group = np.concatenate([feat5_A, feat5_B, feat5_C])

        X = np.vstack([feature_diff, feature_nodiff, feature_all_nan, feature_zero_var, feature_zero_var_in_group]).T

        var = pd.DataFrame(index=["diff_A_vs_BC", "no_diff", "all_nan", "zero_var_all", "zero_var_in_group"])

        return ad.AnnData(X=X, obs=obs, var=var)

    return _create_adata


@pytest.fixture
def sample_anova_adata(anova_adata_factory) -> ad.AnnData:
    """Provides a standard sample AnnData object for ANOVA tests."""
    return anova_adata_factory()


def test_stats_anova_adds_columns(sample_anova_adata: ad.AnnData) -> None:
    """Test that stats_anova adds the correct columns to adata.var."""
    result_adata = stats_anova(sample_anova_adata, grouping="group")

    expected_cols = ["anova_F", "anova_p-unc", "anova_sig_BH", "anova_p_corr", "-log10_anova_p_corr"]
    for col in expected_cols:
        assert col in result_adata.var.columns


def test_stats_anova_returns_copy(sample_anova_adata: ad.AnnData) -> None:
    """Test that the function returns a copy, not a view."""
    result_adata = stats_anova(sample_anova_adata, grouping="group")
    assert result_adata is not sample_anova_adata
    result_adata.var["anova_F"].iloc[0] = 999
    assert "anova_F" not in sample_anova_adata.var.columns


def test_stats_anova_correctness(sample_anova_adata: ad.AnnData) -> None:
    """Test the correctness of the ANOVA calculations."""
    result_adata = stats_anova(sample_anova_adata, grouping="group", FDR_threshold=0.05)

    # Feature 'diff_A_vs_BC': A vs B/C, expect significant p-value
    res_diff = result_adata.var.loc["diff_A_vs_BC"]
    assert res_diff["anova_sig_BH"]
    assert res_diff["anova_p_corr"] < 0.05

    # Feature 'no_diff': No difference, expect non-significant p-value
    res_nodiff = result_adata.var.loc["no_diff"]
    assert not res_nodiff["anova_sig_BH"]
    assert res_nodiff["anova_p_corr"] > 0.05


def test_posthoc_results_storage_and_content(sample_anova_adata: ad.AnnData) -> None:
    """Test that post-hoc results are stored correctly in adata.uns."""
    result_adata = stats_anova(sample_anova_adata, grouping="group", FDR_threshold=0.05)

    assert "anova_posthoc" in result_adata.uns
    posthoc_df = result_adata.uns["anova_posthoc"]
    assert isinstance(posthoc_df, pd.DataFrame)

    # Post-hoc should only be run on the significant feature
    assert posthoc_df["feature"].unique().tolist() == ["diff_A_vs_BC"]

    # Check the specific comparisons (A vs B/C should be significant, B vs C not)
    posthoc_df = posthoc_df.set_index(["A", "B"])
    assert posthoc_df.loc[("A", "B"), "p-tukey"] < 0.05
    assert posthoc_df.loc[("A", "C"), "p-tukey"] < 0.05
    assert posthoc_df.loc[("B", "C"), "p-tukey"] > 0.05


def test_posthoc_disabled(sample_anova_adata: ad.AnnData) -> None:
    """Test that no post-hoc is run when posthoc=None."""
    result_adata = stats_anova(sample_anova_adata, grouping="group", posthoc=None)
    assert "anova_posthoc" not in result_adata.uns


def test_no_significant_features_no_posthoc(sample_anova_adata: ad.AnnData) -> None:
    """Test that no post-hoc is run if no features are significant."""
    result_adata = stats_anova(sample_anova_adata, grouping="group", FDR_threshold=1e-50)
    assert "anova_posthoc" not in result_adata.uns


def test_handling_problematic_features(sample_anova_adata: ad.AnnData) -> None:
    """Test that features with NaNs or zero variance are handled gracefully."""
    result_adata = stats_anova(sample_anova_adata, grouping="group")

    for feature in ["all_nan", "zero_var_all", "zero_var_in_group"]:
        res = result_adata.var.loc[feature]
        assert np.isnan(res["anova_F"])
        assert np.isnan(res["anova_p-unc"])
        assert np.isnan(res["anova_p_corr"])
        assert not res["anova_sig_BH"]


def test_invalid_grouping_col(sample_anova_adata: ad.AnnData) -> None:
    """Test that a ValueError is raised for a non-existent grouping column."""
    with pytest.raises(ValueError, match="Grouping column 'non_existent_group' not found"):
        stats_anova(sample_anova_adata, grouping="non_existent_group")
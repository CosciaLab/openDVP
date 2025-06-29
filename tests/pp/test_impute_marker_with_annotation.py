import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from opendvp.pp import impute_marker_with_annotation


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Creates a sample AnnData object for testing."""
    X = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [0, 0, 0]], dtype=np.float32
    )
    obs = pd.DataFrame(
        {"annotation": [True, False, True, False, True]},
        index=[f"cell_{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=["gene_A", "gene_B", "gene_C"])
    return ad.AnnData(X, obs=obs, var=var)


def test_basic_imputation(sample_adata: ad.AnnData):
    """Test basic functionality of imputation."""
    adata_original_X = sample_adata.X.copy()

    # Impute 'gene_B' where 'annotation' is True
    adata_imputed = impute_marker_with_annotation(
        sample_adata,
        target_variable="gene_B",
        target_annotation_column="annotation",
        quantile_for_imputation=0.1,  # Use a low quantile for predictability
    )

    # 1. Check if a copy is returned and original is unchanged
    assert adata_imputed is not sample_adata
    np.testing.assert_array_equal(sample_adata.X, adata_original_X)

    # 2. Calculate expected imputation value
    # Original 'gene_B' values: [2, 5, 8, 11, 0]
    expected_value = np.quantile([2, 5, 8, 11, 0], 0.1)

    # 3. Check if the correct cells were imputed (indices 0, 2, 4)
    assert adata_imputed.X[0, 1] == pytest.approx(expected_value)
    assert adata_imputed.X[2, 1] == pytest.approx(expected_value)
    assert adata_imputed.X[4, 1] == pytest.approx(expected_value)

    # 4. Check if non-imputed cells are unchanged
    assert adata_imputed.X[1, 1] == sample_adata.X[1, 1]  # 5
    assert adata_imputed.X[3, 1] == sample_adata.X[3, 1]  # 11

    # 5. Check other columns are unchanged
    np.testing.assert_array_equal(adata_imputed.X[:, 0], sample_adata.X[:, 0])
    np.testing.assert_array_equal(adata_imputed.X[:, 2], sample_adata.X[:, 2])


def test_imputation_with_sparse_matrix(sample_adata: ad.AnnData):
    """Test imputation works correctly with a sparse matrix."""
    sample_adata.X = csr_matrix(sample_adata.X)

    adata_imputed = impute_marker_with_annotation(
        sample_adata,
        target_variable="gene_A",
        target_annotation_column="annotation",
        quantile_for_imputation=0.5,
    )

    # Check that output is a dense array as per implementation
    assert isinstance(adata_imputed.X, np.ndarray)
    expected_value = np.median([1, 4, 7, 10, 0])  # median is 4
    assert adata_imputed.X[0, 0] == pytest.approx(expected_value)


def test_no_rows_to_impute(sample_adata: ad.AnnData):
    """Test behavior when the annotation mask is all False."""
    sample_adata.obs["annotation"] = False
    adata_original_X = sample_adata.X.copy()

    adata_imputed = impute_marker_with_annotation(
        sample_adata, "gene_C", "annotation"
    )
    np.testing.assert_array_equal(adata_imputed.X, adata_original_X)


def test_invalid_quantile(sample_adata: ad.AnnData):
    """Test that an invalid quantile raises a ValueError."""
    with pytest.raises(ValueError, match="Quantile should be between 0 and 1"):
        impute_marker_with_annotation(sample_adata, "gene_A", "annotation", -0.1)


def test_missing_variable(sample_adata: ad.AnnData):
    """Test that a missing target variable raises a ValueError."""
    with pytest.raises(ValueError, match="Variable missing_gene not found"):
        impute_marker_with_annotation(sample_adata, "missing_gene", "annotation")


def test_missing_annotation_column(sample_adata: ad.AnnData):
    """Test that a missing annotation column raises a ValueError."""
    with pytest.raises(ValueError, match="Annotation column missing_annotation not found"):
        impute_marker_with_annotation(sample_adata, "gene_A", "missing_annotation")
"""Tests for `scimap_spatial_lda`.

Builds neighbourhoods out of spatial coordinates and fits a gensim LDA over them, so the fixture
needs X/Y centroids, a phenotype column and an imageid. `num_motifs` is kept small to keep the
fit quick.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl import scimap_spatial_lda


@pytest.fixture
def spatial_adata() -> ad.AnnData:
    """Three spatially separated patches, each dominated by a different phenotype."""
    rng = np.random.default_rng(11)
    centres = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0)]
    dominant = ["T_cell", "B_cell", "Macrophage"]
    x, y, phenotypes = [], [], []
    for (cx, cy), main in zip(centres, dominant, strict=True):
        for _ in range(40):
            x.append(cx + rng.normal(scale=10))
            y.append(cy + rng.normal(scale=10))
            # mostly the dominant type, with some mixing so the motifs are not degenerate
            phenotypes.append(main if rng.random() < 0.8 else rng.choice(dominant))
    n = len(x)
    return ad.AnnData(
        X=rng.normal(size=(n, 5)),
        obs=pd.DataFrame(
            {"X_centroid": x, "Y_centroid": y, "phenotype": phenotypes, "imageid": ["img1"] * n},
            index=[f"cell{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"M{i}" for i in range(5)]),
    )


def test_stores_weights_in_uns(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, num_motifs=3, verbose=False)
    assert "spatial_lda" in result.uns


def test_label_is_configurable(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, num_motifs=3, label="motifs", verbose=False)
    assert "motifs" in result.uns
    assert "motifs_probability" in result.uns


def test_one_weight_row_per_cell(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, num_motifs=3, verbose=False)
    weights = result.uns["spatial_lda"]
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == spatial_adata.n_obs
    assert weights.index.tolist() == spatial_adata.obs.index.tolist()


def test_num_motifs_sets_the_weight_columns(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, num_motifs=4, verbose=False)
    assert result.uns["spatial_lda"].shape[1] == 4


def test_weights_are_a_distribution_over_motifs(spatial_adata: ad.AnnData):
    """Rows sum to at most 1, and slightly under it: gensim drops topics below its minimum
    probability and the implementation fills those with zero rather than renormalising."""
    result = scimap_spatial_lda(spatial_adata, num_motifs=3, verbose=False)
    weights = result.uns["spatial_lda"].to_numpy()
    totals = weights.sum(axis=1)
    assert (weights >= 0).all()
    assert (totals <= 1.0 + 1e-6).all()
    assert (totals > 0.95).all()


def test_probability_table_is_indexed_by_phenotype(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, num_motifs=3, verbose=False)
    probabilities = result.uns["spatial_lda_probability"]
    assert set(probabilities.index) == set(spatial_adata.obs["phenotype"].unique())


def test_knn_method_also_works(spatial_adata: ad.AnnData):
    result = scimap_spatial_lda(spatial_adata, method="knn", knn=5, num_motifs=3, verbose=False)
    assert len(result.uns["spatial_lda"]) == spatial_adata.n_obs


def test_is_reproducible_for_a_fixed_random_state(spatial_adata: ad.AnnData):
    first = scimap_spatial_lda(spatial_adata.copy(), num_motifs=3, random_state=0, verbose=False)
    second = scimap_spatial_lda(spatial_adata.copy(), num_motifs=3, random_state=0, verbose=False)
    assert np.allclose(first.uns["spatial_lda"].to_numpy(), second.uns["spatial_lda"].to_numpy())


def test_unknown_method_raises(spatial_adata: ad.AnnData):
    with pytest.raises(ValueError, match="method"):
        scimap_spatial_lda(spatial_adata, method="not_a_method", num_motifs=3, verbose=False)

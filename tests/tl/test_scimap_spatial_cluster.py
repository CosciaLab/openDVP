"""Tests for `scimap_spatial_cluster`.

Defaults are `method="kmeans"`, `use_raw=True` and `log=True`, so the fixture has to carry a
`.raw` with non-negative values for `np.log1p` to be meaningful.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl import scimap_spatial_cluster


@pytest.fixture
def clusterable() -> ad.AnnData:
    """Sixty cells in three well-separated blobs over eight markers."""
    rng = np.random.default_rng(3)
    blobs = []
    for centre in (2.0, 8.0, 14.0):
        blobs.append(rng.normal(loc=centre, scale=0.3, size=(20, 8)))
    X = np.vstack(blobs)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "phenotype": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
                "imageid": ["img1"] * 60,
            },
            index=[f"cell{i}" for i in range(60)],
        ),
        var=pd.DataFrame(index=[f"M{i}" for i in range(8)]),
    )
    adata.raw = adata
    return adata


def test_adds_the_method_named_column(clusterable: ad.AnnData):
    result = scimap_spatial_cluster(clusterable, k=3, verbose=False)
    assert "kmeans" in result.obs.columns


def test_label_overrides_the_column_name(clusterable: ad.AnnData):
    result = scimap_spatial_cluster(clusterable, k=3, label="rcn", verbose=False)
    assert "rcn" in result.obs.columns
    assert "kmeans" not in result.obs.columns


def test_every_cell_is_assigned(clusterable: ad.AnnData):
    result = scimap_spatial_cluster(clusterable, k=3, verbose=False)
    assert result.obs["kmeans"].notna().all()


def test_k_controls_the_number_of_clusters(clusterable: ad.AnnData):
    result = scimap_spatial_cluster(clusterable, k=3, verbose=False)
    assert result.obs["kmeans"].nunique() == 3


def test_recovers_the_planted_blobs(clusterable: ad.AnnData):
    """Three separated blobs of twenty cells should come back as three groups of twenty."""
    result = scimap_spatial_cluster(clusterable, k=3, verbose=False)
    assert sorted(result.obs["kmeans"].value_counts().tolist()) == [20, 20, 20]


def test_is_reproducible_for_a_fixed_random_state(clusterable: ad.AnnData):
    first = scimap_spatial_cluster(clusterable.copy(), k=3, random_state=0, verbose=False)
    second = scimap_spatial_cluster(clusterable.copy(), k=3, random_state=0, verbose=False)
    assert first.obs["kmeans"].tolist() == second.obs["kmeans"].tolist()


def test_runs_without_raw_when_use_raw_is_false(clusterable: ad.AnnData):
    del clusterable.raw
    result = scimap_spatial_cluster(clusterable, k=3, use_raw=False, verbose=False)
    assert result.obs["kmeans"].nunique() == 3


def test_sub_clustering_refines_within_a_phenotype(clusterable: ad.AnnData):
    result = scimap_spatial_cluster(
        clusterable,
        k=2,
        sub_cluster=True,
        sub_cluster_column="phenotype",
        sub_cluster_group=["A"],
        verbose=False,
    )
    labels = result.obs["kmeans"]
    # cells outside group A keep their phenotype name, group A gets sub-labels derived from it
    assert set(labels[20:]) == {"B", "C"}
    assert labels[:20].nunique() == 2

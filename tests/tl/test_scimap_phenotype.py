"""Tests for `scimap_phenotype`.

The phenotype workflow is a DataFrame shaped like scimap's `phenotype_workflow.csv`: column 0 is
the parent group (`all`, or another phenotype), column 1 is the phenotype name, and the remaining
columns are markers holding `pos` / `neg` / `anypos` / `anyneg` / `allpos` / `allneg`. The data is
expected to be rescaled so that 0.5 is the gate.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.tl import scimap_phenotype

MARKERS = ["panCK", "CD3e", "CD8", "CD20"]


@pytest.fixture
def workflow() -> pd.DataFrame:
    """Epithelial from panCK; immune from any T/B marker; then T and B under Immune."""
    return pd.DataFrame(
        [
            ["all", "Epithelial", "pos", None, None, None],
            ["all", "Immune", None, "anypos", "anypos", "anypos"],
            ["Immune", "T_cell", None, "pos", None, None],
            ["Immune", "B_cell", None, None, None, "pos"],
        ],
        columns=["parent", "phenotype", *MARKERS],
    )


@pytest.fixture
def gated_adata() -> ad.AnnData:
    """Six cells: two clearly epithelial, two T cells, two B cells, all rescaled around 0.5."""
    X = np.array(
        [
            [0.9, 0.1, 0.1, 0.1],  # epithelial
            [0.8, 0.2, 0.1, 0.1],  # epithelial
            [0.1, 0.9, 0.8, 0.1],  # T cell
            [0.1, 0.8, 0.9, 0.2],  # T cell
            [0.1, 0.1, 0.1, 0.9],  # B cell
            [0.2, 0.2, 0.1, 0.8],  # B cell
        ]
    )
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"imageid": ["img1"] * 6}, index=[f"cell{i}" for i in range(6)]),
        var=pd.DataFrame(index=MARKERS),
    )


def test_adds_the_phenotype_column(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    assert "phenotype" in result.obs.columns
    assert len(result.obs["phenotype"]) == gated_adata.n_obs


def test_label_is_configurable(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, label="my_cells", verbose=False)
    assert "my_cells" in result.obs.columns


def test_every_cell_is_assigned(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    assert result.obs["phenotype"].notna().all()


def test_epithelial_and_immune_cells_are_separated(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    calls = result.obs["phenotype"].tolist()
    assert calls[0] == calls[1] == "Epithelial"
    assert "Epithelial" not in calls[2:]


def test_immune_cells_are_refined_into_subtypes(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    calls = result.obs["phenotype"].tolist()
    assert calls[2] == calls[3] == "T_cell"
    assert calls[4] == calls[5] == "B_cell"


def test_results_are_ordered_like_obs(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    """The implementation reindexes onto adata.obs.index, so order must survive."""
    shuffled = gated_adata[[5, 0, 3, 1, 4, 2]].copy()
    result = scimap_phenotype(shuffled, phenotype=workflow, verbose=False)
    assert result.obs.index.tolist() == shuffled.obs.index.tolist()
    assert result.obs.loc["cell0", "phenotype"] == "Epithelial"


def test_rare_phenotypes_can_be_dropped_by_percentage(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    """Two of six cells is 33%, so a 50% threshold should discard every subtype."""
    result = scimap_phenotype(gated_adata, phenotype=workflow, pheno_threshold_percent=50, verbose=False)
    assert "T_cell" not in result.obs["phenotype"].tolist()


def test_rare_phenotypes_can_be_dropped_by_count(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    result = scimap_phenotype(gated_adata, phenotype=workflow, pheno_threshold_abs=3, verbose=False)
    assert "T_cell" not in result.obs["phenotype"].tolist()


def test_a_higher_gate_makes_calls_stricter(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    # copy first: scimap_phenotype writes into the object it is handed, see the test below
    lenient = scimap_phenotype(gated_adata.copy(), phenotype=workflow, gate=0.5, verbose=False)
    strict = scimap_phenotype(gated_adata.copy(), phenotype=workflow, gate=0.95, verbose=False)
    assert lenient.obs["phenotype"].tolist() != strict.obs["phenotype"].tolist()
    assert set(strict.obs["phenotype"]) == {"Unknown"}


def test_modifies_the_input_in_place(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    """Unlike stats_anova / stats_ttest, this one does not copy - worth pinning until it does."""
    result = scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    assert result is gated_adata
    assert "phenotype" in gated_adata.obs.columns


def test_does_not_emit_pandas_future_warnings(gated_adata: ad.AnnData, workflow: pd.DataFrame):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scimap_phenotype(gated_adata, phenotype=workflow, verbose=False)
    ours = [w for w in caught if "scimap_phenotype.py" in str(w.filename)]
    assert ours == [], [str(w.message) for w in ours]

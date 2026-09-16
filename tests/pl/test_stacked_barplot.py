import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from opendvp.pl.stacked_barplot import stacked_barplot

COLORS = {"T_cell": "#1f77b4", "B_cell": "#ff7f0e"}


@pytest.fixture
def composition() -> pd.DataFrame:
    """Two motifs: motif_0 is 3:1 T cells, motif_1 is 1:1."""
    return pd.DataFrame(
        {
            "phenotype": ["T_cell"] * 3 + ["B_cell"] + ["T_cell"] * 2 + ["B_cell"] * 2,
            "rcn": ["motif_0"] * 4 + ["motif_1"] * 4,
        }
    )


def test_returns_figure_and_axes(composition: pd.DataFrame):
    fig, ax = stacked_barplot(composition, phenotype_col="phenotype", rcn_col="rcn", phenotype_colors=COLORS)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_normalised_bars_sum_to_one(composition: pd.DataFrame):
    fig, ax = stacked_barplot(composition, phenotype_col="phenotype", rcn_col="rcn", phenotype_colors=COLORS)
    totals: dict[str, float] = {}
    for container in ax.containers:
        for bar in container:
            totals[bar.get_x()] = totals.get(bar.get_x(), 0) + bar.get_height()
    assert all(abs(total - 1.0) < 1e-9 for total in totals.values())
    plt.close(fig)


def test_counts_are_used_when_not_normalised(composition: pd.DataFrame):
    fig, ax = stacked_barplot(
        composition, phenotype_col="phenotype", rcn_col="rcn", phenotype_colors=COLORS, normalize=False
    )
    heights = sorted(bar.get_height() for container in ax.containers for bar in container)
    assert heights == [1.0, 2.0, 2.0, 3.0]
    assert ax.get_ylabel() == "Count"
    plt.close(fig)


def test_uses_the_supplied_axes(composition: pd.DataFrame):
    _, existing = plt.subplots()
    fig, ax = stacked_barplot(
        composition, phenotype_col="phenotype", rcn_col="rcn", phenotype_colors=COLORS, ax=existing
    )
    assert ax is existing
    plt.close(fig)


def test_phenotypes_without_a_colour_are_skipped(composition: pd.DataFrame):
    fig, ax = stacked_barplot(
        composition, phenotype_col="phenotype", rcn_col="rcn", phenotype_colors={"T_cell": "#1f77b4"}
    )
    labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert labels == {"T_cell"}
    plt.close(fig)

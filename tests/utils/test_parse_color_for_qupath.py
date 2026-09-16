import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.utils import parse_color_for_qupath


@pytest.fixture
def adata() -> ad.AnnData:
    """Three categories in obs, which is what the default palette is zipped against."""
    obs = pd.DataFrame(
        {"celltype": pd.Categorical(["T_cell", "B_cell", "Macrophage", "T_cell"])},
        index=[f"c{i}" for i in range(4)],
    )
    return ad.AnnData(X=np.zeros((4, 2)), obs=obs, var=pd.DataFrame(index=["A", "B"]))


def test_defaults_cover_every_category(adata: ad.AnnData):
    colors = parse_color_for_qupath(None, adata, "celltype")
    assert set(colors) == set(adata.obs["celltype"].cat.categories)


def test_defaults_are_rgb_triples_in_0_255(adata: ad.AnnData):
    for rgb in parse_color_for_qupath(None, adata, "celltype").values():
        assert len(rgb) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)


def test_defaults_cycle_when_categories_outnumber_the_palette():
    obs = pd.DataFrame({"g": pd.Categorical([f"cat{i}" for i in range(7)])}, index=[f"c{i}" for i in range(7)])
    adata = ad.AnnData(X=np.zeros((7, 2)), obs=obs, var=pd.DataFrame(index=["A", "B"]))
    colors = parse_color_for_qupath(None, adata, "g")
    assert len(colors) == 7
    # the built-in palette has five entries, so the sixth reuses the first
    assert colors["cat5"] == colors["cat0"]


def test_hex_is_converted_to_0_255(adata: ad.AnnData):
    colors = parse_color_for_qupath({"T_cell": "#ff0000"}, adata, "celltype")
    assert colors["T_cell"] == [255, 0, 0]


def test_short_hex_is_accepted(adata: ad.AnnData):
    assert parse_color_for_qupath({"T_cell": "#0f0"}, adata, "celltype")["T_cell"] == [0, 255, 0]


def test_fraction_tuples_are_scaled(adata: ad.AnnData):
    colors = parse_color_for_qupath({"T_cell": (1.0, 0.0, 0.5)}, adata, "celltype")
    assert colors["T_cell"] == [255, 0, 127]


def test_rgb_lists_pass_through_unchanged(adata: ad.AnnData):
    assert parse_color_for_qupath({"T_cell": [12, 34, 56]}, adata, "celltype")["T_cell"] == [12, 34, 56]


def test_a_custom_dict_is_not_padded_with_defaults(adata: ad.AnnData):
    """Only the keys given are returned - categories without a colour are the caller's problem."""
    colors = parse_color_for_qupath({"T_cell": "#ff0000"}, adata, "celltype")
    assert set(colors) == {"T_cell"}


@pytest.mark.parametrize(
    "bad",
    [
        "red",  # named colours are not handled
        "#12345",  # not 3 or 6 hex digits
        (1.0, 0.0),  # wrong length
        [300, 0, 0],  # out of range
        [1.0, 0.0, 0.0],  # floats in a list, not a tuple
        None,
    ],
)
def test_invalid_colour_formats_raise(adata: ad.AnnData, bad: object):
    with pytest.raises(ValueError, match="Invalid color format"):
        parse_color_for_qupath({"T_cell": bad}, adata, "celltype")


def test_non_categorical_obs_column_raises(adata: ad.AnnData):
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    with pytest.raises(AttributeError):
        parse_color_for_qupath(None, adata, "celltype")

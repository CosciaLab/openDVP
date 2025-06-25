import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.io import adata_to_voronoi


@pytest.fixture
def basic_adata():
    obs = pd.DataFrame({
        "X_centroid": [1.0, 2.0, 3.0, 2.5, 1.5],
        "Y_centroid": [1.0, 1.0, 1.0, 2.0, 2.0],
        "celltype": pd.Series(["A", "B", "A", "B", "A"], dtype="category")
    })
    return ad.AnnData(obs=obs)


def test_returns_geodataframe(basic_adata):
    gdf = adata_to_voronoi(basic_adata, classify_by="celltype")
    assert gdf is not None
    assert "geometry" in gdf.columns
    assert gdf.geometry.notnull().all()


def test_object_type_column(basic_adata):
    gdf = adata_to_voronoi(basic_adata)
    assert "objectType" in gdf.columns
    assert (gdf["objectType"] == "detection").all()


def test_invalid_input_type():
    with pytest.raises(ValueError, match="adata must be an instance of anndata.AnnData"):
        adata_to_voronoi("not_an_adata_object")


def test_missing_centroids():
    adata = ad.AnnData(obs=pd.DataFrame({"X": [1], "Y": [2]}))
    with pytest.raises(ValueError, match="X_centroid or Y_centroid not found"):
        adata_to_voronoi(adata)


def test_classify_by_missing_column(basic_adata):
    adata = basic_adata.copy()
    adata.obs.drop(columns=["celltype"], inplace=True)
    with pytest.raises(ValueError, match="celltype not in adata.obs.columns"):
        adata_to_voronoi(adata, classify_by="celltype")


def test_nan_in_classify_by(basic_adata):
    adata = basic_adata.copy()
    adata.obs.loc[0, "celltype"] = np.nan
    with pytest.raises(ValueError, match="contains NaN values"):
        adata_to_voronoi(adata, classify_by="celltype")


def test_color_dict_validation(basic_adata):
    with pytest.raises(ValueError, match="provided color_dict is not a dict"):
        adata_to_voronoi(basic_adata, classify_by="celltype", color_dict="notadict")


def test_merge_adjacent_shapes(basic_adata):
    gdf = adata_to_voronoi(basic_adata, classify_by="celltype", merge_adjacent_shapes=True)
    assert "class" in gdf.columns
    assert gdf.geometry.is_valid.all()

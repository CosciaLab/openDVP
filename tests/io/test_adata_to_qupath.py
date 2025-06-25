import geopandas as gpd
import pandas as pd
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

from opendvp.io import adata_to_qupath


@pytest.fixture
def adata_with_classes():
    obs = pd.DataFrame({
        "CellID": [1, 2, 3],
        "celltype": ["A", "B", "A"]
    })
    obs["celltype"] = obs["celltype"].astype("category")
    return AnnData(obs=obs)


@pytest.fixture
def simple_gdf():
    return gpd.GeoDataFrame({
        "CellID": [1, 2, 3],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]
    })


def test_basic_export_with_classification(adata_with_classes, simple_gdf):
    color_dict = {"A": [255, 0, 0], "B": [0, 255, 0]}
    gdf_out = adata_to_qupath(
        adata=adata_with_classes,
        geodataframe=simple_gdf,
        adataobs_on="CellID",
        gdf_on="CellID",
        classify_by="celltype",
        color_dict=color_dict,
        simplify_value=None,
        save_as_detection=True
    )
    assert isinstance(gdf_out, gpd.GeoDataFrame)
    assert "classification" in gdf_out.columns
    assert "objectType" in gdf_out.columns
    assert all(entry["name"] in {"A", "B"} for entry in gdf_out["classification"])


def test_simplify_geometry(adata_with_classes):
    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    gdf = gpd.GeoDataFrame({"CellID": [1, 2, 3]}, geometry=[poly, poly, poly])
    simplified = adata_to_qupath(
        adata=adata_with_classes,
        geodataframe=gdf,
        adataobs_on="CellID",
        gdf_on="CellID",
        classify_by="celltype",
        color_dict={"A": [0, 0, 0], "B": [1, 1, 1]},
        simplify_value=1.0,
    )
    assert simplified["geometry"].is_valid.all()


def test_use_gdf_index(adata_with_classes):
    gdf = gpd.GeoDataFrame(index=[1, 2, 3], geometry=[Point(0, 0), Point(1, 1), Point(2, 2)])
    out = adata_to_qupath(
        adata=adata_with_classes,
        geodataframe=gdf,
        adataobs_on="CellID",
        gdf_on=None,
        gdf_index=True,
        classify_by="celltype",
        color_dict={"A": [0, 0, 255], "B": [255, 255, 0]},
    )
    assert "classification" in out.columns


def test_no_matching_ids_raises_error(adata_with_classes):
    gdf = gpd.GeoDataFrame({"CellID": [10, 11, 12]}, geometry=[Point(0, 0), Point(1, 1), Point(2, 2)])
    with pytest.raises(ValueError, match="No matching values between adata and geodataframe"):
        adata_to_qupath(
            adata=adata_with_classes,
            geodataframe=gdf,
            adataobs_on="CellID",
            gdf_on="CellID"
        )


def test_missing_column_raises_error(adata_with_classes, simple_gdf):
    with pytest.raises(ValueError, match="not in adata.obs.columns"):
        adata_to_qupath(
            adata=adata_with_classes,
            geodataframe=simple_gdf,
            adataobs_on="MissingID",
            gdf_on="CellID"
        )


def test_color_dict_validation(adata_with_classes, simple_gdf):
    with pytest.raises(ValueError, match="color_dict is not a dict"):
        adata_to_qupath(
            adata=adata_with_classes,
            geodataframe=simple_gdf,
            adataobs_on="CellID",
            gdf_on="CellID",
            classify_by="celltype",
            color_dict="not a dict"
        )


def test_autoconvert_classify_to_category(simple_gdf):
    obs = pd.DataFrame({
        "CellID": [1, 2, 3],
        "celltype": ["A", "B", "A"]  # Not categorical
    })
    adata = AnnData(obs=obs)
    result = adata_to_qupath(
        adata=adata,
        geodataframe=simple_gdf,
        adataobs_on="CellID",
        gdf_on="CellID",
        classify_by="celltype",
        color_dict={"A": [0, 0, 255], "B": [0, 255, 0]}
    )
    assert isinstance(result, gpd.GeoDataFrame)
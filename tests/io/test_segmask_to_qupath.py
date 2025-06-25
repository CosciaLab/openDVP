import geopandas as gpd
import numpy as np
import pytest
import tifffile

from opendvp.io import segmask_to_qupath


@pytest.fixture
def dummy_mask(tmp_path):
    array = np.zeros((100, 100), dtype=np.uint8)
    array[10:30, 10:30] = 1
    array[50:80, 50:80] = 2
    file_path = tmp_path / "test_mask.tif"
    tifffile.imwrite(str(file_path), array)
    return str(file_path)


def test_returns_geodataframe(dummy_mask):
    gdf = segmask_to_qupath(path_to_mask=dummy_mask)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf.columns
    assert "objectType" in gdf.columns
    assert (gdf["objectType"] == "detection").all()


def test_simplify_geometry(dummy_mask):
    gdf = segmask_to_qupath(path_to_mask=dummy_mask, simplify_value=0.5)
    assert gdf.geometry.is_valid.all() # type: ignore


def test_disable_simplification(dummy_mask):
    gdf = segmask_to_qupath(path_to_mask=dummy_mask, simplify_value=None)
    assert isinstance(gdf, gpd.GeoDataFrame)


def test_invalid_path_type_raises():
    with pytest.raises(ValueError, match="path_to_mask must be a string"):
        segmask_to_qupath(path_to_mask=123)


def test_invalid_file_extension_raises(tmp_path):
    invalid_file = tmp_path / "not_a_tiff.txt"
    invalid_file.write_text("fake data")
    with pytest.raises(ValueError, match="must end with .tif"):
        segmask_to_qupath(path_to_mask=str(invalid_file))


def test_missing_dependency(monkeypatch, dummy_mask):
    monkeypatch.setitem(__import__("sys").modules, "spatialdata", None)
    with pytest.raises(ImportError, match="spatialdata"):
        segmask_to_qupath(path_to_mask=dummy_mask)

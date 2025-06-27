import ast

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from opendvp.tl.filter_by_annotation import filter_by_annotation


@pytest.fixture
def sample_adata() -> ad.AnnData:
    """Create a sample AnnData object for testing."""
    # Create deterministic coordinates for cells
    coords = {
        "in_A_only": (10, 10),
        "in_B_only": (90, 90),
        "in_C_only": (10, 90),
        "in_A_and_B_overlap": (60, 60), # In the overlap of A and B
        "unannotated": (150, 150)
    }
    
    obs_data = pd.DataFrame({
        "CellID": list(coords.keys()),
        "X_centroid": [c[0] for c in coords.values()],
        "Y_centroid": [c[1] for c in coords.values()],
        "custom_x": [c[0] for c in coords.values()],
        "custom_y": [c[1] for c in coords.values()],
    })

    return ad.AnnData(obs=obs_data)


@pytest.fixture
def sample_geojson() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame representing a GeoJSON file."""
    # Create two polygons with classification
    polygon1 = Polygon([(0, 0), (0, 70), (70, 70), (70, 0)]) # ClassA
    polygon2 = Polygon([(50, 50), (50, 120), (120, 120), (120, 50)]) # ClassB, overlaps with A
    polygon3 = Polygon([(0, 80), (0, 120), (40, 120), (40, 80)]) # ClassC, no overlap
    polygon4 = Polygon([(200, 200), (200, 210), (210, 210), (210, 200)]) # ClassD, no cells inside
    return gpd.GeoDataFrame({
        "geometry": [polygon1, polygon2, polygon3, polygon4],
        "classification": ['{"name": "ClassA"}', '{"name": "ClassB"}', '{"name": "ClassC"}', '{"name": "ClassD"}']
    }, crs="EPSG:4326")


@pytest.fixture
def temp_geojson_file(tmp_path, sample_geojson) -> str:
    """Create a temporary GeoJSON file for testing."""
    filepath = tmp_path / "temp.geojson"
    sample_geojson.to_file(filepath, driver="GeoJSON")
    return str(filepath)


def test_filter_with_default_params(sample_adata, temp_geojson_file):
    """Test filtering with deterministic data and overlapping polygons."""
    adata_annotated = filter_by_annotation(sample_adata, temp_geojson_file)
    
    assert "ClassA" in adata_annotated.obs.columns
    assert "ClassB" in adata_annotated.obs.columns
    assert "ClassD" in adata_annotated.obs.columns # New assertion for ClassD
    assert "ClassC" in adata_annotated.obs.columns
    assert "ANY" in adata_annotated.obs.columns
    assert "annotation" in adata_annotated.obs.columns

    # Check that 'ANY' column is True for cells inside any polygon
    inside_any = adata_annotated.obs["ANY"]
    assert (inside_any == (adata_annotated.obs["ClassA"] | adata_annotated.obs["ClassB"] | adata_annotated.obs["ClassC"])).all()

    # Check the 'annotation' column for specific cells
    obs_df = adata_annotated.obs.set_index('CellID')
    assert obs_df.loc["in_A_only", "annotation"] == "ClassA"
    assert obs_df.loc["in_B_only", "annotation"] == "ClassB"
    assert obs_df.loc["in_C_only", "annotation"] == "ClassC"
    assert obs_df.loc["in_A_and_B_overlap", "annotation"] == "MIXED"
    assert obs_df.loc["unannotated", "annotation"] == "Unannotated"

    # Check that ClassD column exists and is all False
    assert (adata_annotated.obs["ClassD"] == False).all()

    # Check boolean columns for the MIXED cell
    assert obs_df.loc["in_A_and_B_overlap", "ClassA"]
    assert obs_df.loc["in_A_and_B_overlap", "ClassB"]
    assert not obs_df.loc["in_A_and_B_overlap", "ClassC"]

def test_filter_with_custom_params(sample_adata, temp_geojson_file):
    """Test filtering with custom cell_id_col and x_y parameters."""
    adata_annotated = filter_by_annotation(
        sample_adata, temp_geojson_file, cell_id_col="CellID", x_y=("custom_x", "custom_y"), any_label="annotation_group"
    )

    assert "ClassA" in adata_annotated.obs.columns
    assert "ClassB" in adata_annotated.obs.columns
    assert "ClassC" in adata_annotated.obs.columns
    assert "annotation_group" in adata_annotated.obs.columns
    assert "annotation" in adata_annotated.obs.columns


def test_error_missing_x_column(sample_adata, temp_geojson_file):
    """Test ValueError when X coordinate column is missing."""
    with pytest.raises(ValueError, match=r"missing_x"):
        filter_by_annotation(sample_adata, temp_geojson_file, x_y=("missing_x", "Y_centroid"))


def test_error_missing_y_column(sample_adata, temp_geojson_file):
    """Test ValueError when Y coordinate column is missing."""
    with pytest.raises(ValueError, match=r"missing_y"):
        filter_by_annotation(sample_adata, temp_geojson_file, x_y=("X_centroid", "missing_y"))


def test_error_missing_cell_id_column(sample_adata, temp_geojson_file):
    """Test ValueError when cell ID column is missing."""
    with pytest.raises(ValueError, match=r"missing_id"):
        filter_by_annotation(sample_adata, temp_geojson_file, cell_id_col="missing_id")


def test_error_geojson_not_polygons(sample_adata, tmp_path):
    """Test ValueError when GeoJSON contains non-polygon geometries."""
    # Create a GeoDataFrame with Point geometry (invalid)
    points_gdf = gpd.GeoDataFrame({
        "geometry": gpd.points_from_xy([1, 2], [3, 4]),
        "classification": ['{"name": "ClassA"}', '{"name": "ClassB"}']
    }, crs="EPSG:4326")
    filepath = tmp_path / "points.geojson"
    points_gdf.to_file(filepath, driver="GeoJSON")

    with pytest.raises(ValueError, match="Only polygon geometries are supported"):
        filter_by_annotation(sample_adata, str(filepath))


def test_all_cells_unannotated(sample_adata, tmp_path):
    """Test scenario where all cells are outside any annotation."""
    # Create a GeoJSON far away from the cell centroids in sample_adata
    polygon = Polygon([(200, 200), (200, 250), (250, 250), (250, 200)])
    far_geojson = gpd.GeoDataFrame({
        "geometry": [polygon],
        "classification": ['{"name": "FarAwayClass"}']
    }, crs="EPSG:4326")
    filepath = tmp_path / "far.geojson"
    far_geojson.to_file(filepath, driver="GeoJSON")

    adata_annotated = filter_by_annotation(sample_adata, str(filepath))

    assert (adata_annotated.obs["FarAwayClass"] == False).all()  # All False for the annotation class
    assert (adata_annotated.obs["annotation"] == "Unannotated").all()  # All "Unannotated"
    assert (adata_annotated.obs["ANY"] == False).all()


def test_some_cells_annotated(sample_adata, temp_geojson_file):
    """Test scenario where some cells are annotated and some are not."""
    # This test uses the deterministic sample_adata which has annotated and unannotated cells
    adata_annotated = filter_by_annotation(sample_adata, temp_geojson_file)

    # Check that some cells are annotated with "ClassA"
    assert adata_annotated.obs["ClassA"].sum() > 0
    # Check that some cells are annotated with "ClassB"
    assert adata_annotated.obs["ClassB"].sum() > 0
    # Check that some cells are annotated with "ClassC"
    assert adata_annotated.obs["ClassC"].sum() > 0
    # Check that some cells are "Unannotated"
    assert (adata_annotated.obs["annotation"] == "Unannotated").sum() > 0

    # Check 'ANY' column correctly reflects cells within any annotation
    assert (adata_annotated.obs["ANY"] == (
        adata_annotated.obs["ClassA"] | adata_annotated.obs["ClassB"] | adata_annotated.obs["ClassC"]
        )).all()


def test_new_obs_columns_present(sample_adata, temp_geojson_file):
    """Test that the expected new columns are added to adata.obs."""
    adata_filtered = filter_by_annotation(sample_adata, temp_geojson_file)
    
    # Get expected columns from the GeoJSON (annotation classes) and the function's defaults
    gdf = gpd.read_file(temp_geojson_file)
    expected_annotation_cols = [
        name for name in gdf['classification'].apply(lambda x: ast.literal_eval(x).get('name')).unique()
    ]
    expected_cols = expected_annotation_cols + ["ANY", "annotation"]
    
    # Check if all expected columns are present in adata.obs
    for col in expected_cols:
        assert col in adata_filtered.obs.columns, f"Expected column '{col}' not found in adata.obs"
import os
import tempfile

import geopandas as gpd
import numpy as np
import pytest
import tifffile
from shapely.geometry import MultiPolygon, Polygon

# Helper function for counting coordinates in geometries
def _count_coords(geometry):
    """Counts total coordinates in a Polygon or MultiPolygon."""
    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) + sum(len(interior.coords) for interior in geometry.interiors)
    elif isinstance(geometry, MultiPolygon):
        total_coords = 0
        for poly in geometry.geoms:
            total_coords += len(poly.exterior.coords) + sum(len(interior.coords) for interior in poly.interiors)
        return total_coords
    return 0

from opendvp.imaging.mask_to_polygons import mask_to_polygons


@pytest.fixture
def create_temp_mask_tif():
    """Fixture to create a temporary TIFF file for testing."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "test_mask.tif")
    yield file_path
    # Clean up after test
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest.fixture
def create_blobs_labels_tif():
    """Fixture to create a synthetic 'blobs_labels.tif' for testing."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "blobs_labels.tif")

    # Create a synthetic mask resembling blobs
    # 0: background
    # 1: blob 1 (single square)
    # 2: blob 2 (single square)
    # 3: blob 3 (two disconnected parts)
    mask_data = np.zeros((50, 50), dtype=np.uint16)

    # Blob 1
    mask_data[10:20, 10:20] = 1
    # Blob 2
    mask_data[30:40, 30:40] = 2
    # Blob 3 (two disconnected parts)
    mask_data[5:10, 40:45] = 3
    mask_data[40:45, 5:10] = 3

    tifffile.imwrite(file_path, mask_data)
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_basic_polygon_extraction(create_temp_mask_tif):
    """Test basic extraction of a single polygon."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((10, 10), dtype=np.uint16)
    mask_data[2:8, 2:8] = 1  # A square blob
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.iloc[0]['cellId'] == 1
    assert isinstance(gdf.iloc[0]['geometry'], Polygon)
    # Check approximate area (6x6 square = 36 units)
    assert gdf.iloc[0]['geometry'].area == pytest.approx(36, abs=1)  # Allow for slight differences due to contouring


def test_multiple_polygons_extraction(create_temp_mask_tif):
    """Test extraction of multiple distinct polygons."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((20, 20), dtype=np.uint16)
    mask_data[2:5, 2:5] = 1  # Small square
    mask_data[10:15, 10:15] = 2  # Another square
    mask_data[1:3, 18:19] = 3  # Tiny blob
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)

    assert len(gdf) == 3
    assert set(gdf['cellId'].tolist()) == {1, 2, 3}
    for _, row in gdf.iterrows():
        assert isinstance(row['geometry'], Polygon)


def test_multipolygon_extraction(create_temp_mask_tif):
    """Test extraction of a MultiPolygon from disconnected parts of the same label."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((20, 20), dtype=np.uint16)
    mask_data[2:5, 2:5] = 1  # Part 1 of label 1
    mask_data[10:13, 10:13] = 1  # Part 2 of label 1
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)

    assert len(gdf) == 1
    assert gdf.iloc[0]['cellId'] == 1
    assert isinstance(gdf.iloc[0]['geometry'], MultiPolygon)
    assert len(gdf.iloc[0]['geometry'].geoms) == 2
    # Check approximate total area (2 * 3x3 squares = 18 units)
    assert gdf.iloc[0]['geometry'].area == pytest.approx(18, abs=1)


def test_background_label_ignored(create_temp_mask_tif):
    """Test that background (label 0) is ignored."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((10, 10), dtype=np.uint16)
    mask_data[0:10, 0:10] = 0  # All background
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)
    assert len(gdf) == 0  # No polygons should be extracted


def test_simplify_geometry(create_temp_mask_tif):
    """Test that geometry simplification works."""
    mask_path = create_temp_mask_tif
    # Create a mask that would benefit from simplification (e.g., a jagged shape)
    mask_data = np.zeros((10, 10), dtype=np.uint16)
    mask_data[2:8, 2:8] = 1
    mask_data[3, 3] = 0  # Create a small hole/indentation
    tifffile.imwrite(mask_path, mask_data)

    gdf_original = mask_to_polygons(mask_path, simplify=None)
    gdf_simplified = mask_to_polygons(mask_path, simplify=0.2)

    assert len(gdf_original) == 1
    assert len(gdf_simplified) == 1
    # Simplified geometry should have fewer or equal points
    assert _count_coords(gdf_simplified.iloc[0]['geometry']) <= _count_coords(gdf_original.iloc[0]['geometry'])
    # Area should be approximately the same
    assert gdf_simplified.iloc[0]['geometry'].area == pytest.approx(gdf_original.iloc[0]['geometry'].area, rel=0.1)
    

def test_max_memory_mb_raises_error(create_temp_mask_tif):
    """Test that ValueError is raised for exceeding max_memory_mb."""
    mask_path = create_temp_mask_tif
    # Create a large dummy mask (e.g., 1000x1000 uint16 = 2MB)
    large_mask_data = np.zeros((1000, 1000), dtype=np.uint16)
    tifffile.imwrite(mask_path, large_mask_data)

    # Set a very low max_memory_mb to trigger the error
    with pytest.raises(ValueError, match="Estimated mask size is .* MB, exceeding .* MB."):
        mask_to_polygons(mask_path, max_memory_mb=1)  # 1MB is less than 2MB


def test_blobs_labels_tif_scenario(create_blobs_labels_tif):
    """Test with the synthetic blobs_labels.tif."""
    mask_path = create_blobs_labels_tif
    gdf = mask_to_polygons(mask_path)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3  # Should have labels 1, 2, 3
    assert set(gdf['cellId'].tolist()) == {1, 2, 3}

    # Check label 1 (Polygon)
    cell1_geom = gdf[gdf['cellId'] == 1]['geometry'].iloc[0]
    assert isinstance(cell1_geom, Polygon)
    assert cell1_geom.area == pytest.approx(100, abs=1)  # 10x10 square

    # Check label 2 (Polygon)
    cell2_geom = gdf[gdf['cellId'] == 2]['geometry'].iloc[0]
    assert isinstance(cell2_geom, Polygon)
    assert cell2_geom.area == pytest.approx(100, abs=1)  # 10x10 square

    # Check label 3 (MultiPolygon)
    cell3_geom = gdf[gdf['cellId'] == 3]['geometry'].iloc[0]
    assert isinstance(cell3_geom, MultiPolygon)
    assert len(cell3_geom.geoms) == 2
    assert cell3_geom.area == pytest.approx(2 * 5 * 5, abs=1)  # Two 5x5 squares = 50


def test_mask_with_single_pixel_label(create_temp_mask_tif):
    """Test a mask where a label is just a single pixel."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((10, 10), dtype=np.uint16)
    mask_data[5, 5] = 1  # Single pixel label
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)
    assert len(gdf) == 1
    assert gdf.iloc[0]['cellId'] == 1
    assert isinstance(gdf.iloc[0]['geometry'], Polygon)
    # A single pixel contour might result in a very small polygon, or even a point/line
    # depending on skimage version and exact pixel value.
    # We expect a polygon with area close to 1.
    assert gdf.iloc[0]['geometry'].area == pytest.approx(1, abs=0.5)


def test_mask_with_no_labels(create_temp_mask_tif):
    """Test a mask with no labels (all zeros)."""
    mask_path = create_temp_mask_tif
    mask_data = np.zeros((10, 10), dtype=np.uint16)
    tifffile.imwrite(mask_path, mask_data)

    gdf = mask_to_polygons(mask_path)
    assert len(gdf) == 0
    assert gdf.empty
    assert list(gdf.columns) == ['cellId', 'geometry'] # Ensure columns are still defined
    assert gdf.crs == "EPSG:4326" # Ensure CRS is set even for empty GeoDataFrame

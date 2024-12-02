import dask.array as da
import numpy as np
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
from rasterio.features import shapes
import geopandas as gpd

def create_geodataframe_with_multipolygons(array):
    """
    Converts a labeled segmentation mask into a GeoDataFrame with polygons or multipolygons for each cell.
    Args:
        array: A 2D labeled segmentation mask, where pixel values represent cell IDs and background is 0.
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing polygons/multipolygons and their cell IDs.
    Caveats:
        Larger than memory arrays will fail
    """
    
    # Dictionary to store geometries grouped by cell ID
    cell_geometries = {}
    
    # Extract shapes and corresponding values
    for shape_dict, cell_id in shapes(array, mask=(array > 0)):
        polygon = shape(shape_dict)  # Convert to Shapely geometry
        cell_id = int(cell_id)
        if cell_id not in cell_geometries:
            cell_geometries[cell_id] = []
        cell_geometries[cell_id].append(polygon)
    
    # Combine multiple polygons into MultiPolygons if needed
    cell_ids = []
    geometries = []
    for cell_id, polygons in cell_geometries.items():
        if len(polygons) == 1:
            geometries.append(polygons[0])  # Single Polygon
        else:
            geometries.append(MultiPolygon(polygons))  # Combine into MultiPolygon
        cell_ids.append(cell_id)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'cellId': cell_ids, 'geometry': geometries}, crs="EPSG:4326")
    
    return gdf
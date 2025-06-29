import time

import geopandas as gpd
import numpy as np
import tifffile
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

from opendvp.utils import logger


def mask_to_polygons(
    mask_path: str,
    simplify: float | None = None,
    max_memory_mb: int = 16000,
) -> gpd.GeoDataFrame:
    """Convert a labeled segmentation mask (TIFF file) into a GeoDataFrame of polygons and/or multipolygons.

    Parameters:
    ----------
    mask_path : str
        Path to a 2D labeled segmentation mask TIFF. Pixel values represent cell IDs; background is 0.
    simplify : float, optional
        Tolerance for geometry simplification. If None, no simplification is performed.
    max_memory_mb : int, optional
        Maximum memory (in MB) allowed to safely process the image (default: 16000).

    Returns:
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing polygons/multipolygons and their cell IDs.

    Raises:
    ------
    ValueError
        If the estimated memory usage exceeds max_memory_mb or cell IDs exceed int32 range.
    """
    with tifffile.TiffFile(mask_path) as tif:
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        estimated_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        estimated_mb = estimated_bytes / (1024 ** 2)
        logger.info(f"  Mask shape: {shape}, dtype: {dtype}, estimated_mb: {estimated_mb:.1f}")

    if estimated_mb > max_memory_mb:
        raise ValueError(f"Estimated mask size is {estimated_mb:.2f} MB, exceeding {max_memory_mb:.1f} MB.")

    # Load the image data and ensure it's a 2D array
    array = tifffile.imread(mask_path)
    array = np.squeeze(array)

    # Dictionary to store geometries grouped by cell ID
    cell_geometries = {}
    start_time_contours = time.time()

    # Iterate through unique label values (excluding background 0)
    for label_value in np.unique(array[array > 0]):
        binary_mask = (array == label_value).astype(np.uint8) #dtype depends
        contours = measure.find_contours(binary_mask, 0.5)

        polygons_for_label = []
        for contour in contours:
            # Convert contour points to a Shapely polygon
            polygon = Polygon(contour[:, ::-1])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            polygons_for_label.append(polygon)

        if polygons_for_label:
            cell_geometries[int(label_value)] = polygons_for_label

    logger.info(f"Extracted contours in {time.time() - start_time_contours:.2f} seconds")

    # Combine multiple polygons into MultiPolygons if needed and create records
    records = []
    for cell_id, polygons in cell_geometries.items():
        geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
        records.append({'cellId': cell_id, 'geometry': geometry})

    if not records:
        # If no polygons were found, return an empty GeoDataFrame with the correct schema and CRS
        gdf = gpd.GeoDataFrame(columns=['cellId', 'geometry'], crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    if simplify is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify}")
        gdf['geometry'] = gdf['geometry'].simplify(simplify, preserve_topology=True)

    # Ensure 'cellId' is integer type
    gdf['cellId'] = gdf['cellId'].astype(int)

    logger.success(" -- Created geodataframe from segmentation mask -- ")

    return gdf
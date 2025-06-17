import time

import geopandas as gpd
import numpy as np
import tifffile
# from rasterio.features import shapes
from shapely.geometry import MultiPolygon
from shapely.geometry import shape as shapely_shape

from opendvp.utils import logger


# TODO Replace rasterio.features.shapes with:
# skimage.measure.find_contours OR
# shapely + scikit-image combo OR
# scipy.ndimage.label + regionprops (via skimage.measure)



def mask_to_polygons(
    mask_path: str,
    savepath: str | None = None,
    simplify: float | None = None,
    max_memory_mb: int = 16000,
) -> gpd.GeoDataFrame:
    """Convert a labeled segmentation mask (TIFF file) into a GeoDataFrame of polygons or multipolygons.

    Parameters
    ----------
    mask_path : str
        Path to a 2D labeled segmentation mask TIFF. Pixel values represent cell IDs; background is 0.
    savepath : str, optional
        Path to save the resulting GeoJSON file. If None, does not save to disk.
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
    logger.info(f" -- Converting {mask_path} to geodataframe of polygons -- ")

    # Load image metadata to check shape and memory usage
    with tifffile.TiffFile(mask_path) as tif:
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        estimated_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        estimated_mb = estimated_bytes / (1024 ** 2)
        logger.info(f"  Mask shape: {shape}, dtype: {dtype}, estimated_mb: {estimated_mb:.1f}")

    if estimated_mb > max_memory_mb:
        raise ValueError(
            f"Estimated memory usage is {estimated_mb:.2f} MB, exceeding the threshold of {max_memory_mb:.1f} MB."
        )

    # Load the image data
    array = tifffile.imread(mask_path)

    # Convert to int32
    start_time = time.time()
    max_label = array.max()
    logger.debug(f"Calculated max pixel value in {time.time() - start_time:.1f} seconds")
    if max_label <= np.iinfo(np.int32).max:
        array = array.astype(np.int32)
    else:
        raise ValueError(
            "Cell IDs exceed int32 range, and rasterio doesn't support uint32 or int64."
        )

    # Ensure 2D mask
    array = np.squeeze(array)

    # Dictionary to store geometries grouped by cell ID
    cell_geometries = {}

    # Extract shapes and corresponding values
    start_time = time.time()
    for shape_dict, cell_id in shapes(array, mask=(array > 0)):
        polygon = shapely_shape(shape_dict)
        cell_id = int(cell_id)
        cell_geometries.setdefault(cell_id, []).append(polygon)
    logger.info(f"Transformed pixel mask into polygons in {time.time() - start_time:.1f} seconds")

    # Combine multiple polygons into MultiPolygons if needed
    records = []
    for cell_id, polygons in cell_geometries.items():
        geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
        records.append({'cellId': cell_id, 'geometry': geometry})

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    if simplify is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify}")
        gdf['geometry'] = gdf['geometry'].simplify(simplify, preserve_topology=True)

    if savepath is not None:
        logger.info(f"Writing geodataframe as GeoJSON here {savepath}")
        start_time = time.time()
        gdf.to_file(savepath, driver="GeoJSON")
        logger.info(f"Writing of file took {time.time() - start_time:.1f} seconds")

    logger.success(" -- Created geodataframe from segmentation mask -- ")

    return gdf
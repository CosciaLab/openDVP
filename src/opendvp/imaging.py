from loguru import logger
import time

import numpy as np
import dask.array as da
import geopandas as gpd

import tifffile



#TODO test functions

def lazy_image_check(image_path):
    """ Check the image metadata without loading the image """
    logger.info(" ---- lazy_image_check : version number 1.0.0 ----")
    time_start = time.time()

    with tifffile.TiffFile(image_path) as image:
        # Getting the metadata
        shape = image.series[0].shape
        dtype = image.pages[0].dtype

        n_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        estimated_size_bytes = n_elements * bytes_per_element
        estimated_size_gb = estimated_size_bytes / 1024 / 1024 / 1024 
        
        logger.info(f"Image shape is {shape}")
        logger.info(f"Image data type: {dtype}")
        logger.info(f"Estimated size: {estimated_size_gb:.4g} GB")

    logger.info(f" ---- lazy_image_check is done, took {int(time.time() - time_start)}s  ----")


def mask_to_polygons(mask_path, savepath=None, simplify=None, max_memory_mb=16000):
    """
    Converts a labeled segmentation mask (TIFF file) into a GeoDataFrame with polygons or multipolygons.
    
    Args:
        mask_path (str): Path to a 2D labeled segmentation mask TIFF. Pixel values represent cell IDs; background is 0.
        max_memory_mb (int): Maximum memory (in MB) allowed to safely process the image.
    
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing polygons/multipolygons and their cell IDs.
    
    Raises:
        ValueError: If the estimated memory usage exceeds the max_memory_mb.
    """

    logger.info(f" -- Convering {mask_path} to geodataframe of polygons -- ")

    # Load image metadata to check shape and memory usage
    with tifffile.TiffFile(mask_path) as tif:
        shape = tif.series[0].shape
        dtype = tif.series[0].dtype
        estimated_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        estimated_mb = estimated_bytes / (1024 ** 2)
        logger.info(f"  Mask shape: {shape}, dtype: {dtype}, estimated_mb: {estimated_mb:.1f}")

    if estimated_mb > max_memory_mb:
        raise ValueError(f"Estimated memory usage is {estimated_mb:.2f} MB, exceeding the threshold of {max_memory_mb:.1f} MB.")

    # Load the image data
    
    array = tifffile.imread(mask_path)

    # convert to int32
    start_time = time.time()
    max_label = array.max()
    logger.debug(f"Calculated max pixel value in {time.time() - start_time:.1f} seconds")
    if max_label <= np.iinfo(np.int32).max:
        array = array.astype(np.int32)
    else:
        raise ValueError(f"Cell IDs exceed int32 range, and rasterio doesn't support uint32 or int64.")

    #Ensure 2D mask
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
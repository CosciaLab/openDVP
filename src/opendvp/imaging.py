from loguru import logger
import time

import numpy as np
import dask.array as da
import geopandas as gpd
import pandas.api.types as ptypes

import tifffile
import shapely
import rasterio
from rasterio.features import shapes
import scipy

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


def adataobs_to_voronoi_geojson(
        df,
        imageid, 
        subset:list=None, 
        category_1:str="phenotype", 
        category_2=None, 
        output_path:str="../data/processed/"):
    """ 
    Description:
    
    """

    logger.debug(f" df shape: {df.shape}")

    df = df.copy()
    #subset per image
    if ptypes.is_numeric_dtype(df['imageid']) is False:
        logger.info("ImageID is not a numeric, changing datatype to int16")
        df['imageid'] = df['imageid'].astype("int16")
    df = df[(df.imageid == imageid)]
    logger.debug(f" df shape after imageid subset: {df.shape}")
    logger.info(f"Processing {imageid}, loaded dataframe")

    #subset per x,y
    if subset is not None:
        logger.info(f"Subsetting to {subset}")
        assert len(subset) == 4, "subset must be a list of 4 integers"
        x_min, x_max, y_min, y_max = subset
        df = df[(df.X_centroid > x_min) &
                (df.X_centroid < x_max) &
                (df.Y_centroid > y_min) &
                (df.Y_centroid < y_max)]
        #clean subset up
        df = df.reset_index(drop=True)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    logger.info("Running Voronoi")
    # run Voronoi 
    # df = df[['X_centroid', 'Y_centroid', category_1, category_2]]    
    vor = scipy.spatial.Voronoi(df[['X_centroid', 'Y_centroid']].values)
    polygons = []
    for i in range(len(df)):
        polygon = shapely.Polygon(
            [vor.vertices[vertex] for vertex in vor.regions[vor.point_region[i]]])
        polygons.append(polygon)
    df['geometry'] = polygons
    logger.info("Voronoi done")

    #transform to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    logger.info("Transformed to geodataframe")

    # filter polygons that go outside of image
    if subset is None:
        x_min = gdf['X_centroid'].min()
        x_max = gdf['X_centroid'].max()
        y_min = gdf['Y_centroid'].min()
        y_max = gdf['Y_centroid'].max()
        logger.info(f"Bounding box: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max {y_max}")

    boundary_box = shapely.box(x_min, y_min, x_max, y_max)
    gdf = gdf[gdf.geometry.apply(lambda poly: poly.within(boundary_box))]
    logger.info("Filtered out infinite polygons")

    # filter polygons that are too large
    gdf['area'] = gdf['geometry'].area
    gdf = gdf[gdf['area'] < gdf['area'].quantile(0.98)]
    logger.info("Filtered out large polygons based on the 98% quantile")
    # filter polygons that are very pointy
    
    # TODO improve filtering by pointiness
    # gdf = process_polygons(gdf, scale_threshold=350, remove_threshold=400, scale_factor=0.3)
    # logger.info("Filtered out pointy polygons")

    # create geodataframe for each cell and their celltype
    gdf2 = gdf.copy()
    # gdf2['objectType'] = 'cell'
    gdf2['objectType'] = "detection"
    gdf2['classification'] = gdf2[category_1]
    
    # merge polygons based on the CN column
    if category_2:
        logger.info("Merging polygons for cellular neighborhoods")
        gdf3 = gdf.copy()
        gdf3 = gdf3.dissolve(by=category_2)
        gdf3[category_2] = gdf3.index
        gdf3 = gdf3.explode(index_parts=True)
        gdf3 = gdf3.reset_index(drop=True)
        gdf3['classification'] = gdf3[category_2].astype(str)
        
    #export to geojson
    datetime = time.strftime("%Y%m%d_%H%M")
    gdf2.to_file(f"{output_path}/{datetime}_{imageid}_cells_voronoi.geojson", driver='GeoJSON')
    if category_2:
        gdf3.to_file(f"{output_path}/{datetime}_{imageid}_RCN_voronoi.geojson", driver='GeoJSON')

    logger.success(f"Exported {imageid} to geojson")


def mask_to_polygons(array):
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
        polygon = shapely.geometry.shape(shape_dict)  # Convert to Shapely geometry
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
            geometries.append(shapely.geometry.MultiPolygon(polygons))  # Combine into MultiPolygon
        cell_ids.append(cell_id)
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'cellId': cell_ids, 'geometry': geometries}, crs="EPSG:4326")
    
    return gdf
import pandas as pd
import anndata as ad
import geopandas as gpd

from loguru import logger
import time, os, re
from itertools import cycle

import scipy
import shapely
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape, MultiPolygon
import matplotlib.colors as mcolors
import pandas.api.types as ptypes
from opendvp.utils import parse_color_for_qupath

def read_quant(csv_data_path) -> ad.AnnData:
    """
    Read the quantification data from a csv file and return an anndata object
    :param csv_data_path: path to the csv file
    :return: an anndata object
    """

    #TODO not general enough, exemplar001 fails

    logger.info(" ---- read_quant : version number 1.1.0 ----")
    time_start = time.time()

    assert csv_data_path.endswith('.csv'), "The file should be a csv file"
    df = pd.read_csv(csv_data_path)
    df.index = df.index.astype(str)

    meta_columns = ['CellID', 'Y_centroid', 'X_centroid',
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'Orientation', 'Extent', 'Solidity']
    assert all([column in df.columns for column in meta_columns]), "The metadata columns are not present in the csv file"

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)
    variables = pd.DataFrame(
        index=data.columns,
        data={"math": [column_name.split("_")[0] for column_name in data.columns],
            "marker": ["_".join(column_name.split("_")[1:]) for column_name in data.columns]})

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    logger.info(f" ---- read_quant is done, took {int(time.time() - time_start)}s  ----")
    return adata

def switch_adat_var_index(adata, new_index):
    """
    Created by Jose Nimo on 2023-07-01
    Lastest modified by Jose Nimo on 2024-11-16

    Description:
    Switch the index of adata.var to a new index. Useful for switching between gene names and protein names.

    Arg:
        adata: anndata object
        new_index: pandas series, new index to switch to
    Returns:
        adata: anndata object, with the new index
    """
    adata_copy = adata.copy()

    adata_copy.var[adata_copy.var.index.name] = adata_copy.var.index
    adata_copy.var.set_index(new_index, inplace=True)
    adata_copy.var.index.name = new_index
    
    return adata_copy

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def save_adata_checkpoint(adata, path_to_dir, checkpoint_name):
    try:    
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_dir,checkpoint_name), exist_ok=True)
        basename = f"{os.path.join(path_to_dir,checkpoint_name)}/{get_datetime()}_{checkpoint_name}_adata"
        
        # Save h5ad file
        try:
            logger.info("Writing h5ad")
            adata.write_h5ad(filename = basename + ".h5ad")
            logger.success("Wrote h5ad file")
        except (OSError, IOError, ValueError) as e:
                logger.error(f"Could not write h5ad file: {e}")
                return
        
        # Save CSV file
        try:
            logger.info("Writing parquet")
            adata.to_df().to_parquet(path=basename + ".parquet")
            logger.success("Wrote parquet file")
        except (OSError, IOError, ValueError) as e:
            logger.error(f"Could not write parquet file: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in save_adata_checkpoint: {e}")

def color_geojson_w_adata(
        geodataframe,
        geodataframe_index_key,
        adata,
        adata_obs_index_key,
        adata_obs_category_key,
        color_dict,
        export_path,
        simplify_value=1,
        return_gdf=False
):
    
    """
    Add classification colors from an AnnData object to a GeoDataFrame for QuPath visualization.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        GeoDataFrame containing polygons to annotate.
    
    geodataframe_index_key : str
        Column in the GeoDataFrame that corresponds to the index or column in adata.obs used for matching.

    adata : anndata.AnnData
        AnnData object containing cell annotations in `adata.obs`.

    adata_obs_index_key : str
        Column name in `adata.obs` used to match to `geodataframe_index_key`.

    adata_obs_category_key : str
        Column in `adata.obs` that defines the classification/grouping to color.

    color_dict : dict, optional
        Dictionary mapping class names to RGB color lists (e.g., {'Tcell': [255, 0, 0]}).
        If None, a default color cycle will be used.

    export_path : str, optional
        Path where the output GeoJSON will be saved.

    simplify_value : float, optional
        Tolerance value for geometry simplification (higher = more simplified).
        default = 1

    return_gdf : bool, optional
        If True, returns the modified GeoDataFrame with classifications.

    Returns
    -------
    geopandas.GeoDataFrame or None
        Returns the updated GeoDataFrame if `return_gdf=True`, else writes to file only.
    """
    
    logger.info(" -- Adding color to polygons for QuPath visualization -- ")
    
    gdf = geodataframe.copy()
    gdf['objectType'] = "detection"
    
    phenotypes_series = adata.obs.set_index(adata_obs_index_key)[adata_obs_category_key]

    if gdf[geodataframe_index_key].dtype != phenotypes_series.index.dtype:
        gdf_dtype = gdf[geodataframe_index_key].dtype
        adata_dtype = phenotypes_series.index.dtype
        logger.warning(f"Data types between geodaframe {gdf_dtype} and adataobs col {adata_dtype} do not match")

    if geodataframe_index_key:
        logger.info(f"Matching gdf[{geodataframe_index_key}] to adata.obs[{adata_obs_index_key}]")
        gdf['class'] = gdf[geodataframe_index_key].map(phenotypes_series)
    else:
        logger.info("geodataframe index key not passed, using index")
        gdf.index = gdf.index.astype(str)
        gdf['class'] = gdf.index.map(phenotypes_series).astype(str)

    gdf['class'] = gdf['class'].astype("category")
    gdf['class'] = gdf['class'].cat.add_categories('filtered_out') 
    gdf['class'] = gdf['class'].fillna('filtered_out')
    gdf['class'] = gdf['class'].replace("nan", "filtered_out")


    color_dict = parse_color_for_qupath(color_dict, adata=adata, adata_obs_key=adata_obs_category_key)

    if 'filtered_out' not in color_dict:
        color_dict['filtered_out'] = [0,0,0]

    gdf['classification'] = gdf.apply(lambda x: {'name': x['class'], 'color': color_dict[x['class']]}, axis=1)
    gdf.drop(columns='class', inplace=True)

    #simplify the geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        start_time = time.time()
        gdf['geometry'] = gdf['geometry'].simplify(simplify_value, preserve_topology=True)
        logger.info(f"Simplified all polygons in {time.time() - start_time:.1f} seconds")

    logger.info("Writing polygons as geojson file")
    start_time = time.time()
    gdf.to_file(export_path, driver='GeoJSON')
    logger.info(f"File written in {time.time() - start_time:.1f} seconds")

    if return_gdf:
        return gdf

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

    #TODO one category at a time
    #TODO decide between annotations and detections
    #TODO pass adata as input not df
    #TODO pass color dict
    #TODO check colors
    #TODO docstring
    #TODO as detections for opaque 

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
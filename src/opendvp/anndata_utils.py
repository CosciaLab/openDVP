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

    #export to geojson
    if export_path:
        logger.info("Writing polygons as geojson file")
        gdf.index.name = 'index'
        gdf.to_file(export_path, driver='GeoJSON')
        logger.success(f"Exported Voronoi projection to {export_path}")
    else:
        logger.success(f" -- Created and returning Voronoi projection -- ")
        return gdf

def adataobs_to_voronoi_geojson(
        adata,
        subset_adata_key = None, 
        subset_adata_value = None,
        color_by_adata_key:str = "phenotype",
        color_dict:dict = None,
        threshold_quantile = 0.98,
        merge_adjacent_shapes = True,
        save_as_detection = True,  
        output_filepath:str = None
        ):
    """
    Generate a Voronoi diagram from cell centroids stored in an AnnData object 
    and export it as a GeoJSON file or return it as a GeoDataFrame.

    This function computes a 2D Voronoi tessellation from the 'X_centroid' and 'Y_centroid' 
    columns in `adata.obs`, optionally filters and merges polygons based on user-defined criteria, 
    and outputs the result in a GeoJSON-compatible format for visualization or downstream analysis 
    (e.g., in QuPath).

    Parameters
    ----------
    adata : AnnData
        AnnData object with centroid coordinates in `adata.obs[['X_centroid', 'Y_centroid']]`.

    subset_adata_key : str, optional
        Column in `adata.obs` used to filter a subset of cells (e.g., a specific image ID or tissue section).

    subset_adata_value : Any, optional
        Value used to subset the `subset_adata_key` column. Only rows matching this value will be used.

    color_by_adata_key : str, default "phenotype"
        Column in `adata.obs` that determines the class or type of each cell. Used for coloring and grouping.

    color_dict : dict, optional
        Dictionary mapping class names to RGB color codes. Used to color each class in QuPath style.
        If not provided, a default palette will be generated.

    threshold_quantile : float, default 0.98
        Polygons with an area greater than this quantile will be excluded (used to remove oversized/outlier regions).

    merge_adjacent_shapes : bool, default True
        If True, merges adjacent polygons with the same class label.

    save_as_detection : bool, default True
        If True, sets the `objectType` property to "detection" in the output for QuPath compatibility.

    output_filepath : str, optional
        If provided, saves the output as a GeoJSON file at this path. 
        If None, returns the GeoDataFrame instead.

    Returns
    -------
    geopandas.GeoDataFrame or None
        If `output_filepath` is None, returns the resulting GeoDataFrame with Voronoi polygons.
        Otherwise, writes to file and returns None.

    Notes
    -----
    - Requires the `scipy`, `shapely`, `geopandas`, and `anndata` packages.
    - Assumes `adata.obs` contains valid `X_centroid` and `Y_centroid` columns.
    """

    #TODO threshold_quantile to area_threshold_quantile

    if 'X_centroid' not in adata.obs or 'Y_centroid' not in adata.obs:
        raise ValueError("`adata.obs` must contain 'X_centroid' and 'Y_centroid' columns.")

    df = adata.obs.copy()

    if subset_adata_key and subset_adata_value is not None:
        if subset_adata_key not in adata.obs.columns:
            raise ValueError(f"{subset_adata_key} not found in adata.obs columns.")
        if subset_adata_value not in adata.obs[subset_adata_key].unique():
            raise ValueError(f"{subset_adata_value} not found in adata.obs[{subset_adata_key}].")

        logger.info(adata.obs[subset_adata_key].unique())
        logger.info(f"Subset adata col dtype: {adata.obs[subset_adata_key].dtype}")
        df = df[df[subset_adata_key] == subset_adata_value]
        logger.info(f" Shape after subset: {df.shape}")

    # Run Voronoi
    logger.info("Running Voronoi")
    vor = scipy.spatial.Voronoi(df[['X_centroid', 'Y_centroid']].values)
    df['geometry'] = [safe_voronoi_polygon(vor, i) for i in range(len(df))]
    logger.info("Voronoi done")

    #transform to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    logger.info("Transformed to geodataframe")

    # filter polygons that go outside of image
    x_min, x_max = gdf['X_centroid'].min(), gdf['X_centroid'].max()
    y_min, y_max = gdf['Y_centroid'].min(), gdf['Y_centroid'].max()
    logger.info(f"Bounding box: x_min: {x_min:.1f}, x_max: {x_max:.1f}, y_min: {y_min:.1f}, y_max {y_max:.1f}")
    boundary_box = shapely.box(x_min, y_min, x_max, y_max)
    # gdf = gdf[gdf.geometry.apply(lambda poly: poly.within(boundary_box))]
    gdf = gdf[gdf.geometry.within(boundary_box)]
    # logger.info("Filtered out infinite polygons")
    logger.info(f"Retaining {len(gdf)} valid polygons after filtering large and infinite ones.")

    # filter polygons that are too large
    gdf['area'] = gdf['geometry'].area
    gdf = gdf[gdf['area'] < gdf['area'].quantile(threshold_quantile)]
    logger.info(f"Filtered out large polygons based on the {threshold_quantile} quantile")

    # create geodataframe for each cell and their celltype
    if save_as_detection:
        gdf['objectType'] = "detection"

    # merge polygons based on the CN column
    if merge_adjacent_shapes:
        logger.info("Merging polygons adjacent and of same category")
        gdf = gdf.dissolve(by=color_by_adata_key)
        gdf[color_by_adata_key] = gdf.index
        gdf = gdf.explode(index_parts=True)
        gdf = gdf.reset_index(drop=True)
        
    #add color
    gdf['classification'] = gdf[color_by_adata_key].astype(str)
    color_dict = parse_color_for_qupath(color_dict, adata=adata, adata_obs_key=color_by_adata_key)
    gdf['classification'] = gdf.apply(lambda x: {'name': x['classification'], 'color': color_dict[x['classification']]}, axis=1)

    #export to geojson
    if output_filepath:
        gdf.to_file(output_filepath, driver='GeoJSON')
        logger.success(f"Exported Voronoi projection to {output_filepath}")
    else:
        logger.success(f" -- Created and returning Voronoi projection -- ")
        return gdf
    
def safe_voronoi_polygon(vor, i):
    region_index = vor.point_region[i]
    region = vor.regions[region_index]
    # Invalid if empty or contains -1 (infinite vertex)
    if -1 in region or len(region) < 3:
        return None
    vertices = vor.vertices[region]
    if len(vertices) < 3:
        return None
    polygon = shapely.Polygon(vertices)
    # Validate: must have 4+ coords to form closed polygon
    if not polygon.is_valid or len(polygon.exterior.coords) < 4:
        return None
    return polygon
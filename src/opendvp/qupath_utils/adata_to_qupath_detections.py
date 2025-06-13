import time

from opendvp.logger import logger


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
    """Add classification colors from an AnnData object to a GeoDataFrame for QuPath visualization.

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

    Returns:
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
        logger.success(" -- Created and returning Voronoi projection -- ")
        return gdf
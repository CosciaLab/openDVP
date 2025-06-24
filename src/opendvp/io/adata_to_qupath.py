from typing import Literal

import anndata as ad
import geopandas as gpd
import scipy
import shapely

from opendvp.utils import logger, parse_color_for_qupath

#TODO Separate into adata_to_Voronoi and color_geojson_w_adataobs

def adata_to_qupath(
    adata: ad.AnnData,
    mode: Literal["voronoi", "polygons"] = "polygons",
    geodataframe: gpd.GeoDataFrame | None = None,
    geodataframe_index_key: str | None = None,
    adata_obs_index_key: str | None = None,
    adata_obs_category_key: str | None = None,
    color_dict: dict | None = None,
    export_path: str | None = None,
    simplify_value: float | int = 1,
    voronoi_area_quantile: float | None = 0.98,
    merge_adjacent_shapes: bool = False,
    subset_adata_key: str | None = None,
    subset_adata_value : str | None = None,
    save_as_detection: bool = True,
) -> gpd.GeoDataFrame | None:
    """Flexible function to output QuPath-compatible detections from AnnData, using either Voronoi or provided polygons.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell metadata and (optionally) centroid coordinates.
    mode : {"voronoi", "polygons"}
        If "voronoi", generates polygons from centroids. If "polygons", uses provided geodataframe.
    geodataframe : GeoDataFrame, optional
        If mode="polygons", the polygons to annotate.
    geodataframe_index_key : str, optional
        Column in geodataframe to match to adata.obs (for mode="polygons").
    adata_obs_index_key : str, optional
        Column in adata.obs to match to geodataframe (for mode="polygons").
    adata_obs_category_key : str, optional
        Column in adata.obs for class/color annotation.
        If no color is given resulting shapes will be white
    color_dict : dict, optional
        Mapping from class names to RGB color lists.
    export_path : str, optional
        If provided, writes output to this path as GeoJSON.
    simplify_value : float, default 1
        Geometry simplification tolerance (for polygons mode).
    voronoi_area_quantile : float, default 0.98
        Area quantile threshold for filtering large Voronoi polygons.
    merge_adjacent_shapes : bool, default False
        If True, merges adjacent polygons of the same class
        Selected with adata_obs_category_key (Voronoi mode).
    subset_adata_key : str, optional
        Column in adata.obs to filter for Voronoi mode.
    subset_adata_value : any, optional
        Value to filter subset_adata_key by (Voronoi mode).
    save_as_detection : bool, default True
        If True, sets objectType to "detection" for QuPath. Otherwise as Annotations.

    Returns:
    -------
    GeoDataFrame or None
        Returns GeoDataFrame if export_path is None, else writes to file and returns None.
    """
    ### VORONOI ###
    if mode == "voronoi":
        #check coords
        if 'X_centroid' not in adata.obs or 'Y_centroid' not in adata.obs:
            raise ValueError("`adata.obs` must contain 'X_centroid' and 'Y_centroid' columns for Voronoi mode.")
        obs_df = adata.obs.copy()
        #subset
        if subset_adata_key and subset_adata_value is not None:
            if subset_adata_key not in adata.obs.columns:
                raise ValueError(f"{subset_adata_key} not found in adata.obs columns.")
            if subset_adata_value not in adata.obs[subset_adata_key].unique():
                raise ValueError(f"{subset_adata_value} not found in adata.obs[{subset_adata_key}].")
            logger.info(adata.obs[subset_adata_key].unique())
            logger.info(f"Subset adata col dtype: {adata.obs[subset_adata_key].dtype}")
            obs_df = obs_df[obs_df[subset_adata_key] == subset_adata_value] #COULD BE A LIST
            logger.info(f" Shape after subset: {obs_df.shape}")
        
        logger.info("Running Voronoi")
        vor = scipy.spatial.Voronoi(obs_df[['X_centroid', 'Y_centroid']].values)
        def safe_voronoi_polygon(vor,  i : int) -> shapely.Polygon | None :
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            if -1 in region or len(region) < 3:
                return None
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                return None
            polygon = shapely.Polygon(vertices)
            if not polygon.is_valid or len(polygon.exterior.coords) < 4:
                return None
            return polygon
        
        obs_df['geometry'] = [safe_voronoi_polygon(vor, i) for i in range(len(obs_df))]
        logger.info("Voronoi done")
        gdf = gpd.GeoDataFrame(obs_df, geometry='geometry')
        logger.info("Transformed to geodataframe")

        # Filter polygons outside bounding box
        x_min, x_max = gdf['X_centroid'].min(), gdf['X_centroid'].max()
        y_min, y_max = gdf['Y_centroid'].min(), gdf['Y_centroid'].max()
        boundary_box = shapely.box(x_min, y_min, x_max, y_max)
        gdf = gdf[gdf.geometry.within(boundary_box)]
        logger.info(f"Retaining {len(gdf)} valid polygons after filtering large and infinite ones.")

        # Area filter
        if voronoi_area_quantile:
            gdf['area'] = gdf['geometry'].area
            gdf = gdf[gdf['area'] < gdf['area'].quantile(voronoi_area_quantile)]
            logger.info(f"Filtered out large polygons based on the {voronoi_area_quantile} quantile")
        if save_as_detection:
            gdf['objectType'] = "detection"
        if merge_adjacent_shapes:
            logger.info("Merging polygons adjacent and of same category")
            if adata_obs_category_key is None:
                raise ValueError("You need adata_obs_category_key if merge_adjacent_shapes is True")
            gdf = gdf.dissolve(by=adata_obs_category_key)
            gdf[adata_obs_category_key] = gdf.index
            gdf = gdf.explode(index_parts=True)
            gdf = gdf.reset_index(drop=True)
        if adata_obs_category_key:
            gdf['classification'] = gdf[adata_obs_category_key].astype(str)
    
    ### POLYGONS ###
    elif mode == "polygons":
        if geodataframe is None:
            raise ValueError("geodataframe must be provided for mode='polygons'")
        gdf = geodataframe.copy()
        gdf['objectType'] = "detection" if save_as_detection else None
        if adata_obs_category_key:
            phenotypes_series = adata.obs.set_index(adata_obs_index_key)[adata_obs_category_key]
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
            gdf['classification'] = gdf['class']
            gdf = gdf.drop(columns='class')
        if simplify_value is not None:
            logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
            gdf['geometry'] = gdf['geometry'].simplify(simplify_value, preserve_topology=True)
    else:
        raise ValueError("mode must be either 'voronoi' or 'polygons'")
    
    # Color annotation
    if color_dict and adata_obs_category_key:
        color_dict = parse_color_for_qupath(color_dict, adata=adata, adata_obs_key=adata_obs_category_key)
        if 'filtered_out' not in color_dict:
            color_dict['filtered_out'] = [0,0,0]
        gdf['classification'] = gdf.apply(
            lambda row: {
                'name': row['classification'],
                'color': color_dict.get(row['classification']), #Careful here
            },
        axis=1,
        )
    # Export or return
    if export_path:
        gdf.index.name = 'index'
        gdf.to_file(export_path, driver='GeoJSON')
        logger.success(f"Exported detections to {export_path}")
        return None
    else:
        logger.success(" -- Created and returning detections -- ")
        return gdf
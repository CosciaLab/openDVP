import geopandas as gpd
import scipy
import shapely

from opendvp.logger import logger


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
    """Generate a Voronoi diagram from cell centroids stored in an AnnData object
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

    Returns:
    -------
    geopandas.GeoDataFrame or None
        If `output_filepath` is None, returns the resulting GeoDataFrame with Voronoi polygons.
        Otherwise, writes to file and returns None.

    Notes:
    -----
    - Requires the `scipy`, `shapely`, `geopandas`, and `anndata` packages.
    - Assumes `adata.obs` contains valid `X_centroid` and `Y_centroid` columns.
    """

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
    gdf['classification'] = gdf.apply(
        lambda x: {
            'name': x['classification'],
            'color': color_dict[x['classification']],
        },
        axis=1,
    )

    #export to geojson
    if output_filepath:
        gdf.to_file(output_filepath, driver='GeoJSON')
        logger.success(f"Exported Voronoi projection to {output_filepath}")
    else:
        logger.success(" -- Created and returning Voronoi projection -- ")
        return gdf
import anndata as ad
import geopandas as gpd
import scipy
import shapely

from opendvp.utils import logger, parse_color_for_qupath


def adata_to_voronoi(
    adata: ad.AnnData,
    x_y: tuple = ("X_centroid", "Y_centroid"),
    classify_by: str | None = None,
    color_dict: dict | None = None,
    voronoi_area_quantile: float | None = 0.98,
    merge_adjacent_shapes: bool = False,
    save_as_detection: bool = True,
) -> gpd.GeoDataFrame | None:
    """Generate a GeoDataFrame of Voronoi polygons from AnnData centroids.
    
    This function computes Voronoi polygons from centroid coordinates in AnnData.obs
    Optionally annotates them with class labels and colors for QuPath, and returns a GeoDataFrame.

    Parameters:
    ----------
    adata: AnnData
        AnnData object with cell metadata and centroid coordinates.
    x_y: tuple, default ("X_centroid", "Y_centroid")
        Tuple of column names in adata.obs for X and Y coordinates.
    classify_by: str or None, optional
        Column in adata.obs to use for classifying detections (e.g., cell type).
    color_dict: dict or None, optional
        Dictionary mapping class/category names to RGB color lists.
    voronoi_area_quantile: float or None, default 0.98
        Area quantile threshold for filtering large Voronoi polygons.
    merge_adjacent_shapes: bool, default False
        If True, merges adjacent polygons of the same class.
    save_as_detection: bool, default True
        If True, sets 'objectType' to 'detection' for QuPath.

    Returns:
    -------
    GeoDataFrame:
        GeoDataFrame of Voronoi polygons with optional class/color annotation.

    Raises:
    -------
    ValueError:
        If required columns are missing or input types are incorrect.

    Examples:
    --------
    >>> import anndata as ad
    >>> import pandas as pd
    >>> import numpy as np
    >>> from opendvp.io.adata_to_voronoi import adata_to_voronoi
    >>> obs = pd.DataFrame({
    ...     'X_centroid': np.random.rand(5),
    ...     'Y_centroid': np.random.rand(5),
    ...     'celltype': ['A', 'B', 'A', 'B', 'A']
    ... })
    >>> adata = ad.AnnData(obs=obs)
    >>> gdf = adata_to_voronoi(adata, classify_by='celltype')
    >>> print(gdf.head())
    """
    if not isinstance(adata, ad.AnnData):
        raise ValueError("adata must be an instance of anndata.AnnData")
    if x_y[0] not in adata.obs.columns or x_y[1] not in adata.obs.columns:
        raise ValueError(f"{x_y[0]} or {x_y[1]} not found in adata.obs.columns")
    if classify_by:
        if classify_by not in adata.obs.columns:
            raise ValueError(f"{classify_by} not in adata.obs.columns")
        if adata.obs[classify_by].isna().any():
            raise ValueError(f"The {classify_by} contains NaN values")
        if adata.obs[classify_by].dtype.name != 'category':
            logger.warning(f"{classify_by} is not a categorical, converting to categorical")
            adata.obs[classify_by] = adata.obs[classify_by].astype('category')
    if color_dict and not isinstance(color_dict, dict):
        raise ValueError("provided color_dict is not a dict")

    obs_df = adata.obs.copy()

    logger.info("Running Voronoi")
    vor = scipy.spatial.Voronoi(obs_df[[x_y[0], x_y[1]]].values)
    logger.info("Voronoi done")

    def safe_voronoi_polygon(vor, i: int) -> shapely.Polygon | None:
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
    gdf = gpd.GeoDataFrame(obs_df, geometry='geometry')
    logger.info("Transformed to geodataframe")

    # Filter polygons outside bounding box
    x_min, x_max = gdf[x_y[0]].min(), gdf[x_y[0]].max()
    y_min, y_max = gdf[x_y[1]].min(), gdf[x_y[1]].max()
    boundary_box = shapely.box(x_min, y_min, x_max, y_max)
    gdf = gdf[gdf.geometry.within(boundary_box)]
    logger.info(f"Retaining {len(gdf)} valid polygons after filtering infinite ones.")

    # Area filter
    if voronoi_area_quantile:
        gdf['area'] = gdf['geometry'].area
        gdf = gdf[gdf['area'] < gdf['area'].quantile(voronoi_area_quantile)]
        logger.info(f"Filtered out large polygons larger than {voronoi_area_quantile} quantile")
    if save_as_detection:
        gdf['objectType'] = "detection"

    if classify_by:
        if merge_adjacent_shapes:
            logger.info("Merging polygons adjacent of the same category")
            gdf = gdf.dissolve(by=classify_by)
            gdf[classify_by] = gdf.index
            gdf = gdf.explode(index_parts=True)
            gdf = gdf.reset_index(drop=True)
        gdf['class'] = gdf[classify_by].astype(str)
        if color_dict:
            color_dict = parse_color_for_qupath(color_dict, adata=adata, adata_obs_key=classify_by)
            if 'filtered_out' not in color_dict:
                color_dict['filtered_out'] = [0, 0, 0]
            gdf['classification'] = gdf.apply(
                lambda row: {'name': row['class'], 'color': color_dict[row['class']]}, axis=1)

    return gdf
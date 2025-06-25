import anndata as ad
import geopandas

from opendvp.utils import logger, parse_color_for_qupath


def adata_to_qupath(  # noqa: C901
    adata: ad.AnnData,
    geodataframe: geopandas.GeoDataFrame,
    adataobs_on: str = "CellID",
    gdf_on: str | None = "CellID",
    gdf_index : bool = False,
    classify_by: str | None = None,
    color_dict: dict | None = None,
    simplify_value: float | None = 1.0,
    save_as_detection : bool = True,
) -> geopandas.GeoDataFrame | None:
    """Export a GeoDataFrame with QuPath-compatible annotations, using AnnData for classification and color mapping.

    This function matches shapes in a GeoDataFrame to metadata in an AnnData object.
    Assigns class labels and colors for QuPath visualization, and optionally simplifies geometries.

    Parameters:
    ----------
    adata: AnnData
        AnnData object containing cell metadata (e.g., cell types, phenotypes).
    geodataframe: GeoDataFrame
        GeoDataFrame containing shapes (detections or annotations) to be exported.
    adataobs_on: str, default "CellID"
        Column in adata.obs to match with geodataframe.
    gdf_on: str or None, default "CellID"
        Column in geodataframe to match with adata.obs. If None and gdf_index is True, uses the index.
    gdf_index: bool, default False
        If True, uses the geodataframe index for matching instead of a column.
    classify_by: str or None, optional
        Column in adata.obs to use for classifying detections (e.g., cell type or phenotype).
    color_dict: dict or None, optional
        Dictionary mapping class/category names to RGB color lists (e.g., {'Tcell': [255, 0, 0]}).
        If not provided, a default color cycle will be generated.
        Hexcodes colors are acceptable as well.
    simplify_value: float, default 1.0
        Tolerance for geometry simplification. Set to None to disable simplification.
    save_as_detection: bool, default True
        If True, sets 'objectType' to 'detection' for QuPath.

    Returns:
        GeoDataFrame:
            The resulting GeoDataFrame with QuPath-compatible columns.

    Raises:
        ValueError:
            If input types are incorrect, required columns are missing, or no matches are found between adata and geodataframe.

    Example:
        >>> import anndata as ad
        >>> import geopandas as gpd
        >>> from opendvp.io.adata_to_qupath import adata_to_qupath
        >>> # Create example AnnData
        >>> import pandas as pd
        >>> obs = pd.DataFrame({'CellID': [1, 2, 3], 'celltype': ['A', 'B', 'A']})
        >>> adata = ad.AnnData(obs=obs)
        >>> # Create example GeoDataFrame
        >>> from shapely.geometry import Point
        >>> gdf = gpd.GeoDataFrame({'CellID': [1, 2, 3]}, geometry=[Point(0,0), Point(1,1), Point(2,2)])
        >>> # Export with classification
        >>> result = adata_to_qupath(
        ...     adata=adata,
        ...     geodataframe=gdf,
        ...     adataobs_on="CellID",
        ...     gdf_on="CellID",
        ...     classify_by="celltype",
        ...     color_dict={"A": [255,0,0], "B": [0,255,0]},
        ...     simplify_value=0.5,
        ...     save_as_detection=True
        ... )
        >>> print(result.head())
    """
    if not isinstance(adata, ad.AnnData):
        raise ValueError("adata must be an instance of anndata.AnnData")
    if not isinstance(geodataframe, geopandas.GeoDataFrame):
        raise ValueError("gdf must be an instance of geopandas.GeoDataFrame")
    if adataobs_on not in adata.obs.columns:
        raise ValueError(f"{adataobs_on} not in adata.obs.columns")
    if (gdf_on and gdf_index) or (gdf_on is None and not gdf_index):
        raise ValueError("You must set exactly one of gdf_on or gdf_index (not both, not neither).")
    if gdf_on and gdf_on not in geodataframe.columns:
        raise ValueError(f"{gdf_on} not in gdf.columns")
    if classify_by:
        if classify_by not in adata.obs.columns:
            raise ValueError(f"{classify_by} not in adata.obs.columns")
        if adata.obs[classify_by].isna().any():
            raise ValueError(f"The {classify_by} contains NaN values")
        if adata.obs[classify_by].dtype.name != 'category':
            logger.warning(f"{classify_by} is not a categorical, converting to categorical")
            adata.obs[classify_by] = adata.obs[classify_by].astype('category')        
    if color_dict and not isinstance(color_dict,dict):
        raise ValueError("provided color_dict is not a dict")
    
    # Check matches between adata and geodataframe
    adata_index_values = set(adata.obs[adataobs_on])
    gdf_index_values = set(geodataframe[gdf_on]) if gdf_on else set(geodataframe.index)
    n_matches = len(adata_index_values & gdf_index_values)
    logger.info(f"Found {n_matches} matching IDs between adata.obs['{adataobs_on}'] and geodataframe['{gdf_on}'].")
    if n_matches == 0:
        raise ValueError("No matching values between adata and geodataframe")

    gdf = geodataframe.copy()
    if save_as_detection:
        gdf['objectType'] = "detection"

    # Filter shapes of gdf by adata
    gdf = gdf[gdf[gdf_on].isin(adata.obs[adataobs_on])] if gdf_on else gdf[gdf.index.isin(adata.obs[adataobs_on])]

    if classify_by:
        index_class = adata.obs.set_index(adataobs_on)[classify_by].copy()
        if gdf_on:
            gdf['class'] = gdf[gdf_on].map(index_class).astype(str).fillna('filtered_out')
        elif gdf_index:
            gdf['class'] = gdf.index.map(index_class).astype(str).fillna('filtered_out')
        logger.info(f"Classes now in shapes: {gdf['class'].unique()}")
        color_dict = parse_color_for_qupath(color_dict, adata=adata, adata_obs_key=classify_by)
        gdf['classification'] = gdf.apply(lambda row: {'name': row['class'], 'color': color_dict[row['class']]}, axis=1)
        gdf = gdf.drop(columns='class')

    # Simplify geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        gdf['geometry'] = gdf['geometry'].simplify(simplify_value, preserve_topology=True)

    return gdf
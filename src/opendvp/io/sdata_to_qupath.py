import anndata as ad
import geopandas

from opendvp.utils import logger, parse_color_for_qupath


def sdata_to_qupath(
        sdata,
        key_to_shapes: str,
        export_path: str | None = None,
        table_key: str| None =None,
        index_table_by : str="CellID",
        classify_by: str | None = None,
        color_dict: dict | None = None,
        simplify_value : float =1.0,
        return_gdf : bool = False
) -> geopandas.GeoDataFrame | None:
    """Export shapes from a SpatialData object as QuPath-compatible detections in GeoJSON format.

    This function converts shape or label elements from a SpatialData object into a GeoDataFrame suitable for QuPath.
    It supports both direct shape elements (GeoDataFrame) and label elements (xarray.DataArray, which are polygonized).
    Optionally, it can annotate detections with class/category information and color them for QuPath visualization.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The SpatialData object containing shape or label elements and (optionally) a table element with cell metadata.
    key_to_shapes : str
        Key in sdata._shared_keys for the shape or label element. If a label (xarray.DataArray), it will be polygonized.
    export_path : str
        Path to export the detections as a GeoJSON file. Must end with '.geojson'.
    table_key : str, optional
        Key in sdata._shared_keys for the table (AnnData) element containing cell metadata. Required for classification.
    index_table_by : str, default "CellID"
        Column in the table (AnnData.obs) used to match shapes to metadata.
    classify_by : str, optional
        Column in the table (AnnData.obs) to use for classifying detections (e.g., cell type or phenotype).
        If provided, detections will be annotated and colored by this category.
    color_dict : dict, optional
        Dictionary mapping class/category names to RGB color lists (e.g., {'Tcell': [255, 0, 0]}).
        If not provided, a default color cycle will be generated.
    simplify_value : float, default 1.0
        Tolerance for geometry simplification. Set to None to disable simplification.
    return_gdf : bool, default False
        If True, returns the resulting GeoDataFrame instead of just exporting to file.

    Returns:
    -------
    geopandas.GeoDataFrame or None
        The resulting GeoDataFrame if `return_gdf` is True, otherwise None.

    Raises:
    ------
    ImportError
        If the 'spatialdata' package is not installed.
    AssertionError
        If required keys or columns are missing, or if input types are incorrect.
    ValueError
        If the shape element is not a GeoDataFrame or DataArray, or if classification fails.

    Notes:
    -----
    - The function expects that the shape and table elements are properly indexed and aligned.
    - If a label element is provided, it will be polygonized using spatialdata.to_polygons.
    - The exported GeoJSON is compatible with QuPath for detection import and visualization.
    - Requires the 'spatialdata', 'geopandas', and 'anndata' packages.
    """
    try:
        import spatialdata
    except ImportError as e:
        raise ImportError("The 'spatialdata' package is required. Use 'pip install opendvp[spatialdata]'.") from e
    import xarray

    if not isinstance(sdata, spatialdata.SpatialData):
        raise ValueError("sdata must be an instance of spatialdata.SpatialData")
    #key to shapes
    if key_to_shapes not in sdata._shared_keys:
        raise ValueError(f"key_to_shapes {key_to_shapes} not found in sdata")
    if isinstance(sdata[key_to_shapes], geopandas.geodataframe.GeoDataFrame):
        logger.info(f"Converting {key_to_shapes} geodataframe to detections")
    elif isinstance(sdata[key_to_shapes], xarray.core.dataarray.DataArray):
        logger.info(f"Converting {key_to_shapes} dataarray to polygons, and then to detections")
    else:
        raise ValueError(f"key_to_shapes {key_to_shapes} must be a geodataframe or dataarray")
    #table key
    if not isinstance(sdata[table_key], ad.AnnData):
        raise ValueError(f"table_key {table_key} must be an anndata object")
    #classify by
    if classify_by not in sdata[table_key].obs.columns:
        raise ValueError(f"classify_by {classify_by} not found in table")
    if sdata[table_key].obs[classify_by].isna().any():
        raise ValueError(f"The {classify_by} contains NaN values, potential misindexing between elements")
    if sdata[table_key].obs[classify_by].dtype.name != 'category':
        logger.warning(f"{classify_by} is not a categorical, converting to categorical")
        sdata[table_key].obs[classify_by] = sdata[table_key].obs[classify_by].astype('category')
    # shape index and table.obs.index by match
    if sdata[table_key].obs[index_table_by].dtype != sdata[key_to_shapes].index.dtype:
        logger.error("Indexing is not matching between table.obs and shapes")
        logger.error(f"sdata table indexing is: {sdata[table_key].obs.index.dtype}")
        logger.error(f"sdata table indexing is: {sdata[key_to_shapes].index.dtype}")
        return
    #export path
    if not isinstance(export_path, str):
        raise ValueError("export_path must be a string")
    if not export_path.endswith('.geojson'):
        raise ValueError("export_path must end with .geojson")
    #color dict
    if color_dict:
        if not isinstance(color_dict, dict):
            raise ValueError("color_dict must be a dictionary")
        if not set(sdata[table_key].obs[classify_by].cat.categories).issubset(set(color_dict.keys())):
            raise ValueError("categories in classify_by, must be present in color_dict")

    logger.info("Check of inputs completed, starting conversion to detections")

    #convert xarray to polygons if necessary
    if isinstance(sdata[key_to_shapes], xarray.core.dataarray.DataArray):
        logger.info(f"Converting {key_to_shapes} xarray to {key_to_shapes}_polygons element")
        logger.info("This may take a 2-10 minutes depending on the size of the array")
        sdata[f'{key_to_shapes}_polygons'] = spatialdata.to_polygons(sdata[key_to_shapes])
        logger.info(f"Conversion of {key_to_shapes} to {key_to_shapes}_polygons element complete")
        key_to_shapes = f'{key_to_shapes}_polygons'

    # name them after their cellid, this will be shown in Qupath, might be useful to track them
    logger.info("Naming detections as cellID")
    sdata[key_to_shapes]['name'] = "cellID_" + sdata[key_to_shapes]['label'].astype(int).astype(str)
    
    # label geometries as detections
    logger.info("Labeling geometries as detections, for smooth viewing in QuPath")
    sdata[key_to_shapes]['objectType'] = "detection"

    if classify_by:
        logger.info(f"Classifying detections by {classify_by}")
        logger.info(f"Classes found in table:\n{sdata[table_key].obs[classify_by].value_counts().to_string()}")
        phenotypes_series = sdata[table_key].obs.set_index(index_table_by)[classify_by]
        sdata[key_to_shapes]['class'] = sdata[key_to_shapes].index.map(phenotypes_series).astype(str)
        sdata[key_to_shapes]['class'] = sdata[key_to_shapes]['class'].replace("nan", "filtered_out")
        logger.info(f"Classes now in shapes: {sdata[key_to_shapes]['class'].unique()}")

        color_dict = parse_color_for_qupath(color_dict, adata=sdata[table_key], adata_obs_key=classify_by)

        sdata[key_to_shapes]['classification'] = sdata[key_to_shapes].apply(
            lambda x: {'name': x['class'], 'color': color_dict[x['class']]}, axis=1
        )
        
        # remove class column to keep clean
        sdata[key_to_shapes] = sdata[key_to_shapes].drop(columns='class')

    #simplify the geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        sdata[key_to_shapes]['geometry'] = sdata[key_to_shapes]['geometry'].simplify(
            simplify_value, preserve_topology=True)

    # export detections
    if export_path:
        if 'label' in sdata[key_to_shapes].columns: # sdata.to_polygonize creates double label column, we drop it
            gdf_tmp = sdata[key_to_shapes].drop(columns='label', inplace=False)
            gdf_tmp.to_file(export_path, driver='GeoJSON')    
        else:
            sdata[key_to_shapes].to_file(export_path, driver='GeoJSON')

    if return_gdf:
        return sdata[key_to_shapes]
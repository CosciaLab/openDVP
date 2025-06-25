import geopandas

from opendvp.utils import logger


def segmask_to_qupath(
    path_to_mask : str,
    simplify_value : float = 1.0,
    save_as_detection: bool = True,
    ) -> geopandas.GeoDataFrame | None:
    """Convert a segmentation mask (TIFF) to QuPath-compatible detections as a GeoDataFrame or GeoJSON file.

    This function loads a 2D segmentation mask image, converts it to polygons using spatialdata,
    and prepares the result for QuPath detection import. Optionally, it can simplify the geometry,
    export the detections as a GeoJSON file, and/or return the resulting GeoDataFrame.

    Parameters
    ----------
    path_to_mask : str
        Path to the segmentation mask image (must be a .tif file).
    simplify_value : float, default 1
        Tolerance for geometry simplification. Set to None to disable simplification.
    export_path : str, optional
        If provided, writes the detections to this path as a GeoJSON file.
    return_gdf : bool, default False
        If True, returns the resulting GeoDataFrame. If False, returns None.

    Returns:
    -------
    geopandas.GeoDataFrame or None
        The resulting GeoDataFrame if `return_gdf` is True, otherwise None.

    Raises:
    ------
    ImportError
        If required packages (dask, dask_image, spatialdata) are not installed.
    ValueError
        If input types or file extensions are incorrect.

    Notes:
    -----
    - Requires the 'dask', 'dask_image', and 'spatialdata' packages.
    - The exported GeoJSON is compatible with QuPath for detection import and visualization.
    """
    try:
        import dask.array as da  # type: ignore
        import dask_image.imread  # type: ignore
        import spatialdata  # type: ignore
    except ImportError as e:
        raise ImportError("The 'spatialdata' package is required. Use 'pip install opendvp[spatialdata]'.") from e

    # checks
    if not isinstance(path_to_mask, str):
        raise ValueError("path_to_mask must be a string")
    if not path_to_mask.endswith('.tif'):
        raise ValueError("path_to_mask must end with .tif")

    #create empty sdata
    sdata = spatialdata.SpatialData()
    # load image
    mask = dask_image.imread.imread(path_to_mask)
    mask = da.squeeze(mask)
    sdata['mask'] = spatialdata.models.Labels2DModel.parse(mask)
    # convert to polygons
    sdata['mask_polygons'] = spatialdata.to_polygons(sdata['mask'])
    gdf = sdata['mask_polygons']
    if save_as_detection:
        gdf['objectType'] = "detection"
    #simplify the geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        gdf['geometry'] = gdf['geometry'].simplify(simplify_value, preserve_topology=True)
    #remove label column
    gdf = gdf.drop(columns='label')

    return gdf
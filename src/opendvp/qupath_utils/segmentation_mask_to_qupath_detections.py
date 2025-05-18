
def segmentation_mask_to_qupath_detections(
        path_to_mask: str,
        simplify_value: float=1,
    ): 

    # checks
    assert isinstance(path_to_mask, str), "path_to_mask must be a string"
    assert path_to_mask.endswith('.tif'), "path_to_mask must end with .tif"

    sdata = spatialdata.SpatialData()
    # load image
    mask = dask_image.imread.imread(path_to_mask)
    mask = da.squeeze(mask)
    sdata['mask'] = spatialdata.models.Labels2DModel.parse(mask)

    # convert to polygons
    sdata['mask_polygons'] = spatialdata.to_polygons(sdata['mask'])
    gdf = sdata['mask_polygons']

    gdf['objectType'] = "detection"

    #simplify the geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        gdf['geometry'] = gdf['geometry'].simplify(simplify_value, preserve_topology=True)

    #remove label column
    gdf.drop(columns='label', inplace=True)

    return gdf
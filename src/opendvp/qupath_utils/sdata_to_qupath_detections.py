def sdata_to_qupath_detections(
        sdata,
        key_to_shapes: str,
        export_path: str,
        table_key: str=None,
        index_table_by : str="CellID",
        classify_by: str=None,
        color_dict: dict=None,
        simplify_value=1.0,
        return_gdf=False
):
    """
    Export the shapes as detections
        adata: anndata object
        key_to_shapes: key in sdata._shared_keys where the shapes, must be shape element or labels, if labels it will be polygonized
        table_key: key in sdata._shared_keys referring to table element
        classify_by: key in table for categorical
        export_path: path to export the detections as geojson
        color_dict: dictionary with color mappings found in table.uns, should match keys in classify_by
        simplify_value: simplify the geometry, tolerance 1.0 is default, replace with None for no simplification
    """

    #checks 
    assert isinstance(sdata, spatialdata.SpatialData), "sdata must be an instance of spatialdata.SpatialData"
    #key to shapes
    assert key_to_shapes in sdata._shared_keys, f"key_to_shapes {key_to_shapes} not found in sdata"
    if isinstance(sdata[key_to_shapes], geopandas.geodataframe.GeoDataFrame):
        logger.info(f"Converting {key_to_shapes} geodataframe to detections")
    elif isinstance(sdata[key_to_shapes], xarray.core.dataarray.DataArray):
        logger.info(f"Converting {key_to_shapes} dataarray to polygons, and then to detections")
    else:
        raise ValueError(f"key_to_shapes {key_to_shapes} must be a geodataframe or dataarray")
    #table key
    assert isinstance(sdata[table_key], ad.AnnData), f"table_key {table_key} must be an anndata object"
    #classify by
    assert classify_by in sdata[table_key].obs.columns, f"classify_by {classify_by} not found in table"
    assert not sdata[table_key].obs[classify_by].isna().any(), f"The {classify_by} contains NaN values, potential misindexing between elements"
    if not sdata[table_key].obs[classify_by].dtype.name == 'category':
        logger.warning(f"{classify_by} is not a categorical, converting to categorical")
        sdata[table_key].obs[classify_by] = sdata[table_key].obs[classify_by].astype('category')

    # shape index and table.obs.index by match
    if not sdata[table_key].obs[index_table_by].dtype == sdata[key_to_shapes].index.dtype:
        logger.error("Indexing is not matching between table.obs and shapes")
        logger.error(f"sdata table indexing is: {sdata[table_key].obs.index.dtype}")
        logger.error(f"sdata table indexing is: {sdata[key_to_shapes].index.dtype}")
        return
    #export path
    assert isinstance(export_path, str), "export_path must be a string"
    assert export_path.endswith('.geojson'), "export_path must end with .geojson"
    #color dict
    if color_dict:
        assert isinstance(sdata[table_key].uns[color_dict], dict), "color_dict must be a dictionary"
        assert set(sdata[table_key].obs[classify_by].cat.categories).issubset(set(sdata[table_key].uns[color_dict].keys())), "categories in classify_by, must be present in color_dict"

    #TODO ensure that indexes match between polygon and table

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
        sdata[key_to_shapes]['class'] = sdata[key_to_shapes]['class'].replace("nan", "filtered_out") #incase filtered out cells
        logger.info(f"Classes now in shapes: {sdata[key_to_shapes]['class'].unique()}")


        color_dict = parse_color_for_qupath(color_dict, adata=sdata[table_key], adata_obs_key=classify_by)

        # if color_dict:
        #     logger.info(f"Using color_dict found in table.uns[{color_dict}]")
        #     logger.info(f"color dict looks like this: {sdata[table_key].uns[color_dict]}")
        #     color_dict = sdata[table_key].uns[color_dict]
        #     color_dict = parse_color_for_qupath(color_dict)
        # else:
        #     logger.info("No color_dict found, using defaults")
        #     default_colors = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189]]
        #     color_cycle = cycle(default_colors)
        #     color_dict = dict(zip(sdata[table_key].obs[classify_by].cat.categories.astype(str), color_cycle))
        #     logger.info(f"color_dict created: {color_dict}")

        sdata[key_to_shapes]['classification'] = sdata[key_to_shapes].apply(
            lambda x: {'name': x['class'], 'color': color_dict[x['class']]}, axis=1)
        
        # remove class column to keep clean
        sdata[key_to_shapes].drop(columns='class', inplace=True)

    #simplify the geometry
    if simplify_value is not None:
        logger.info(f"Simplifying the geometry with tolerance {simplify_value}")
        sdata[key_to_shapes]['geometry'] = sdata[key_to_shapes]['geometry'].simplify(simplify_value, preserve_topology=True)

    # export detections
    if 'label' in sdata[key_to_shapes].columns: # sdata.to_polygonize creates double label column, we drop it
        gdf_tmp = sdata[key_to_shapes].drop(columns='label', inplace=False)
        gdf_tmp.to_file(export_path, driver='GeoJSON')    
    else:
        sdata[key_to_shapes].to_file(export_path, driver='GeoJSON')

    if return_gdf:
        return sdata[key_to_shapes] 
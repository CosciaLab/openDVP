import os

import geopandas

from opendvp.logger import logger


def import_qupath_geojson_to_sdata(path_to_geojson: str, sdata, key: str):
    try:
        import spatialdata
    except ImportError as e:
        raise ImportError("The 'spatialdata' package is required for this functionality. Install with 'pip install opendvp[spatialdata]'.") from e

    """
    Import the geojson from qupath to sdata
    Args:
        path_to_geojson: path to the geojson file
        sdata: spatialdata object
        key: key to store the geodataframe in sdata
    """
    
    # path to geojson
    assert isinstance(path_to_geojson, str), "path_to_geojson must be a string"
    assert path_to_geojson.endswith('.geojson'), "path_to_geojson must end with .geojson"
    assert os.path.isfile(path_to_geojson), f"path_to_geojson {path_to_geojson} not found"
    # sdata
    assert isinstance(sdata, spatialdata.SpatialData), "sdata must be an instance of spatialdata.SpatialData"
    # key
    assert key not in sdata._shared_keys, f"key {key} already present in sdata"
    
    logger.info(f"Reading the geojson from {path_to_geojson}")
    gdf = geopandas.read_file(path_to_geojson)
    logger.info(f"Geojson read, storing in sdata with key {key}")
    sdata[key] = spatialdata.models.ShapesModel.parse(gdf)
    return sdata
import os

import geopandas

from opendvp.utils import logger


def geojson_to_sdata(
    path_to_geojson: str, 
    sdata,  # noqa: ANN001
    key: str
) -> "spatialdata.SpatialData":  # type: ignore # noqa: F821
    """Import the geojson from QuPath to a SpatialData object.

    Parameters
    ----------
    path_to_geojson : str
        Path to the geojson file.
    sdata : spatialdata.SpatialData
        The SpatialData object to which the geojson will be added.
    key : str
        Key to store the geodataframe in sdata.

    Returns:
    -------
    spatialdata.SpatialData
        The updated SpatialData object with the new geojson data.

    Raises:
    ------
    ImportError
        If the 'spatialdata' package is not installed.
    AssertionError
        If input types or file existence checks fail.
    """
    try:
        import spatialdata # type: ignore  # noqa: I001
    except ImportError as e:
        raise ImportError("The 'spatialdata' package is required. Use 'pip install opendvp[spatialdata]'.") from e

    if not path_to_geojson.endswith('.geojson'):
        raise ValueError("path_to_geojson must end with .geojson")
    if not os.path.isfile(path_to_geojson):
        raise ValueError(f"path_to_geojson {path_to_geojson} not found")
    # key
    if key not in sdata._shared_keys:
        raise ValueError(f"key {key} already present in sdata")
    
    logger.info(f"Reading the geojson from {path_to_geojson}")
    gdf = geopandas.read_file(path_to_geojson)
    logger.info(f"Geojson read, storing in sdata with key {key}")
    sdata[key] = spatialdata.models.ShapesModel.parse(gdf)
    return sdata
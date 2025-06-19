import ast

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd

from opendvp.utils import logger


def filter_by_annotation(
    adata: ad.AnnData,
    path_to_geojson: str,
    any_label: str = "artefact",
) -> ad.AnnData:
    """Filter cells by annotation in a geojson file using spatial indexing.

    This function assigns annotation classes to cells in an AnnData object by spatially joining cell centroids
    with polygons from a GeoJSON file. Each annotation class becomes a boolean column in `adata.obs`.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with cell centroids in `adata.obs[['X_centroid', 'Y_centroid']]` and unique 'CellID'.
    path_to_geojson : str
        Path to the GeoJSON file containing polygon annotations with a 'classification' property.
    any_label : str, default 'artefact'
        Name for the column indicating if a cell is inside any annotation.
    plot_QC : bool, default True
        If True, enables plotting for quality control (not implemented).

    Returns:
    -------
    ad.AnnData
        The input AnnData with new boolean columns in `.obs` for each annotation class and a summary column.

    Raises:
    ------
    ValueError
        If the GeoJSON is missing geometry, not polygons, or if required columns are missing.
    """
    logger.info(" ---- filter_by_annotation : version number 2.0.1 ----")
    logger.info(" Each class of annotation will be a different column in adata.obs")
    logger.info(" TRUE means cell was inside annotation, FALSE means cell not in annotation")

    # Load GeoJSON
    gdf = gpd.read_file(path_to_geojson)
    if gdf.geometry is None:
        raise ValueError("No geometry found in the geojson file")
    if gdf.geometry.type.unique()[0] != 'Polygon':
        raise ValueError("Only polygon geometries are supported")
    logger.info(f"GeoJson loaded, detected: {len(gdf)} annotations")

    # Extract class names
    gdf['class_name'] = gdf['classification'].apply(
        lambda x: ast.literal_eval(x).get('name') if isinstance(x, str) else x.get('name')
    )

    # Check required columns in AnnData
    if 'X_centroid' not in adata.obs or 'Y_centroid' not in adata.obs or 'CellID' not in adata.obs:
        raise ValueError("AnnData must have 'X_centroid', 'Y_centroid', and 'CellID' columns in .obs")

    # Convert AnnData cell centroids to a GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        adata.obs.copy(),
        geometry=gpd.points_from_xy(adata.obs['X_centroid'], adata.obs['Y_centroid']),
        crs=gdf.crs
    )
    # This checks if any point, from point_gdf, is inside any polygon from gdf.
    joined = gpd.sjoin(points_gdf, gdf[['geometry', 'class_name']], how='left', predicate='within')

    #This is a tad complex
    df_grouped = joined.groupby("CellID")['class_name'].agg(lambda x: list(set(x))).reset_index()
    df_expanded = df_grouped.copy()

    for cat in set(cat for sublist in df_grouped['class_name'] for cat in sublist):
        df_expanded[cat] = df_expanded['class_name'].apply(lambda x: cat in x)
    
    df_expanded = df_expanded.drop(columns=['class_name', np.nan])
    df_expanded[any_label] = df_expanded.drop(columns=["CellID"]).any(axis=1)
    category_cols = [col for col in df_expanded.columns if col not in ["CellID", any_label]]
    df_expanded["annotation"] = df_expanded[category_cols].apply(lambda row: next((col for col in category_cols if row[col]), None), axis=1)

    adata.obs = pd.merge(adata.obs, df_expanded, on="CellID")
    return adata
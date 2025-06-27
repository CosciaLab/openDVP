import ast
from typing import Sequence

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd

from opendvp.utils import logger


def filter_by_annotation(
    adata: ad.AnnData,
    path_to_geojson: str,
    cell_id_col: str = "CellID",
    x_y: Sequence[str] = ("X_centroid", "Y_centroid"),
    any_label: str = "ANY",
    ) -> ad.AnnData:
    """Filter cells by annotation in a geojson file using spatial indexing.

    This function assigns annotation classes to cells in an AnnData object by spatially joining cell centroids
    with polygons from a GeoJSON file. Each annotation class becomes a boolean column in `adata.obs`.

    Parameters:
    ----------
    adata : ad.AnnData
        AnnData object with cell centroids in `adata.obs[['X_centroid', 'Y_centroid']]` and unique 'CellID'.
    path_to_geojson : str
        Path to the GeoJSON file containing polygon annotations with a 'classification' property.
    cell_id_col : str, default 'CellID'
        Name of the column in `adata.obs` that uniquely identifies each cell.
    x_y : Sequence[str], default ("X_centroid", "Y_centroid")
        Names of columns in `adata.obs` containing the X and Y spatial coordinates of cells.
    any_label : str, default 'ANY'
        Name for the column indicating if a cell is inside any annotation.
        This is to be used for naming the group of annotations, for example:
        If pathologist annotated tissue regions, call this: 'tissue_ann'
        If microscopist annotated imaging artefacts, call this: 'img_arts'

    Returns:
    -------
    ad.AnnData
        The input AnnData with new boolean columns in `.obs` for each annotation class and a summary column.

    Raises:
    ------
    ValueError
        If the GeoJSON is missing geometry, not polygons, or if required columns are missing.
    """
    logger.info(" Each class of annotation will be a different column in adata.obs")
    logger.info(" TRUE means cell was inside annotation, FALSE means cell not in annotation")

    # Create a copy of adata to avoid modifying the original object in place.
    # All subsequent modifications will be applied to adata_copy.
    adata_copy = adata.copy()

    # Load GeoJSON
    gdf = gpd.read_file(path_to_geojson)
    if gdf.geometry is None:
        raise ValueError("No geometry found in the geojson file")
    if not all(geom_type == 'Polygon' for geom_type in gdf.geometry.type.unique()):
        raise ValueError("Only polygon geometries are supported")

    logger.info(f"GeoJSON loaded, detected: {len(gdf)} annotations")

    # Extract class names from GeoJSON properties
    gdf['class_name'] = gdf['classification'].apply(
        lambda x: ast.literal_eval(x).get('name') if isinstance(x, str) else x.get('name')
    )

    # Get all unique class names from the GeoJSON, which should become columns in adata.obs
    all_geojson_classes = gdf['class_name'].dropna().unique().tolist()

    # Check required columns in adata.obs
    required_cols = list(x_y) + [cell_id_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols: # Check against original adata.obs for missing columns
        raise ValueError(f"Required column(s) missing from adata.obs: {', '.join(missing_cols)}")
    # Convert AnnData cell centroids to a GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        adata_copy.obs, geometry=gpd.points_from_xy(adata_copy.obs[x_y[0]], adata_copy.obs[x_y[1]]), crs=gdf.crs
    )    
    # Perform spatial join: find which points fall within which polygons
    joined = gpd.sjoin(points_gdf, gdf[['geometry', 'class_name']], how='left', predicate='within')

    # --- Process spatial join results to create annotation columns ---
    
    # 1. Create boolean columns for each unique annotation class
    #    Use get_dummies to convert 'class_name' into one-hot encoded columns.
    #    Then group by the cell_id_col and take the maximum (True if cell is in at least one instance of that class).
    #    This handles cases where a cell might intersect multiple polygons of the same class.
    annotation_dummies = pd.get_dummies(joined[[cell_id_col, 'class_name']], columns=['class_name'], prefix='', prefix_sep='')
    # Group by cell_id_col and take max to handle multiple annotations for a single cell
    annotation_presence = annotation_dummies.groupby(cell_id_col).max()

    # Drop the 'nan' column if it exists (this column represents cells not inside any polygon)
    if np.nan in annotation_presence.columns:
        annotation_presence = annotation_presence.drop(columns=[np.nan])

    # Ensure all unique GeoJSON classes are present as columns, filling missing ones with False
    for geo_class in all_geojson_classes:
        if geo_class not in annotation_presence.columns:
            annotation_presence[geo_class] = False
    annotation_presence = annotation_presence.astype(bool) # Ensure all class columns are boolean

    # 2. Create the 'any_label' column: True if the cell is in ANY annotation class
    actual_annotation_cols = all_geojson_classes # These are the columns representing actual annotation classes
    annotation_presence[any_label] = annotation_presence[actual_annotation_cols].any(axis=1)

    # 3. Create the 'annotation' column: a single string representing the first annotation found
    #    Initialize with 'Unannotated' for cells not in any polygon.
    annotation_presence['annotation'] = annotation_presence[actual_annotation_cols].apply(
        lambda row: next((col for col in actual_annotation_cols if row[col]), 'Unannotated'), axis=1
    )

    # 4. Merge the new annotation columns back into adata.obs
    adata_copy.obs = adata_copy.obs.merge(annotation_presence.reset_index(), on=cell_id_col, how='left')

    # Fill any NaNs introduced by the merge (for cells that had no spatial annotation)
    for col in actual_annotation_cols + [any_label]: # Apply fillna to all relevant boolean columns
        adata_copy.obs[col] = adata_copy.obs[col].fillna(False)
    adata_copy.obs['annotation'] = adata_copy.obs['annotation'].fillna('Unannotated')

    return adata_copy
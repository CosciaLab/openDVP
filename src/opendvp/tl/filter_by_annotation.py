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


    adata_copy = adata.copy()
    gdf = gpd.read_file(path_to_geojson)
    if gdf.geometry is None:
        raise ValueError("No geometry found in the geojson file")
    if not all(geom_type == 'Polygon' for geom_type in gdf.geometry.geom_type.unique()):
        raise ValueError("Only polygon geometries are supported")
    logger.info(f"GeoJSON loaded, detected: {len(gdf)} annotations")

    # Extract class names from GeoJSON properties
    gdf['class_name'] = gdf['classification'].apply(
        lambda x: ast.literal_eval(x).get('name') if isinstance(x, str) else x.get('name')
    )

    all_geojson_classes = gdf['class_name'].dropna().unique().tolist()

    required_cols = list(x_y) + [cell_id_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Required column(s) missing from adata.obs: {', '.join(missing_cols)}")
    
    # Convert AnnData cell centroids to a GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        adata_copy.obs, geometry=gpd.points_from_xy(adata_copy.obs[x_y[0]], adata_copy.obs[x_y[1]]), crs=gdf.crs
    )    
    # Perform spatial join: find which points fall within which polygons
    joined = gpd.sjoin(points_gdf, gdf[['geometry', 'class_name']], how='left', predicate='within')

    # --- Process spatial join results to create annotation columns ---
    
    # 1. Create boolean columns for each unique annotation class
    annotated_cells = joined.dropna(subset=['class_name'])

    # Use crosstab to create a matrix of cell IDs vs. annotation classes.
    if not annotated_cells.empty:
        presence_matrix = pd.crosstab(annotated_cells[cell_id_col], annotated_cells['class_name'])
        annotation_presence = (presence_matrix > 0)
    else:
        logger.warning("No cells were found inside any of the provided annotations.")
        annotation_presence = pd.DataFrame(index=pd.Index([], name=cell_id_col))

    # Ensure all unique GeoJSON classes are present as columns, filling missing ones with False.
    for geo_class in all_geojson_classes:
        if geo_class not in annotation_presence.columns:
            annotation_presence[geo_class] = False

    # Reindex to include all original cells, filling unannotated ones with False.
    all_cell_ids = adata_copy.obs[cell_id_col].unique()
    annotation_presence = annotation_presence.reindex(all_cell_ids, fill_value=False)
    annotation_presence = annotation_presence.astype(bool)

    # 2. Create the 'any_label' column: True if the cell is in ANY annotation class
    actual_annotation_cols = all_geojson_classes # These are the columns representing actual annotation classes
    annotation_presence[any_label] = annotation_presence[actual_annotation_cols].any(axis=1)

    # 3. Create the 'annotation' column: a single string representing the annotation status.
    num_annotations = annotation_presence[actual_annotation_cols].sum(axis=1)
    annotation_presence['annotation'] = 'Unannotated'
    annotation_presence.loc[num_annotations > 1, 'annotation'] = 'MIXED'
    
    # Set single class name for cells in exactly one annotation
    single_mask = num_annotations == 1
    if single_mask.any():
        annotation_presence.loc[single_mask, 'annotation'] = annotation_presence.loc[
            single_mask, actual_annotation_cols].idxmax(axis=1)

    # 4. Merge the new annotation columns back into adata.obs
    adata_copy.obs = adata_copy.obs.merge(annotation_presence.reset_index(), on=cell_id_col, how='left')

    # Fill any NaNs introduced by the merge (for cells that had no spatial annotation)
    for col in actual_annotation_cols + [any_label]:
        adata_copy.obs[col] = adata_copy.obs[col].fillna(False)
    adata_copy.obs['annotation'] = adata_copy.obs['annotation'].fillna('Unannotated')

    return adata_copy
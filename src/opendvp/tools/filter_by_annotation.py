from opendvp.logger import logger
import ast
import geopandas as gpd
import pandas as pd
import anndata as ad
import numpy as np

def filter_by_annotation(adata, path_to_geojson, any_label="artefact", plot_QC=True) -> ad.AnnData:
    """ Filter cells by annotation in a geojson file efficiently using spatial indexing """

    # 100x faster

    logger.info(" ---- filter_by_annotation : version number 2.0.1 ----")
    logger.info(" Each class of annotation will be a different column in adata.obs")
    logger.info(" TRUE means cell was inside annotation, FALSE means cell not in annotation")
    
    # Load GeoJSON
    gdf = gpd.read_file(path_to_geojson)
    assert gdf.geometry is not None, "No geometry found in the geojson file"
    assert gdf.geometry.type.unique()[0] == 'Polygon', "Only polygon geometries are supported"
    
    logger.info(f"GeoJson loaded, detected: {len(gdf)} annotations")

    # Extract class names
    gdf['class_name'] = gdf['classification'].apply(lambda x: ast.literal_eval(x).get('name') if isinstance(x, str) else x.get('name'))

    # Convert AnnData cell centroids to a GeoDataFrame
    points_gdf = gpd.GeoDataFrame(adata.obs.copy(), 
                                  geometry=gpd.points_from_xy(adata.obs['X_centroid'], adata.obs['Y_centroid']),
                                  crs=gdf.crs)  # Assume same CRS
    
    joined = gpd.sjoin(points_gdf, gdf[['geometry', 'class_name']], how='left', predicate='within')
    
    df_grouped = joined.groupby("CellID")['class_name'].agg(lambda x: list(set(x))).reset_index() #fails here
    df_expanded = df_grouped.copy()
    for cat in set(cat for sublist in df_grouped['class_name'] for cat in sublist):
        df_expanded[cat] = df_expanded['class_name'].apply(lambda x: cat in x)
    
    df_expanded.drop(columns=['class_name', np.nan], inplace=True)
    df_expanded[any_label] = df_expanded.drop(columns=["CellID"]).any(axis=1)
    category_cols = [col for col in df_expanded.columns if col not in ["CellID", any_label]]
    df_expanded["annotation"] = df_expanded[category_cols].apply(lambda row: next((col for col in category_cols if row[col]), None), axis=1)

    adata.obs = pd.merge(adata.obs, df_expanded, on="CellID")

    # if plot_QC:

    #     #plotting
    #     labels_to_plot = list(gdf.class_name.unique())
    #     max_x, max_y = adata.obs[['X_centroid', 'Y_centroid']].max()
    #     min_x, min_y = adata.obs[['X_centroid', 'Y_centroid']].min()

    #     tmp_df_ann = adata.obs[adata.obs['annotation'].isin(labels_to_plot)]
    #     tmp_df_notann = adata.obs[~adata.obs['annotation'].isin(labels_to_plot)].sample(frac=0.2, random_state=0).reset_index(drop=True)

    #     fig, ax = plt.subplots(figsize=(7,5))
    #     sns.scatterplot(data=tmp_df_notann, x='X_centroid', y='Y_centroid', linewidth=0, s=2, alpha=0.1)
    #     sns.scatterplot(data=tmp_df_ann, x='X_centroid', y='Y_centroid', hue='annotation', palette='bright', linewidth=0, s=8)

    #     plt.xlim(min_x, max_x)
    #     plt.ylim(max_y, min_y)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=3)

    #     # Show value counts
    #     value_counts = tmp_df_ann['annotation'].value_counts()
    #     value_counts_str = "\n".join([f"{cat}: {count}" for cat, count in value_counts.items()])

    #     plt.gca().text(1.25, 1, f"Cells Counts:\n{value_counts_str}",
    #             transform=plt.gca().transAxes, 
    #             fontsize=12, 
    #             verticalalignment='top',
    #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    #     plt.show()

    #     #drop object columns ( this would block saving to h5ad)
    #     adata.obs = adata.obs.drop(columns=['annotation'])

    return adata
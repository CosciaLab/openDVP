from loguru import logger
import time
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
import geopandas as gpd
import ast

# General assumption: TRUE means keep, FALSE means filter out

def filter_adata_by_gates(adata: ad.AnnData, gates: pd.DataFrame, sample_id=None) -> ad.AnnData:
    """ Filter the adata object by the gates """
    logger.info(" ---- filter_adata_by_gates : version number 1.0.0 ----")
    time_start = time.time()
    assert gates.marker_id.isin(adata.var.index).all(), "Some markers in the gates are not present in the adata object"
    
    if sample_id is not None:
        assert sample_id in gates.columns, "The sample_id is not present in the gates"
        gates = gates[gates['sample_id']==sample_id]
    
    adata = adata[:, gates.marker_id]
    logger.info(f" ---- filter_adata_by_gates is done, took {int(time.time() - time_start)}s  ----")
    return adata

def filter_by_abs_value(adata, marker, value=None, quantile=None, direction='above', plot=False) -> ad.AnnData:
    """ 
    Filter cells by absolute value 
    Args:
        adata: anndata object
        marker: name of the marker to filter, string present in adata.var_names
        value: value to use as threshold
        quantile: quantile to use as threshold
        direction: 'above' or 'below', denoting which cells are kept
    """

    logger.info(" ---- filter_by_abs_value : version number 1.1.0 ----")
    time_start = time.time()

    # checks
    assert type(adata) is ad.AnnData, "adata should be an AnnData object"
    assert marker in adata.var_names, f"Marker {marker} not found in adata.var_names"
    # value or quantile
    if value is not None:
        assert quantile is None, "Only one of value or quantile should be provided"
        assert isinstance(value, (int, float)), "Value should be a number"
    elif quantile is not None:
        assert value is None, "Only one of value or quantile should be provided"
        assert isinstance(quantile, float), "Quantile should be a float"
        assert 0 < quantile < 1, "Quantile should be between 0 and 1"
    else:
        raise ValueError("Either value or quantile should be provided")
    # direction
    assert direction in ['above', 'below'], "Direction should be either 'above' or 'below'"

    # set objects up 
    adata_copy = adata.copy()
    df = pd.DataFrame(data=adata_copy.X, columns=adata_copy.var_names)

    # calculate threshold
    if value is None:
        threshold = df[marker].quantile(quantile)
    else:
        threshold = value

    # Filter
    label = f"{marker}_{direction}_{threshold}"
    operator = '>' if direction == 'above' else '<'
    df[label] = df.eval(f"{marker} {operator} @threshold")
    adata_copy.obs[label] = df[label].values
    logger.info(f"Number of cells with {marker} {direction} {threshold}: {sum(df[label])}")

    if plot:
        sns.histplot(df[marker], bins=500)
        plt.yscale('log')
        plt.xscale('log')
        plt.title(f'{marker} distribution')
        plt.axvline(threshold, color='black', linestyle='--', alpha=0.5)

        if direction == 'above':
            plt.text(threshold + 10, 1000, f"cells with {marker} > {threshold}", fontsize=9, color='black')
        elif direction == 'below':
            plt.text(threshold - 10, 1000, f"cells with {marker} < {threshold}", fontsize=9, color='black', horizontalalignment='right')
        plt.show()

    logger.info(f" ---- filter_by_abs_value is done, took {int(time.time() - time_start)}s  ----")
    return adata_copy

def filter_by_ratio(adata, end_cycle, start_cycle, label="DAPI", min_ratio=0.5, max_ratio=1.05) -> ad.AnnData:
    """ Filter cells by ratio """

    logger.info(" ---- filter_by_ratio : version number 1.1.0 ----")
    #adapt to use with adata
    time_start = time.time()

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data=adata.X, columns=adata.var_names)
    df[f'{label}_ratio'] = df[end_cycle] / df[start_cycle]
    df[f'{label}_ratio_pass_nottoolow'] = df[f'{label}_ratio'] > min_ratio
    df[f'{label}_ratio_pass_nottoohigh'] = df[f'{label}_ratio'] < max_ratio
    df[f'{label}_ratio_pass'] = df[f'{label}_ratio_pass_nottoolow'] & df[f'{label}_ratio_pass_nottoohigh']

    # Pass to adata object
    adata.obs[f'{label}_ratio'] = df[f'{label}_ratio'].values
    adata.obs[f'{label}_ratio_pass_nottoolow']     = df[f'{label}_ratio_pass_nottoolow'].values
    adata.obs[f'{label}_ratio_pass_nottoohigh']    = df[f'{label}_ratio_pass_nottoohigh'].values
    adata.obs[f'{label}_ratio_pass']            = adata.obs[f'{label}_ratio_pass_nottoolow'] & adata.obs[f'{label}_ratio_pass_nottoohigh']

    # print out statistics
    logger.info(f"Number of cells with {label} ratio < {min_ratio}: {sum(df[f'{label}_ratio'] < min_ratio)}")
    logger.info(f"Number of cells with {label} ratio > {max_ratio}: {sum(df[f'{label}_ratio'] > max_ratio)}")
    logger.info(f"Number of cells with {label} ratio between {min_ratio} and {max_ratio}: {sum(df[f'{label}_ratio_pass'])}")
    logger.info(f"Percentage of cells filtered out: {round(100 - sum(df[f'{label}_ratio_pass'])/len(df)*100,2)}%")

    # plot histogram

    fig, ax = plt.subplots()

    sns.histplot(df[f'{label}_ratio'], color='blue')
    plt.yscale('log')

    plt.axvline(min_ratio, color='black', linestyle='--', alpha=0.5)
    plt.axvline(max_ratio, color='black', linestyle='--', alpha=0.5)
    plt.text(max_ratio + 0.05, 650, f"cells that gained >{int(max_ratio*100-100)}% {label}", fontsize=9, color='black')
    plt.text(min_ratio - 0.05, 650, f"cells that lost >{int(min_ratio*100-100)}% {label}", fontsize=9, color='black', horizontalalignment='right')

    plt.ylabel('cell count')
    plt.xlabel(f'{label} ratio (last/cycle)')
    plt.xlim(min_ratio-1, max_ratio+1)

    plt.show()

    logger.info(f" ---- filter_by_ratio is done, took {int(time.time() - time_start)}s  ----")

    return adata

def filter_by_annotation(adata, path_to_geojson, column_name="filter_by_ann") -> ad.AnnData:
    """ Filter cells by annotation in a geojson file """

    logger.info(" ---- filter_by_annotation : version number 1.3.0 ----")
    time_start = time.time()
    
    #load data
    gdf = gpd.read_file(path_to_geojson)
    assert gdf.geometry is not None, "No geometry found in the geojson file"
    assert gdf.geometry.type.unique()[0] == 'Polygon', "Only polygon geometries are supported"
    logger.info(f"GeoJson loaded, detected: {len(gdf)} annotations")

    #process geodataframe
    gdf['class_name'] = gdf['classification'].apply(lambda x: ast.literal_eval(x).get('name') if isinstance(x, str) else x.get('name'))

    #process adata
    adata.obs['point_geometry'] = adata.obs.apply(lambda cell: shapely.geometry.Point( cell['X_centroid'], cell['Y_centroid']), axis=1)
    
    #label based on polygon index
    def label_point_if_inside_polygon(point, polygons):
        for i, polygon in enumerate(polygons):
            if polygon.contains(point):
                return True
        return False

    # label based on polygon class_name
    def label_cells_with_annotation_class(adata, geodataframe):
        for index, row in geodataframe.iterrows():
            adata.obs[row['class_name']] = adata.obs['point_geometry'].apply(lambda x: label_point_if_inside_polygon(x, [row['geometry']]))
    
    def label_point_if_inside_polygon_wclassname(point, geodataframe):
        for index, row in geodataframe.iterrows():
            if row['geometry'].contains(point):
                return row['class_name']
        return None

    #single column with all classes
    def label_cells_with_annotation_class_single_column(adata, geodataframe):
        adata.obs['annotation'] = adata.obs['point_geometry'].apply(lambda x: label_point_if_inside_polygon_wclassname(x, geodataframe))

    logger.info(f"Found {gdf.class_name.nunique()} unique classes in the geojson file: {gdf.class_name.unique()}")
    
    label_cells_with_annotation_class(adata, gdf)
    label_cells_with_annotation_class_single_column(adata, gdf)

    #plotting
    labels_to_plot = list(gdf.class_name.unique())
    max_x, max_y = adata.obs[['X_centroid', 'Y_centroid']].max()
    min_x, min_y = adata.obs[['X_centroid', 'Y_centroid']].min()

    tmp_df_ann = adata.obs[adata.obs['annotation'].isin(labels_to_plot)]
    tmp_df_notann = adata.obs[~adata.obs['annotation'].isin(labels_to_plot)].sample(frac=0.2, random_state=0).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(data=tmp_df_notann, x='X_centroid', y='Y_centroid', linewidth=0, s=3, alpha=0.25)
    sns.scatterplot(data=tmp_df_ann, x='X_centroid', y='Y_centroid', hue='annotation', palette='bright', linewidth=0, s=8)

    plt.xlim(min_x, max_x)
    plt.ylim(max_y, min_y)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=3)

    # Show value counts
    value_counts = tmp_df_ann['annotation'].value_counts()
    value_counts_str = "\n".join([f"{cat}: {count}" for cat, count in value_counts.items()])

    plt.gca().text(1.25, 1, f"Cells Counts:\n{value_counts_str}",
            transform=plt.gca().transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.show()

    #drop object columns ( this would block saving to h5ad)
    adata.obs = adata.obs.drop(columns=['point_geometry', 'annotation'])

    logger.info(f" ---- filter_by_annotation is done, took {int(time.time() - time_start)}s  ----")
    return adata
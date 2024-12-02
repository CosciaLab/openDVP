import pandas as pd
import anndata as ad
from loguru import logger 
import time

import numpy as np
import tifffile

import matplotlib.pyplot as plt
import seaborn as sns

import shapely
import geopandas as gpd

import scimap as sm

from scipy.spatial import Voronoi


def read_quant(csv_data_path) -> ad.AnnData:
    """
    Read the quantification data from a csv file and return an anndata object
    :param csv_data_path: path to the csv file
    :return: an anndata object
    """
    logger.info(" ---- read_quant : version number 1.1.0 ----")
    time_start = time.time()

    assert csv_data_path.endswith('.csv'), "The file should be a csv file"
    df = pd.read_csv(csv_data_path)
    df.index = df.index.astype(str)

    meta_columns = ['CellID', 'Y_centroid', 'X_centroid',
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'Orientation', 'Extent', 'Solidity']
    assert all([column in df.columns for column in meta_columns]), "The metadata columns are not present in the csv file"

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)
    variables = pd.DataFrame(
        index=data.columns,
        data={"math": [column_name.split("_")[0] for column_name in data.columns],
            "marker": ["_".join(column_name.split("_")[1:]) for column_name in data.columns]})

    adata = ad.AnnData(X=data.values, obs=metadata, var=variables)
    logger.info(f" {adata.shape[0]} cells and {adata.shape[1]} variables")
    logger.info(f" ---- read_quant is done, took {int(time.time() - time_start)}s  ----")
    return adata
    
def read_gates(gates_csv_path, sample_id=None) -> pd.DataFrame:
    """ Read the gates data from a csv file and return a dataframe """
    logger.info(" ---- read_gates : version number 1.1.0 ----")
    time_start = time.time()

    assert gates_csv_path.endswith('.csv'), "The file should be a csv file"
    gates = pd.read_csv(gates_csv_path)
    
    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    assert "gate_value" in gates.columns, "The column gate_value is not present in the csv file"
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")
    logger.info(f"  Markers found: {gates.marker_id.unique()}")
    logger.info(f"  Samples found: {gates.sample_id.unique()}")

    if sample_id is not None:
        assert "sample_id" in gates.columns, "The column sample_id is not present in the csv file"
        gates = gates[gates['sample_id']==sample_id]
        logger.info(f"  Found {gates.shape[0]} valid gates for sample {sample_id}")

    logger.info(f" ---- read_gates is done, took {int(time.time() - time_start)}s  ----")
    return gates

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

def process_gates_for_sm(gates:pd.DataFrame, sample_id:int) -> pd.DataFrame:
    """ Process gates dataframe to be in log1p scale """
    logger.info(" ---- process_gates_for_sm : version number 1.2.0 ----")
    time_start = time.time()

    df = gates.copy()

    df['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = df[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'marker', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap

def lazy_image_check(image_path):
    """ Check the image metadata without loading the image """
    logger.info(" ---- lazy_image_check : version number 1.0.0 ----")
    time_start = time.time()

    with tifffile.TiffFile(image_path) as image:
        # Getting the metadata
        shape = image.series[0].shape
        dtype = image.pages[0].dtype

        n_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        estimated_size_bytes = n_elements * bytes_per_element
        estimated_size_gb = estimated_size_bytes / 1024 / 1024 / 1024 
        
        logger.info(f"Image shape is {shape}")
        logger.info(f"Image data type: {dtype}")
        logger.info(f"Estimated size: {estimated_size_gb:.4g} GB")

    logger.info(f" ---- lazy_image_check is done, took {int(time.time() - time_start)}s  ----")

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
    logger.info(" ---- filter_by_annotation : version number 1.2.0 ----")
    time_start = time.time()
    
    gdf = gpd.read_file(path_to_geojson)
    assert gdf.geometry is not None, "No geometry found in the geojson file"
    assert gdf.geometry.type.unique()[0] == 'Polygon', "Only polygon geometries are supported"
    logger.info(f"GeoJson loaded, detected: {len(gdf)} annotations")

    adata.obs['point_geometry'] = adata.obs.apply(lambda cell: shapely.geometry.Point( cell['X_centroid'], cell['Y_centroid']), axis=1)
    
    def label_point_if_inside_polygon(point, polygons):
        for i, polygon in enumerate(polygons):
            if polygon.contains(point):
                return f"ann_{i+1}"
        return "not_found"
    
    adata.obs['ann'] = adata.obs['point_geometry'].apply(lambda cell: label_point_if_inside_polygon(cell, gdf.geometry))
    adata.obs[column_name] = adata.obs['ann'] == "not_found"
    logger.info("Labelled cells with annotations if they were found inside")
    logger.info(f"Number of cells not found inside any annotation: {sum(adata.obs[column_name])}")

    #plotting
    labels_to_plot = list(adata.obs['ann'].unique())
    labels_to_plot.remove("not_found")
    max_x, max_y = adata.obs[['X_centroid', 'Y_centroid']].max()

    tmp_df_ann = adata.obs[adata.obs['ann'].isin(labels_to_plot)]
    tmp_df_Keep = adata.obs[adata.obs['ann']=="not_found"].sample(frac=0.2, random_state=0).reset_index(drop=True)

    sns.scatterplot(data=tmp_df_Keep, x='X_centroid', y='Y_centroid', hue='ann', palette='grey', linewidth=0, s=3, alpha=0.1)
    sns.scatterplot(data=tmp_df_ann, x='X_centroid', y='Y_centroid', hue='ann', palette='bright', linewidth=0, s=8)

    plt.xlim(0, max_x)
    plt.ylim(max_y, 0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=3)

    # Show value counts
    value_counts = tmp_df_ann['ann'].value_counts()
    value_counts_str = "\n".join([f"{cat}: {count}" for cat, count in value_counts.items()])

    plt.gca().text(1.35, 1, f"Cells Counts:\n{value_counts_str}",
            transform=plt.gca().transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.show()

    #drop object columns ( this would block saving to h5ad)
    adata.obs = adata.obs.drop(columns=['point_geometry', 'ann'])

    logger.info(f" ---- filter_by_annotation is done, took {int(time.time() - time_start)}s  ----")
    return adata

def filter_by_abs_value(adata, marker, value=None, quantile=None, direction='above') -> ad.AnnData:
    """ Filter cells by absolute value """

    logger.info(" ---- filter_by_abs_value : version number 1.0.0 ----")
    time_start = time.time()

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data=adata.X, columns=adata.var_names)

    #set threshold
    
    if value is None:
        threshold = df[marker].quantile(quantile)
    else:
        threshold = value

    if direction == 'above':
        df[f'{marker}_abs_above_value'] = df[marker] > threshold
        adata.obs[f'{marker}_abs_above_value'] = df[f'{marker}_abs_above_value'].values
        logger.info(f"Number of cells with {marker} {direction} {threshold}: {sum(df[f'{marker}_abs_above_value'])}")

    elif direction == 'below':
        df[f'{marker}_abs_below_value'] = df[marker] < threshold
        adata.obs[f'{marker}_abs_below_value'] = df[f'{marker}_abs_below_value'].values
        logger.info(f"Number of cells with {marker} {direction} {threshold}: {sum(df[f'{marker}_abs_below_value'])}")

    else:
        raise ValueError("Direction should be either 'above' or 'below'")

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
    return adata

def negate_var_by_ann(adata, path_to_geojson, marker_column, value_to_impute, label) -> ad.AnnData:

    # first label adata
    adata = filter_by_annotation(adata, path_to_geojson, column_name=label)

    # create array of data to correct
    array = adata[:,marker_column].X.toarray()

    logger.info( f" how many zeroes: {np.count_nonzero(array==value_to_impute)}")
    logger.info( f"array shape {array.shape}")
    logger.info( f"array mean {array.mean()}")
    logger.info( f"data type {type(array)}")

    # impute values
    array[~adata.obs[label].values] = value_to_impute

    logger.info( f" how many zeroes after: {np.count_nonzero(array==value_to_impute)}")
    logger.info( f"array shape {array.shape}")
    logger.info( f"array mean {array.mean()}")

    # replace array in adata
    adata[:,marker_column].X = array

    return adata

def phenotype_with_gate_change(adata, gates, phenotype_matrix, sample_id, marker, new_gate, adata_in_place=False):
    """ Plot the spatial scatter plot with the new gate value """

    logger.info(" ---- phenotype_with_gate_change : version number 1.0.0 ----")

    adata_copy = adata.copy()
    gates_copy = gates.copy()

    gates_copy.loc[gates_copy['marker_id']==marker, 'gate_value'] = new_gate
    gates_copy.loc[gates_copy['marker_id']==marker, 'log1p_gate_value'] = np.log1p(new_gate)
    processed_gates = process_gates_for_sm(gates_copy, sample_id)

    logger.info(f"processed gate {processed_gates.loc[processed_gates['marker']==marker]}")

    # rescale adata
    adata_copy = sm.pp.rescale(adata_copy, gate=processed_gates, log=True, verbose=False)
    adata_copy = sm.tl.phenotype_cells (adata_copy, phenotype=phenotype_matrix, label="phenotype", verbose=False)

    custom_colours = {
        "Cancer_cells" : "red",
        "CD4_Tcells" : "peru",
        "CD8_Tcells" : "lawngreen",
        "Macrophages" : "yellow",
        "COL1A1_cells" : "deepskyblue",
        "Vimentin_cells" : "orange",
        "B_cells" : "black",
        "Unknown" : "whitesmoke"
    }

    sm.pl.spatial_scatterPlot (adata_copy, colorBy = ['phenotype'],figsize=(12,7), s=1, fontsize=10, customColors=custom_colours)
    
    # Get the current figure
    fig = plt.gcf()
    ax = plt.gca()
    plt.table(cellText=gates_copy.values, colLabels=gates_copy.columns, loc='upper right', cellLoc='center')
    plt.show()

    if adata_in_place:
        logger.info(" phenotyping saved in adata object")
        return adata_copy




# Function to calculate major axis length (bounding box diagonal)
def major_axis_length(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)

# Function to scale the farthest vertex toward the centroid
def scale_pointy_vertex(polygon, scale_factor=0.5):
    centroid = polygon.centroid
    coords = np.array(polygon.exterior.coords)
    # Calculate distances of each vertex to the centroid
    distances = np.linalg.norm(coords - np.array([centroid.x, centroid.y]), axis=1)
    # Find the index of the pointy vertex (farthest from the centroid)
    max_dist_idx = np.argmax(distances)
    # Move the pointy vertex closer to the centroid
    pointy_vertex = coords[max_dist_idx]
    new_vertex = centroid.x + scale_factor * (pointy_vertex[0] - centroid.x), centroid.y + scale_factor * (pointy_vertex[1] - centroid.y)
    # Replace the pointy vertex with the new scaled vertex
    coords[max_dist_idx] = new_vertex
    # Return the modified polygon
    return shapely.Polygon(coords)

# Main function to apply filtering and scaling based on thresholds
def process_polygons(gdf, scale_threshold, remove_threshold, scale_factor=0.5):
    def process_polygon(polygon):
        length = major_axis_length(polygon)
        # Remove polygon if the major axis length exceeds remove_threshold
        if length > remove_threshold:
            return None
        # Scale the pointy vertex if the major axis length exceeds scale_threshold
        elif length > scale_threshold:
            return scale_pointy_vertex(polygon, scale_factor)
        # Otherwise, return the original polygon
        return polygon
    # Apply the function to each geometry in the GeoDataFrame
    gdf['geometry'] = gdf['geometry'].apply(process_polygon)
    # Remove None geometries (those that exceeded the remove_threshold)
    return gdf.dropna(subset=['geometry'])

def adataobs_to_voronoi_geojson(
        df,
        imageid, 
        subset:list=None, 
        category_1:str="phenotype", 
        category_2=None, 
        output_path:str="../data/processed/"):
    """ 
    Description:
    
    """

    logger.debug(f" df shape: {df.shape}")

    df = df.copy()
    #subset per image
    df = df[(df.imageid == str(imageid))]
    logger.debug(f" df shape after imageid subset: {df.shape}")
    logger.info(f"Processing {imageid}, loaded dataframe")

    #subset per x,y
    if subset is not None:
        logger.info(f"Subsetting to {subset}")
        assert len(subset) == 4, "subset must be a list of 4 integers"
        x_min, x_max, y_min, y_max = subset
        df = df[(df.X_centroid > x_min) &
                (df.X_centroid < x_max) &
                (df.Y_centroid > y_min) &
                (df.Y_centroid < y_max)]
        #clean subset up
        df = df.reset_index(drop=True)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    logger.info("Running Voronoi")
    # run Voronoi 
    # df = df[['X_centroid', 'Y_centroid', category_1, category_2]]    
    vor = Voronoi(df[['X_centroid', 'Y_centroid']].values)
    polygons = []
    for i in range(len(df)):
        polygon = shapely.Polygon(
            [vor.vertices[vertex] for vertex in vor.regions[vor.point_region[i]]])
        polygons.append(polygon)
    df['geometry'] = polygons
    logger.info("Voronoi done")

    #transform to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    logger.info("Transformed to geodataframe")

    # filter polygons that go outside of image
    if subset is None:
        x_min = gdf['X_centroid'].min()
        x_max = gdf['X_centroid'].max()
        y_min = gdf['Y_centroid'].min()
        y_max = gdf['Y_centroid'].max()
        logger.info(f"Bounding box: x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max {y_max}")

    boundary_box = shapely.box(x_min, y_min, x_max, y_max)
    gdf = gdf[gdf.geometry.apply(lambda poly: poly.within(boundary_box))]
    logger.info("Filtered out infinite polygons")

    # filter polygons that are too large
    gdf['area'] = gdf['geometry'].area
    gdf = gdf[gdf['area'] < gdf['area'].quantile(0.98)]
    logger.info("Filtered out large polygons based on the 98% quantile")
    # filter polygons that are very pointy
    
    # TODO improve filtering by pointiness
    # gdf = process_polygons(gdf, scale_threshold=350, remove_threshold=400, scale_factor=0.3)
    # logger.info("Filtered out pointy polygons")

    # create geodataframe for each cell and their celltype
    gdf2 = gdf.copy()
    gdf2['objectType'] = 'cell'
    gdf2['classification'] = gdf2[category_1]
    
    # merge polygons based on the CN column
    if category_2:
        logger.info("Merging polygons for cellular neighborhoods")
        gdf3 = gdf.copy()
        gdf3 = gdf3.dissolve(by=category_2)
        gdf3[category_2] = gdf3.index
        gdf3 = gdf3.explode(index_parts=True)
        gdf3 = gdf3.reset_index(drop=True)
        gdf3['classification'] = gdf3[category_2].astype(str)
        
    #export to geojson
    datetime = time.strftime("%Y%m%d_%H%M")
    gdf2.to_file(f"{output_path}/{datetime}_{imageid}_cells_voronoi.geojson", driver='GeoJSON')
    if category_2:
        gdf3.to_file(f"{output_path}/{datetime}_{imageid}_RCN_voronoi.geojson", driver='GeoJSON')

    logger.success(f"Exported {imageid} to geojson")
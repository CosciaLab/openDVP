import geopandas as gpd
import anndata as ad
import shapely

def filter_by_annotation(adata, path_to_geojson) -> ad.AnnData:
    """ Filter cells by annotation in a geojson file """

    gdf = gpd.read_file(path_to_geojson)

    #needed but will be deleted before returning adata
    adata.obs['point_geometry'] = adata.obs.apply(lambda cell: shapely.geometry.Point( cell['X_centroid'], cell['Y_centroid']), axis=1)

    def label_point_if_inside_polygon(point, polygons):
        for i, polygon in enumerate(polygons):
            if polygon.contains(point):
                return f"ann_{i+1}"
        return "not_found"
    
    adata.obs['ann'] = adata.obs['point_geometry'].apply(lambda cell: label_point_if_inside_polygon(cell, gdf.geometry))
    adata.obs['filter_by_ann'] = adata.obs['ann'] == "not_found"

    return adata
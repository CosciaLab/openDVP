import anndata as ad
import geopandas as gpd
from opendvp.io import adata_to_qupath

#TODO will be split at some point

def test_adata_to_qupath_callable() -> None:
    """Test that adata_to_qupath is importable and callable."""
    assert callable(adata_to_qupath)


def test_adata_to_qupath_voronoi():
    adata = ad.read_h5ad("../test_data/io/adata.h5ad")
    gdf = adata_to_qupath(adata, mode="voronoi")
    assert gdf is not None
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf.columns
    assert all(gdf.geometry.notnull())


def test_adata_to_qupath_polygons():
    adata = ad.read_h5ad("../test_data/io/adata.h5ad")
    gdf_poly = gpd.read_file("../test_data/io/adata.geojson")
    gdf = adata_to_qupath(
        adata,
        mode="polygons",
        geodataframe=gdf_poly,
        geodataframe_index_key="label",
        adata_obs_index_key="CellID",
    )
    assert gdf is not None
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf.columns
    assert all(gdf.geometry.notnull())
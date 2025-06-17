from opendvp.io import geojson_to_sdata

def test_geojson_to_sdata_callable() -> None:
    """Test that geojson_to_sdata is importable and callable."""
    assert callable(geojson_to_sdata)

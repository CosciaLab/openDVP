from opendvp.io import sdata_to_qupath

def test_sdata_to_qupath_callable() -> None:
    """Test that sdata_to_qupath is importable and callable."""
    assert callable(sdata_to_qupath)

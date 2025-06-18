from opendvp.io import segmask_to_qupath

def test_segmask_to_qupath_callable() -> None:
    """Test that segmask_to_qupath is importable and callable."""
    assert callable(segmask_to_qupath)

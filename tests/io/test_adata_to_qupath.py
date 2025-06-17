from opendvp.io import adata_to_qupath

def test_adata_to_qupath_callable() -> None:
    """Test that adata_to_qupath is importable and callable."""
    assert callable(adata_to_qupath)

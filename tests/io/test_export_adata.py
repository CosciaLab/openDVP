from opendvp.io import export_adata

def test_export_adata_callable() -> None:
    """Test that export_adata is importable and callable."""
    assert callable(export_adata)

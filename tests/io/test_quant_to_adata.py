from opendvp.io import quant_to_adata

def test_quant_to_adata_callable() -> None:
    """Test that quant_to_adata is importable and callable."""
    assert callable(quant_to_adata)

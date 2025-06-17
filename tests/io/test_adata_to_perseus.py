from opendvp.io import adata_to_perseus


def test_adata_to_perseus_callable() -> None:
    """Test that adata_to_perseus is importable and callable."""
    assert callable(adata_to_perseus)

from opendvp.io import import_perseus

def test_import_perseus_callable() -> None:
    """Test that import_perseus is importable and callable."""
    assert callable(import_perseus)

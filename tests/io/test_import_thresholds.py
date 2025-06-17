from opendvp.io import import_thresholds

def test_import_thresholds_callable() -> None:
    """Test that import_thresholds is importable and callable."""
    assert callable(import_thresholds)

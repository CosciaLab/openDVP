from opendvp.io import export_figure

def test_export_figure_callable() -> None:
    """Test that export_figure is importable and callable."""
    assert callable(export_figure)

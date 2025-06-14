from opendvp.plotting import plot_dynamic_histogram
import pandas as pd

def test_plot_dynamic_histogram_runs() -> None:
    """Test that plot_dynamic_histogram runs without error on simple data."""
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # Should not raise
    plot_dynamic_histogram(df, "A", bins=5)

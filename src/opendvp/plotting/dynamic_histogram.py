import numpy as np
import pandas as pd
import plotly.graph_objects as go


def dynamic_histogram(
    df: pd.DataFrame, column: str, bins: int = 100
) -> None:
    """Plot a dynamic histogram with a threshold slider and count annotations.

    This function creates an interactive histogram using Plotly, allowing the user to adjust a threshold slider.
    Counts of values to the left and right of the threshold are displayed as annotations. The threshold can be
    interactively moved, and the counts update accordingly.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.
    column : str
        The name of the column in `df` to plot as a histogram.
    bins : int, optional
        Number of bins for the histogram (default is 100).

    Returns:
    --------
    None
        Displays the interactive histogram in the default browser or notebook output.

    Raises:
    -------
    ValueError
        If the specified column is not in the DataFrame or contains no valid data.
    """
    # Validate input
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    data = df[column].dropna()  # Handle missing values
    if data.empty:
        raise ValueError(f"No valid data in column '{column}'.")

    # Initial threshold
    initial_threshold = data.mean()  # Start at the mean as default

    # Function to calculate counts based on threshold
    def calculate_counts(data: pd.Series, threshold: float) -> tuple[int, int]:
        left_count = (data < threshold).sum()
        right_count = (data >= threshold).sum()
        return left_count, right_count

    # Initial counts
    left_count, right_count = calculate_counts(data, initial_threshold)

    # Create figure
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(x=data, nbinsx=bins, name=column))

    # Add initial annotations for counts
    fig.update_layout(
        annotations=[
            dict(x=0.02, y=1.1, xref="paper", yref="paper", text=f"Left Count: {left_count}", showarrow=False),
            dict(x=0.98, y=1.1, xref="paper", yref="paper", text=f"Right Count: {right_count}", showarrow=False),
        ]
    )

    # Initial vertical line at threshold
    fig.add_shape(
        type="line",
        x0=initial_threshold, y0=0, x1=initial_threshold, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )

    # Function to update both the annotations and line
    def update_slider(threshold: float) -> dict:
        # Update counts
        left_count, right_count = calculate_counts(data, threshold)
        # Update annotations
        annotations = [
            dict(x=0.02, y=1.1, xref="paper", yref="paper", text=f"Left Count: {left_count}", showarrow=False),
            dict(x=0.98, y=1.1, xref="paper", yref="paper", text=f"Right Count: {right_count}", showarrow=False)]
        # Update line position
        shapes = [dict(
            type="line",
            x0=threshold, y0=0, x1=threshold, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", width=2, dash="dash"))]
        return {"annotations": annotations, "shapes": shapes}

    # Add slider with threshold steps
    thresholds = np.linspace(data.min(), data.max(), bins)
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Threshold: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "label": str(round(threshold, 2)),
                        "method": "relayout",
                        "args": [update_slider(threshold)],
                    }
                    for threshold in thresholds
                ],
            }
        ]
    )

    # Show figure
    fig.show()
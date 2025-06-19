from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def dual_axis_boxplots(
    adata_obs: pd.DataFrame,
    feature_key: str = "RCN",
    feature_1: str = "Proteins.Identified",
    feature_2: str = "Precursors.Identified",
    ylabel1: str = "Proteins Identified",
    ylabel2: str = "Precursors Identified",
    offset: float = 0.1,
    width: float = 0.2,
    point_alpha: float = 0.2,
    box1_color: str = 'skyblue',
    box2_color: str = 'lightcoral',
    median_color: str = 'black',
    scatter_color: str = 'black',
    tick1_color: str = 'blue',
    tick2_color: str = 'red',
    figsize: tuple[int, int] = (6, 6),
    return_fig: bool = False,
    show_plot: bool = True,
    ax1: Any | None = None,
    ax2: Any | None = None,
    **kwargs
) -> Figure | None:
    """Generates a dual-axis plot with boxplots and stripplots for two features grouped by a specified feature key.

    Parameters
    ----------
    adata_obs : pd.DataFrame
        DataFrame typically derived from an AnnData object's observation metadata (adata.obs).
    feature_key : str
        Column name to group by (e.g., "RCN").
    feature_1 : str
        Column name for the first feature to plot on the left y-axis.
    feature_2 : str
        Column name for the second feature to plot on the right y-axis.
    ylabel1 : str
        Label for the left y-axis.
    ylabel2 : str
        Label for the right y-axis.
    offset : float
        Offset for positioning the boxplots side-by-side.
    width : float
        Width of the boxplots.
    point_alpha : float
        Alpha transparency for the scatter plot points.
    box1_color : str
        Face color for the boxplots of feature_1.
    box2_color : str
        Face color for the boxplots of feature_2.
    median_color : str
        Color of the median line in boxplots.
    scatter_color : str
        Color of the points in stripplots.
    tick1_color : str
        Color of the left y-axis tick labels and axis label.
    tick2_color : str
        Color of the right y-axis tick labels and axis label.
    figsize : tuple
        Figure size (width, height).
    return_fig : bool
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    show_plot : bool
        If True, displays the plot. Otherwise, only returns the figure/axes.
    ax1, ax2 : matplotlib.axes.Axes, optional
        Axes objects to plot on. If None, new axes are created.
    **kwargs
        Additional keyword arguments passed to matplotlib boxplot/scatter.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    df = adata_obs.copy()
    df = df[[feature_key, feature_1, feature_2]]
    df = df.melt(id_vars=feature_key, var_name="variable", value_name="value")
    groups = df[feature_key].unique()
    try:
        groups = sorted(groups)
    except TypeError:
        pass
    x_base = np.arange(len(groups))
    group_to_x = {group: i for i, group in enumerate(groups)}
    df1 = df[df["variable"] == feature_1]
    df2 = df[df["variable"] == feature_2]
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
    else:
        fig = ax1.figure
        if ax2 is None:
            ax2 = ax1.twinx()
    for group in groups:
        x_pos = group_to_x[group]
        x1_box = x_pos - offset
        x2_box = x_pos + offset
        y1 = df1[df1[feature_key] == group]["value"].dropna()
        y2 = df2[df2[feature_key] == group]["value"].dropna()
        if not y1.empty:
            ax1.boxplot(
                y1,
                positions=[x1_box],
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=box1_color, alpha=0.6),
                medianprops=dict(color=median_color),
                showfliers=False,
                **kwargs,
            )
            ax1.scatter(
                np.random.normal(x1_box, 0.03, size=len(y1)),
                y1,
                color=scatter_color,
                alpha=point_alpha,
                s=10,
                zorder=3,
                **kwargs,
            )
        if not y2.empty:
            ax2.boxplot(
                y2,
                positions=[x2_box],
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=box2_color, alpha=0.6),
                medianprops=dict(color=median_color),
                showfliers=False,
                **kwargs,
            )
            ax2.scatter(
                np.random.normal(x2_box, 0.03, size=len(y2)),
                y2,
                color=scatter_color,
                alpha=point_alpha,
                s=10,
                zorder=3,
                **kwargs,
            )
    ax1.set_xticks(x_base)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel(ylabel1, color=tick1_color)
    ax2.set_ylabel(ylabel2, color=tick2_color)
    ax1.tick_params(axis='y', labelcolor=tick1_color)
    ax2.tick_params(axis='y', labelcolor=tick2_color)
    ax1.set_xlabel(feature_key)
    ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout()
    if return_fig:
        return fig
    elif show_plot:
        plt.show()
        return None
    else:
        return None
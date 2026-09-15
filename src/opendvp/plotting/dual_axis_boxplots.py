from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def dual_axis_boxplots(
    adata: AnnData,
    feature_1: str,
    feature_2: str,
    group_by: str | None = None,
    ylabel1: str | None = None,
    ylabel2: str | None = None,
    return_fig: bool = False,
    ax: Axes | None = None,
    **kwargs: Mapping[str, Any],
) -> Figure | None:
    """Generates a dual-axis plot with boxplots and stripplots for two features.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object's observation metadata (adata.obs) is used.
    feature_1 : str
        Column name in `adata.obs` for the first feature to plot on the left y-axis.
    feature_2 : str
        Column name in `adata.obs` for the second feature to plot on the right y-axis.
    group_by : str, optional
        Column name in `adata.obs` to group by. If None, plots features for all samples.
    ylabel1 : str, optional
        Label for the left y-axis. If None, `feature_1` is used.
    ylabel2 : str, optional
        Label for the right y-axis. If None, `feature_2` is used.
    return_fig : bool
        If True, returns the matplotlib Figure object for further customization.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, new axes are created.
    **kwargs
        Additional keyword arguments passed to matplotlib boxplot/scatter.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    # Validate features
    required_cols = [feature_1, feature_2]
    if group_by:
        required_cols.append(group_by)
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs.")

    # plotting logic
    internal_ax = ax is None
    if internal_ax:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax2 = ax.twinx()
    df = adata.obs.copy()

    # Style settings
    offset = 0.2
    width = 0.3
    point_alpha = 0.3
    box1_color = "skyblue"
    box2_color = "lightcoral"
    median_color = "black"
    scatter_color = "black"
    tick1_color = "blue"
    tick2_color = "red"

    if group_by:
        groups = df[group_by].unique()
        try:
            groups = sorted(groups)
        except TypeError:
            pass
        x_base = np.arange(len(groups))
        group_to_x = {group: i for i, group in enumerate(groups)}

        for group in groups:
            x_pos = group_to_x[group]
            x1_box = x_pos - offset
            x2_box = x_pos + offset

            y1 = df[df[group_by] == group][feature_1].dropna()
            y2 = df[df[group_by] == group][feature_2].dropna()

            if not y1.empty:
                ax.boxplot(
                    y1,
                    positions=[x1_box],
                    widths=width,
                    patch_artist=True,
                    boxprops={"facecolor": box1_color, "alpha": 0.6},
                    medianprops={"color": median_color},
                    showfliers=False,
                    **kwargs,
                )
                ax.scatter(
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
                    boxprops={"facecolor": box2_color, "alpha": 0.6},
                    medianprops={"color": median_color},
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
        ax.set_xticks(x_base)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_xlabel(group_by)
    else:
        # No grouping
        y1 = df[feature_1].dropna()
        y2 = df[feature_2].dropna()
        if not y1.empty:
            ax.boxplot(
                y1,
                positions=[-offset],
                widths=width,
                patch_artist=True,
                boxprops={"facecolor": box1_color, "alpha": 0.6},
                medianprops={"color": median_color},
                showfliers=False,
                **kwargs,
            )
            ax.scatter(
                np.random.normal(-offset, 0.03, size=len(y1)),
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
                positions=[offset],
                widths=width,
                patch_artist=True,
                boxprops={"facecolor": box2_color, "alpha": 0.6},
                medianprops={"color": median_color},
                showfliers=False,
                **kwargs,
            )
            ax2.scatter(
                np.random.normal(offset, 0.03, size=len(y2)),
                y2,
                color=scatter_color,
                alpha=point_alpha,
                s=10,
                zorder=3,
                **kwargs,
            )
        ax.set_xticks([])
        ax.set_xlabel("Features")

    ax.set_ylabel(ylabel1 or feature_1, color=tick1_color)
    ax2.set_ylabel(ylabel2 or feature_2, color=tick2_color)
    ax.tick_params(axis="y", labelcolor=tick1_color)
    ax2.tick_params(axis="y", labelcolor=tick2_color)

    ax.grid(False)
    ax2.grid(False)
    fig.tight_layout()

    if return_fig:
        return fig
    elif internal_ax:
        plt.show()

    return None

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_heatmap(
    adata: AnnData,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "spearman",
    sample_label: str | None = None,
    color_map: str = "magma",
    vmin: float = 0.7,
    vmax: float = 1.0,
    return_fig: bool = False,
    ax: Any | None = None,
) -> Figure | None:
    """Plot a clustered correlation heatmap of protein abundance for all samples in an AnnData object.

    This function computes the pairwise correlation matrix between features (columns) in the AnnData object,
    then clusters and reorders the matrix using hierarchical clustering (scipy), so that similar samples/features
    are grouped together. The heatmap is plotted with annotated values in the top-right triangle and colors only
    in the bottom-left triangle for clarity.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    correlation_method : {"pearson", "kendall", "spearman"}, optional
        Method to calculate the correlation (default = "spearman").
    sample_label : str, optional
        Column name in adata.obs to label samples with. If None, uses adata.obs_names.
    color_map : str, optional
        Colormap for the heatmap (default = "magma").
    vmin : float, optional
        Minimum value for colormap scaling (default = 0.7).
    vmax : float, optional
        Maximum value for colormap scaling (default = 1.0).
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot and returns None.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.

    Notes
    -----
    - The correlation matrix is clustered and reordered using hierarchical clustering (average linkage) on 1 - correlation distance.
    - Both rows and columns are reordered to group similar features/samples together.
    """
    # plotting logic
    internal_ax = ax is None
    if internal_ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # data shaping
    if sample_label is None:
        data = pd.DataFrame(index=adata.obs_names, data=np.asarray(adata.X), columns=adata.var_names)
    else:
        data = pd.DataFrame(index=adata.obs[sample_label], data=np.asarray(adata.X), columns=adata.var_names)
    data = data.transpose()
    correlation_matrix = data.corr(method=correlation_method)

    # cluster rows and columns
    distance_matrix = 1 - correlation_matrix
    condensed_dist = squareform(distance_matrix.values)
    linkage_matrix = linkage(condensed_dist, method="average")
    order = leaves_list(linkage_matrix)
    ordered_corr = correlation_matrix.iloc[order, :].iloc[:, order]

    # plot heatmap
    mask_bottom_left = np.triu(np.ones_like(ordered_corr, dtype=bool), k=1)
    mask_top_right = np.tril(np.ones_like(ordered_corr, dtype=bool))

    sns.heatmap(
        ordered_corr,
        annot=True,
        annot_kws={"size": 8},
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        fmt=".2f",
        linewidths=0.5,
        mask=mask_top_right,
        square=True,
        cbar=True,
        ax=ax,
    )
    sns.heatmap(
        ordered_corr,
        annot=False,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        mask=mask_bottom_left,
        square=True,
        cbar=False,
        ax=ax,
    )

    fig.suptitle(f"{correlation_method} correlation heatmap", fontsize=14)

    # plotting logic
    if return_fig:
        return fig
    elif internal_ax:
        plt.show()
    return None

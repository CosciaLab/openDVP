from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def correlation_heatmap(
    adata: AnnData,
    correlation_method: str = "spearman",
    title: str = "Spearman Correlation Heatmap",
    sample_label: str = "raw_file_id",
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs,
) -> Figure | None:
    """Plot a correlation heatmap of the protein abundance for all samples in adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    correlation_method : str, optional
        Method to calculate the correlation (default = "spearman").
    title : str, optional
        Title of the plot.
    sample_label : str, optional
        Column name in adata.obs to label samples with.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to seaborn.heatmap.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    data = pd.DataFrame(index=adata.obs[sample_label], data=np.asarray(adata.X), columns=adata.var.Genes)
    data = data.transpose()
    allowed_methods = ["pearson", "kendall", "spearman"]
    if correlation_method not in allowed_methods:
        raise ValueError(f"correlation_method must be one of {allowed_methods}")
    # Explicitly cast to Literal for type checkers
    method_literal = cast(Literal["pearson", "kendall", "spearman"], correlation_method)
    correlation_matrix = data.corr(method=method_literal)
    mask_bottom_left = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    mask_top_right = np.tril(np.ones_like(correlation_matrix, dtype=bool))
    if ax is None:
        fig, ax = plt.subplots(figsize=((0.55 * data.shape[1]), (0.4 * data.shape[1])))
    else:
        fig = ax.figure
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='magma',
        fmt='.2f',
        linewidths=0.5,
        vmin=0.7,
        mask=mask_top_right,
        square=True,
        annot_kws={"size": 8},
        ax=ax,
        **kwargs,
    )
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap='magma',
        fmt='.2f',
        linewidths=0.5,
        vmin=0.7,
        mask=mask_bottom_left,
        square=True,
        cbar=False,
        ax=ax,
        **kwargs,
    )
    if return_fig:
        return fig
    else:
        plt.show()
        return None
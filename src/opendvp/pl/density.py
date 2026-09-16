from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def density(
    adata: AnnData,
    color_by: str,
    color_dict: dict | None = None,
    return_fig: bool = False,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Figure | None:
    """Plot density (KDE) plots of protein abundance grouped by a categorical variable in AnnData.obs.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    color_by : str
        Column in adata.obs to group/hue by.
    color_dict : dict, optional
        Dictionary mapping group names (values in adata.obs[color_by]) to colors.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to seaborn.kdeplot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    # plotting logic
    internal_ax = ax is None
    if internal_ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    adata_copy = adata.copy()

    if color_by not in adata_copy.obs.columns:
        raise ValueError(f"{color_by} not found in adata.obs")

    X = np.asarray(adata_copy.X)
    df_kde = pd.DataFrame(data=X, columns=adata_copy.var_names, index=adata_copy.obs[color_by])
    df_kde = df_kde.reset_index()
    df_kde = pd.melt(df_kde, id_vars=[color_by], var_name="Protein", value_name="Abundance")

    if color_dict is None:
        unique_vals = df_kde[color_by].unique()
        tab10 = plt.get_cmap("tab10")
        color_dict = {g: tab10(i % 10) for i, g in enumerate(unique_vals)}

    if not set(color_dict.keys()) == set(adata_copy.obs[color_by].unique()):
        raise ValueError(f"color_dict keys does not match {color_by} unique columns values")

    sns.kdeplot(
        data=df_kde,
        x="Abundance",
        hue=color_by,
        multiple="layer",
        common_norm=False,
        ax=ax,
        palette=color_dict,
        **kwargs,
    )

    ax.set_title(f"Density plot grouped by {color_by}")
    ax.set_xlabel("Abundance")
    ax.set_ylabel("Density")
    ax.grid(False)

    # plotting logic
    if return_fig:
        return fig
    elif internal_ax:
        plt.show()
    return None

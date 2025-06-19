from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def density(
    adata: AnnData,
    color_by: str,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot density (KDE) plots of protein abundance grouped by a categorical variable in AnnData.obs.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color_by : str
        Column in adata.obs to group/hue by.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to seaborn.kdeplot.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    adata_copy = adata.copy()
    if color_by not in adata_copy.obs.columns:
        raise ValueError(f"{color_by} not found in adata.obs")

    X = np.asarray(adata_copy.X)
    df = pd.DataFrame(data=X, columns=adata_copy.var_names, index=adata_copy.obs[color_by])
    df = df.reset_index()
    df = pd.melt(df, id_vars=[color_by], var_name="Protein", value_name="Abundance")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sns.kdeplot(data=df, x="Abundance", hue=color_by, multiple="layer", common_norm=False, ax=ax, **kwargs)
    ax.set_title(f"Density plot grouped by {color_by}")
    ax.set_xlabel("Abundance")
    ax.set_ylabel("Density")
    ax.grid(False)

    if return_fig:
        return fig
    else:
        plt.show()
        return None
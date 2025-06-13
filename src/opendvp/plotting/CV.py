from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def coefficient_of_variation(
    adata: AnnData,
    group_by: str,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot coefficient of variation (CV) for each group in AnnData.obs[group_by].

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_by : str
        Column in adata.obs to group by.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to seaborn.boxplot.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    adata_copy = adata.copy()
    if group_by not in adata_copy.obs.columns:
        raise ValueError(f"{group_by} not found in adata.obs")

    df_tmp = pd.DataFrame()
    for group in adata_copy.obs[group_by].unique():
        adata_group = adata_copy[adata_copy.obs[group_by] == group].copy()
        if adata_group.shape[0] < 3:
            print(f"{group} in dataset has less than 3 samples, leading to poor statistics")
        X_group = np.asarray(adata_group.X)
        means = np.mean(X_group, axis=0)
        stds = np.std(X_group, axis=0)
        cvs = stds / means
        adata_copy.var[f"{group}_mean"] = means
        adata_copy.var[f"{group}_std"] = stds
        adata_copy.var[f"{group}_cv"] = cvs
        group_df = pd.DataFrame({f"{group}_cv": cvs, group_by: group})
        df_tmp = pd.concat([df_tmp, group_df], ignore_index=True)

    df_tmp = df_tmp.melt(id_vars=[group_by], var_name='metric', value_name='cv')
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    sns.boxplot(data=df_tmp, y="cv", hue=group_by, width=0.3, ax=ax, **kwargs)
    ax.set_title(f"Coefficient of Variation by {group_by}")
    ax.set_ylabel("CV (std/mean)")
    ax.set_xlabel(group_by)
    ax.grid(False)
    if return_fig:
        return fig
    else:
        plt.show()
        return None
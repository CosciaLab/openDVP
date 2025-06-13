from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.figure import Figure


def pca_loadings(
    adata: AnnData,
    top: int = 30,
    n_pcs: int = 2,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot PCA protein loadings for the top features in the first two principal components.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA results in adata.varm['PCs'] and adata.uns['pca']['variance_ratio'].
    top : int, optional
        Number of top features to label per PC.
    n_pcs : int, optional
        Number of principal components to plot (default 2).
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to matplotlib scatter.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    PCs = np.asarray(adata.varm['PCs'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    all_top_indices = []
    for i in range(n_pcs):
        PC = PCs[:, i]
        indices = np.argsort(np.abs(PC))[::-1]
        top_indices = indices[:top]
        all_top_indices.append(top_indices.tolist())
    flattened_list = np.concatenate(all_top_indices).tolist()
    ax.scatter(PCs[:, 0], PCs[:, 1], c="b", s=7, **kwargs)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.4, linestyle='--')
    x = PCs[flattened_list, 0]
    y = PCs[flattened_list, 1]
    ax.set_xlabel(f"PC1 {np.round(adata.uns['pca']['variance_ratio'][0]*100, 2)} %")
    ax.set_ylabel(f"PC2 {np.round(adata.uns['pca']['variance_ratio'][1]*100, 2)} %")
    genenames = adata.var.iloc[flattened_list]['Genes'].values
    ax.scatter(x, y, s=12, c="r", **kwargs)
    texts = []
    for i, label in enumerate(genenames):
        text = ax.text(x[i], y[i], label, fontsize=8)
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black'))
    ax.set(xticklabels=[], yticklabels=[])
    ax.grid(False)
    if return_fig:
        return fig
    else:
        plt.show()
        return None
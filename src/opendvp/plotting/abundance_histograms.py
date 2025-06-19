import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def abundance_histograms(
    adata: AnnData,
    n_cols: int = 4,
    return_fig: bool = False,
    **kwargs : dict,
) -> Figure | None:
    """Plot histograms of protein abundance for each sample in adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_cols : int, optional
        Number of columns for the subplots (default = 4).
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    **kwargs
        Additional keyword arguments passed to seaborn.histplot.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    X = np.asarray(adata.X)
    n_of_samples = X.shape[0]
    n_rows = int(np.ceil(n_of_samples / n_cols))
    fixed_subplot_size = (5, 5)
    fig_width = fixed_subplot_size[0] * n_cols
    fig_height = fixed_subplot_size[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.arange(0, 25, 1)
    for i in range(n_of_samples):
        ax = axes[i]
        x = X[i]
        sns.histplot(x, bins=bins, ax=ax, kde=True, **kwargs)
        ax.set_box_aspect(1)
        ax.set_xlim(5, 25)
        res = scipy.stats.shapiro(x)
        ax.text(
            x=0.80,
            y=0.86,
            s=f"Schapiro p: {res[1]:.3g}",
            transform=ax.transAxes,
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8),
        )
        ax.set_title(f'file_id: {adata.obs.raw_file_id[i]}')
        ax.grid(False)
    fig.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()
        return None
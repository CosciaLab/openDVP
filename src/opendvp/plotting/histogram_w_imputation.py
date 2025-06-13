
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def histogram_w_imputation(
    adata_before_imputation: AnnData,
    adata_after_imputation: AnnData,
    n_cols: int = 4,
    return_fig: bool = False,
    save: bool = False,
    save_name: str | None = None,
    **kwargs
) -> Figure | None:
    """Plot histograms for each sample showing raw and imputed values before and after imputation.

    Parameters
    ----------
    adata_before_imputation : AnnData
        AnnData object with the raw data.
    adata_after_imputation : AnnData
        AnnData object with the imputed data.
    n_cols : int, optional
        Number of columns for the subplot grid.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    save : bool, optional
        If True, saves the figure to file.
    save_name : str, optional
        Name of the file to save the figure.
    **kwargs
        Additional keyword arguments passed to seaborn.histplot.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    adata1 = adata_before_imputation.copy()
    adata2 = adata_after_imputation.copy()
    raw_data = np.asarray(adata1.X)
    imputed_data = np.asarray(adata2.X)
    n_samples = adata1.shape[0]
    n_rows = int(np.ceil(n_samples / n_cols))
    fixed_subplot_size = (5, 5)
    fig_width = fixed_subplot_size[0] * n_cols
    fig_height = fixed_subplot_size[1] * n_rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.arange(5, 25.5, 0.5)
    
    for i in range(n_samples):
        row_raw = raw_data[i, :]
        row_imputed = imputed_data[i, :]
        ax = axes[i]
        row_raw_for_plot = row_raw[~np.isnan(row_raw)]
        sns.histplot(row_raw_for_plot, bins=bins, color='blue', label='Raw Data', ax=ax, kde=True, **kwargs)
        imputed_data_only = row_imputed[~np.isin(row_imputed, row_raw)]
        sns.histplot(imputed_data_only, bins=bins, color='red', label='Imputed Data', ax=ax, kde=True, **kwargs)
        ax.set_box_aspect(1)
        ax.set_xlim(5, 25)
        ax.grid(False)
        ax.set_title(f'Histogram for {adata2.obs.raw_file_id[i]}')
        ax.set_xlabel('Log2 Quantified Protein Abundance')
        ax.set_ylabel('Protein hits')
        ax.legend()
    fig.tight_layout()
    fig.suptitle("Gaussian Imputation (per protein) for each sample", fontsize=30, y=1.015)
    
    if save and save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()
        return None
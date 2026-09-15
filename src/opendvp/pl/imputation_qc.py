import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def imputation_qc(
    adata: AnnData,
    unimputed_layer: str = "unimputed",
    return_fig: bool = False,
    ax: Axes | None = None,
    highlight_genes: list[str] | None = None,
    show_highlighted_genes_names: bool = True,
) -> Figure | None:
    """Generate a quality control plot to visualize the effect of imputation.

    This function creates a scatter plot comparing the mean expression of genes
    before and after imputation. The x-axis represents the mean of the raw (unimputed)
    data, and the y-axis shows the difference between the raw mean and the imputed
    mean. This helps to identify genes that were most affected by the imputation
    process. The plot also includes a 2D histogram and kernel density estimate to
    visualize the distribution of data points.

    Parameters
    ----------
    adata
        An AnnData object containing the imputed data in `adata.X` and the
        unimputed data in a specified layer.
    unimputed_layer
        The name of the layer in `adata.layers` that contains the unimputed data.
        Defaults to "unimputed".
    return_fig
        If True, returns the matplotlib Figure object. Defaults to False.
    ax
        A matplotlib Axes object to plot on. If None, a new figure and axes
        are created. Defaults to None.
    highlight_genes
        A list of gene names to highlight in the plot. These genes will be
        plotted in a different color. Defaults to None.
    show_highlighted_genes_names
        If True, displays the names of the highlighted genes on the plot.
        Defaults to True.

    Returns
    -------
    A matplotlib Figure object if `return_fig` is True, otherwise None.

    Examples
    --------
    >>> import anndata
    >>> import numpy as np
    >>> import pandas as pd
    >>> from opendvp.pl import imputation_qc

    >>> # Create a dummy AnnData object
    >>> n_obs, n_vars = 100, 50
    >>> X_imputed = np.random.rand(n_obs, n_vars)
    >>> X_raw = X_imputed.copy()
    >>> # Introduce some NaNs to simulate unimputed data
    >>> X_raw[np.random.choice([True, False], size=X_raw.shape, p=[0.1, 0.9])] = np.nan
    >>> adata = anndata.AnnData(X_imputed)
    >>> adata.layers["unimputed"] = X_raw
    >>> adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    >>> adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    >>> # Generate the QC plot
    >>> imputation_qc(adata, highlight_genes=["Gene_5", "Gene_10"])
    """
    # plotting logic
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        internal_ax = True
    else:
        fig = ax.figure
        internal_ax = False

    adata_copy = adata.copy()

    df_imputed = adata_copy.to_df()
    df_raw = pd.DataFrame(
        data=adata_copy.layers[unimputed_layer], index=adata_copy.obs_names, columns=adata_copy.var_names
    )

    imp_mean = df_imputed.mean(axis=0)
    raw_mean = df_raw.mean(skipna=True, axis=0)
    df_sns = pd.DataFrame({"imp_mean": imp_mean, "raw_mean": raw_mean})
    df_sns["diff"] = df_sns.raw_mean - df_sns.imp_mean

    sns.scatterplot(data=df_sns, x="raw_mean", y="diff", s=10, color="black", ax=ax)
    sns.histplot(data=df_sns, x="raw_mean", y="diff", bins=30, pthresh=0.15, cmap="mako", ax=ax)
    sns.kdeplot(data=df_sns, x="raw_mean", y="diff", levels=4, color="w", linewidths=1, ax=ax)

    if highlight_genes is not None:
        highlight_genes_found = [g for g in highlight_genes if g in df_sns.index]

        if len(highlight_genes_found) > 0:
            highlight_df = df_sns.loc[highlight_genes_found]
            sns.scatterplot(data=highlight_df, x="raw_mean", y="diff", s=30, color="red", ax=ax)

            if show_highlighted_genes_names:
                texts_highlight = [
                    ax.text(
                        row["raw_mean"],
                        row["diff"],
                        idx,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="gold",
                    )
                    for idx, row in highlight_df.iterrows()
                ]
                adjust_text(
                    texts_highlight,
                    ax=ax,
                    expand_points=(1.2, 1.2),
                    arrowprops={"arrowstyle": "-", "color": "gold", "lw": 0.6, "alpha": 0.5},
                )

    # plotting logic
    if return_fig:
        return fig
    elif internal_ax:
        plt.show()
    return None

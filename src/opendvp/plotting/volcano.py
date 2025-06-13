from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.figure import Figure


def volcano(
    adata: AnnData,
    x: str = "log2_FC",
    y: str = "-log10(p_val_corr)_BH",
    significant: bool = True,
    FDR: float | None = None,
    tag_top: int | None = None,
    group1: str | None = None,
    group2: str | None = None,
    return_fig: bool = False,
    highlight_genes: list[str] | None = None,
    show_highlighted_genes_names: bool = True,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot a volcano plot from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        An AnnData object with statistical test results stored in `adata.var`.
    x : str
        Column name in `adata.var` for the x-axis (e.g., log2 fold change).
    y : str
        Column name in `adata.var` for the y-axis (e.g., -log10 p-value).
    significant : bool
        Whether to highlight significant points based on FDR threshold.
    FDR : float or None
        Threshold for corrected p-value (e.g., 0.05). Required if `significant=True`.
    tag_top : int or None
        Number of top hits (by `y`) to label with text, regardless of significance.
    group1 : str or None
        Name of the first group (used for x-axis label annotation).
    group2 : str or None
        Name of the second group (used for x-axis label annotation).
    return_fig : bool
        If True, returns the matplotlib `fig` object for further modification.
    highlight_genes : list or None
        List of gene names/IDs to highlight and label on the plot, if present in adata.var.index.
    show_highlighted_genes_names : bool
        If True, shows names of highlighted genes on the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to matplotlib scatter.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The `fig` object if `return_fig=True`, otherwise None.
    """
    adata_copy = adata.copy()
    df = adata_copy.var
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    sns.scatterplot(x=df[x], y=df[y], ax=ax, color='gray', s=20, edgecolor=None, **kwargs)
    if group1 is not None and group2 is not None:
        ax.set_xlabel(f"Difference in mean protein expression (log2)\n{group1} (right) vs {group2} (left)")
    else:
        ax.set_xlabel(x)
    ax.set_ylabel("-log10 corrected p-value BH")
    if significant:
        if FDR is None:
            raise ValueError("FDR must be specified if significant=True.")
        sig_df = df[df["p_val_corr_BH"] < FDR]
        ax.scatter(sig_df[x], sig_df[y], color="red", s=20)
    if tag_top:
        tag_df = df.sort_values(by=y, ascending=False).head(tag_top)
        df_left = tag_df[tag_df[x] < 0]
        df_right = tag_df[tag_df[x] > 0]
        texts_left = [
            ax.text(row[x], row[y], idx, ha='right', va='center', fontsize=8)
            for idx, row in df_left.iterrows()
        ]
        texts_right = [
            ax.text(row[x], row[y], idx, ha='left', va='center', fontsize=8)
            for idx, row in df_right.iterrows()
        ]
        adjust_text(
            texts_left + texts_right,
            ax=ax,
            expand_points=(1.2, 1.2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=0.5)
        )
    if highlight_genes is not None:
        highlight_genes_found = [g for g in highlight_genes if g in df.index]
        if highlight_genes_found:
            highlight_df = df.loc[highlight_genes_found]
            ax.scatter(highlight_df[x], highlight_df[y], color="blue", s=20, edgecolor='black', zorder=5, alpha=0.7)
            if show_highlighted_genes_names:
                texts_highlight = [
                    ax.text(row[x], row[y], idx, ha='center', va='bottom', fontsize=9, fontweight='bold', color='blue')
                    for idx, row in highlight_df.iterrows()
                ]
                adjust_text(
                    texts_highlight,
                    ax=ax,
                    expand_points=(1.2, 1.2),
                    arrowprops=dict(arrowstyle="-", color='blue', lw=0.5, alpha=0.5)
                )
    ax.grid(False)
    if return_fig:
        return fig
    else:
        plt.show()
        return None

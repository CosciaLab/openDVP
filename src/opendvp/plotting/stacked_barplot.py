import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def stacked_barplot(
    df: pd.DataFrame,
    phenotype_col: str,
    rcn_col: str,
    phenotype_colors: dict[str, str],
    normalize: bool = True,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
    **bar_kwargs: dict
) -> tuple[Figure, Axes]:
    """Plot a stacked barplot showing phenotype composition per RCN motif.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing phenotype and RCN columns.
    phenotype_col : str
        Column name for phenotypes.
    rcn_col : str
        Column name for RCN motifs.
    phenotype_colors : dict
        Dictionary mapping phenotype names to colors.
    normalize : bool, optional
        If True, normalize frequencies to proportions per motif (default: True).
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes to plot on. If None, a new figure and axes are created.
    figsize : tuple of int, optional
        Figure size if creating a new figure (default: (8, 6)).
    **bar_kwargs : dict
        Additional keyword arguments passed to `ax.bar`.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    # Count frequencies of each phenotype within each RCN
    count_df = df.groupby([rcn_col, phenotype_col]).size().unstack(fill_value=0)

    # Normalize to proportions if requested
    if normalize:
        count_df = count_df.div(count_df.sum(axis=1), axis=0)

    # Create the stacked barplot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bottoms = [0] * len(count_df)
    for phenotype, color in phenotype_colors.items():
        if phenotype in count_df.columns:
            ax.bar(
                count_df.index,
                count_df[phenotype],
                bottom=bottoms,
                color=color,
                label=phenotype,
                **bar_kwargs
            )
            bottoms = [i + j for i, j in zip(bottoms, count_df[phenotype], strict=False)]

    # Customize plot
    ax.legend(title="Phenotype", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_xlabel("RCN Motif")
    ax.set_title("Phenotype Composition per RCN Motif")
    plt.tight_layout()
    return fig, ax
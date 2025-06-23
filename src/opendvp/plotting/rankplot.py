from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.figure import Figure

#TODO Assume all groups if None provided

def rankplot(
    adata: AnnData,
    adata_obs_key: str,
    groups: list[str],
    proteins_to_label: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_offset: dict[str, float] | None = None,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot a rank plot of average protein abundance in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    adata_obs_key : str
        Key in adata.obs indicating group labels.
    groups : list of str
        Groups from adata.obs[adata_obs_key] to include.
    proteins_to_label : list of str, optional
        List of feature names (in adata.var_names) to label on the plot.
    group_colors : dict, optional
        Dictionary mapping group names to colors.
    group_offset : dict, optional
        Dictionary mapping group names to x-axis offset (for shifting lines).
    return_fig : bool, optional
        If True, returns the matplotlib figure object for further customization.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to matplotlib plot.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    texts = []

    for group in groups:
        color = group_colors[group] if group_colors and group in group_colors else None
        offset = group_offset[group] if group_offset and group in group_offset else 0

        # Use numpy array for boolean mask
        group_mask = np.asarray(adata.obs[adata_obs_key] == group)
        X_group = np.asarray(adata.X)[group_mask, :]
        mean_vals = X_group.mean(axis=0)
        ranks = pd.Series(mean_vals, index=adata.var_names).rank(ascending=False, method='min').astype(int)

        rank_col_name = f"ranking_{group}" if len(groups) > 1 else "ranking_all"
        adata.var[rank_col_name] = ranks

        df_plot = pd.DataFrame({
            'rank': ranks,
            'mean': mean_vals,
            'protein': adata.var_names
        }).sort_values('rank')

        # Apply offset to x-axis values
        df_plot['offset_rank'] = df_plot['rank'] + offset

        alpha = 0.5 if len(groups) > 1 else 1.0
        ax.plot(df_plot['offset_rank'], df_plot['mean'], label=group, alpha=alpha, color=color, **kwargs)

        if proteins_to_label:
            labeled_df = df_plot[df_plot['protein'].isin(proteins_to_label)]
            for _, row in labeled_df.iterrows():
                texts.append(
                    ax.text(
                        row['offset_rank'],
                        row['mean'],
                        row['protein'],
                        fontsize=9,
                        ha='center',
                        bbox=dict(
                            facecolor=color if color else 'white',
                            edgecolor='none',
                            alpha=0.6,
                            boxstyle='round,pad=0.3',
                        ),
                    )
                )

    if proteins_to_label and texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel("Rank (1 = most abundant)")
    ax.set_ylabel("Average abundance")
    ax.set_title("Protein abundance ranking")
    if len(groups) > 1:
        ax.legend(title=adata_obs_key)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()
        return None
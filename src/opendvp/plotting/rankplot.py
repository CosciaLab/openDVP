from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from matplotlib.figure import Figure

from opendvp.utils import logger


def rankplot(
    adata: AnnData,
    adata_obs_key: str,
    min_presence_fraction: float = 0.7,
    groups: list[str] | None = None,
    proteins_to_label: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs,
) -> Figure | None:
    """Plot a rank plot of average protein abundance in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    adata_obs_key : str
        Key in adata.obs indicating group labels.
    groups : list of str
        Groups from adata.obs[adata_obs_key] to include.
    proteins_to_label : list of str, optional
        List of feature names (in adata.var_names) to label on the plot.
    group_colors : dict, optional
        Dictionary mapping group names to colors. Both keys and values should be strings.
    return_fig : bool, optional
        If True, returns the matplotlib figure object for further customization.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    min_presence_fraction : float, optional
        Minimum fraction of non-NaN values required for a feature to be included in ranking (default: 0.7).
    **kwargs
        Additional keyword arguments passed to matplotlib plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    adata_copy = adata.copy()
    unique_groups = adata_copy.obs[adata_obs_key].unique().tolist()
    if groups is None:
        groups = unique_groups
        logger.info(f"no groups passed, using {unique_groups}")
    else:
        missing = [g for g in groups if g not in unique_groups]
        if missing:
            raise ValueError(f"Groups not present in adata.obs['{adata_obs_key}']: {missing}")
    if min_presence_fraction < 0.01 or min_presence_fraction > 1.0:
        raise ValueError("min_presence_fraction should be between 0.01 and 1.0 (inclusive).")
    if group_colors is None:
        tab10 = matplotlib.cm.get_cmap("tab10")
        group_colors = {g: matplotlib.colors.to_hex(tab10(i % 10)) for i, g in enumerate(groups)}

    df_sns = pd.DataFrame(columns=["group", "rank", "mean", "protein"])

    for group in groups:
        # Use numpy array for boolean mask
        group_mask = np.asarray(adata_copy.obs[adata_obs_key] == group)
        X_group = np.asarray(adata_copy.X)[group_mask, :]

        # Filter features by min_presence_fraction
        valid_counts = np.sum(~np.isnan(X_group), axis=0)
        presence_fraction = valid_counts / X_group.shape[0]
        features_to_keep = presence_fraction >= min_presence_fraction
        filtered_var_names = adata_copy.var_names[features_to_keep]
        filtered_X_group = X_group[:, features_to_keep]

        if filtered_X_group.shape[1] == 0:
            logger.warning(f"No features passed the min_presence_fraction filter for group '{group}'. Skipping.")
            continue

        mean_vals = np.nanmean(filtered_X_group, axis=0)
        ranks = pd.Series(mean_vals, index=filtered_var_names).rank(ascending=False, method="min")
        ranks = ranks.astype(int)

        group_df = pd.DataFrame(
            {"group": group, "rank": ranks, "mean": mean_vals, "protein": filtered_var_names}
        ).sort_values("rank")

        df_sns = pd.concat([df_sns, group_df])

    if df_sns.shape[0] < 1:
        raise ValueError("it seems filtering too strict, nothing to plot")

    sns.scatterplot(
        data=df_sns, x="rank", y="mean", hue="group", palette=group_colors, ax=ax, s=40, linewidth=0, **kwargs
    )

    texts = []
    if proteins_to_label:
        labeled_df = df_sns[df_sns["protein"].isin(proteins_to_label)]
        for _, row in labeled_df.iterrows():
            group = row["group"]
            label_color = group_colors[group] if group_colors and group in group_colors else "black"
            texts.append(
                ax.text(
                    row["rank"],
                    row["mean"],
                    row["protein"],
                    fontsize=15,
                    ha="center",
                    color=label_color,
                )
            )
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "->", "color": "black"},
        expand_points=(2, 2),
        expand_text=(1.5, 1.5),
        force_text=0.5,
        only_move={"points": "none", "text": "xy"},
    )

    ax.set_xlabel("Rank (1 = most abundant)")
    ax.set_ylabel("Average abundance")
    ax.set_title("Protein abundance ranking")
    ax.legend(title=adata_obs_key, loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0.1, frameon=True)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()
        return None

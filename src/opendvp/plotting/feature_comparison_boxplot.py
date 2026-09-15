from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from opendvp.utils import logger


def feature_comparison_boxplot(
    adata: AnnData,
    features: list[str],
    group_by: str,
    group1: str | None = None,
    group2: str | None = None,
    zscore: bool = True,
    color_dict: dict[str, str] | None = None,
    return_fig: bool = False,
    ax: Axes | None = None,
    **kwargs: Mapping[str, Any],
) -> Figure | None:
    """Plots boxplots of feature expression from adata.X, comparing two groups.

    Features are sorted by the mean difference between the two groups.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the expression data and observation metadata.
    features : list[str]
        A list of feature names (from `adata.var_names`) to plot.
    group_by : str
        The column name in `adata.obs` that contains the group labels.
    group1 : str, optional
        The name of the first group. If None, inferred from `group_by`.
    group2 : str, optional
        The name of the second group. If None, inferred from `group_by`.
    zscore : bool, optional
        If True, the expression values are converted to Z-scores before plotting and sorting.
        Defaults to True.
    color_dict : dict, optional
        A dictionary mapping group names to colors.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object, by default False.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to `seaborn.boxplot`.

    Returns
    -------
    matplotlib.figure.Figure | None
        The matplotlib Figure object if `return_fig` is True, otherwise None.

    Raises
    ------
    ValueError
        If validation of groups or features fails.
    """
    # --- Validation ---
    if group_by not in adata.obs.columns:
        raise ValueError(f"Grouping variable '{group_by}' not found in adata.obs.")

    if not all(f in adata.var_names for f in features):
        missing = [f for f in features if f not in adata.var_names]
        logger.info(f"{missing} proteins not found in adata.var_names")

    group_categories = adata.obs[group_by].astype("category").cat.categories
    if group1 is None and group2 is None:
        if len(group_categories) != 2:
            raise ValueError(
                f"'group_by' column '{group_by}' has {len(group_categories)} categories. "
                f"Please specify `group1` and `group2` when more than 2 categories exist."
            )
        group1, group2 = group_categories[0], group_categories[1]
    elif group1 is None or group2 is None:
        raise ValueError("Please specify both `group1` and `group2`, or neither.")
    elif group1 not in group_categories or group2 not in group_categories:
        raise ValueError(f"Groups '{group1}' and/or '{group2}' not found in '{group_by}' column.")

    # --- Data Preparation ---
    groups_to_keep = [group1, group2]
    adata_filt = adata[adata.obs[group_by].isin(groups_to_keep), features].copy()

    df = pd.DataFrame(adata_filt.X, columns=features)

    if zscore:
        df[features] = df[features].apply(st.zscore)

    df[group_by] = adata_filt.obs[group_by].values
    df_tidy = df.melt(id_vars=[group_by], var_name="feature", value_name="expression")

    # --- Calculate Mean Difference and Sort (on z-scored data if applicable) ---
    mean_diffs = (
        df_tidy.groupby(["feature", group_by])["expression"]
        .mean()
        .unstack()
        .eval(f"`{group1}` - `{group2}`")
        .sort_values(ascending=False)
    )
    sorted_features = mean_diffs.index.tolist()

    # --- Plotting ---
    internal_ax = ax is None
    if internal_ax:
        fig, ax = plt.subplots(figsize=(len(sorted_features) * 0.5, 6))
    else:
        fig = ax.figure

    sns.boxplot(
        data=df_tidy,
        x="feature",
        y="expression",
        hue=group_by,
        order=sorted_features,
        ax=ax,
        showfliers=False,
        palette=color_dict,
        **kwargs,
    )
    sns.stripplot(
        data=df_tidy,
        x="feature",
        y="expression",
        hue=group_by,
        order=sorted_features,
        ax=ax,
        dodge=True,
        jitter=True,
        size=3,
        alpha=0.7,
        linewidth=0.7,
        palette=color_dict,
    )

    # --- Legend Handling ---
    handles, labels = ax.get_legend_handles_labels()
    # When hue is used, seaborn creates a legend for each plot. We only want one.
    # The number of unique groups determines the split point.
    n_groups = len(df_tidy[group_by].unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title=group_by)

    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("Feature")
    ylabel = "Expression (Z-score)" if zscore else "Expression"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Expression of {len(features)} features: {group1} vs {group2}")
    plt.tight_layout()

    if return_fig:
        return fig
    elif internal_ax:
        plt.show()

    return None

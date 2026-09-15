from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from upsetplot import UpSet, from_indicators


def upset(
    adata: AnnData,
    groupby: str,
    min_presence_fraction: float = 0.7,
    sort_by: str = "cardinality",
    return_fig: bool = False,
    ax: Axes | None = None,
    **kwargs: Mapping[str, Any],
) -> Figure | None:
    """Generate an UpSet plot from an AnnData object based on variable presence across groups.

    Expected input is a non-imputed adata object.

    Variables that are completely NaN across all samples are excluded.
    Variables that do not pass min_presence_fraction of the selected group are removed.
    e.g if 0.5, if only 40% of samples have a valid value that variable is not counted for that group.
    The final UpSet plot shows presence/absence of variables across the specified groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with observations (samples) as rows and variables as columns.
    groupby : str
        Column name in `adata.obs` used to group samples before computing presence.
    min_presence_fraction : float, optional
        Minimum fraction of samples (within a group) where a variable must be present
        for that group to consider the variable as "present". Value between 0.0 and 1.0. Default is 0.7.
    sort_by : str, optional
        Sorter for UpSet plot: cardinality, degree, -cardinality, -degree. Default is "cardinality".
    return_fig : bool, optional
        Whether to return the matplotlib Figure object. Default is False.
    **kwargs
        Additional keyword arguments passed to UpSet.

    Returns
    -------
    matplotlib.figure.Figure or None
        The matplotlib Figure object containing the UpSet plot, or None if not requested.

    Examples
    --------
    >>> upset(adata, groupby="condition", threshold=1000, min_presence_fraction=0.2)
    """
    # TODO, optional removal of proteins not in any group (based on cutoff)
    # plotting logic
    fig = ax.figure if ax is not None else plt.figure()

    # data wraggling
    df = adata.to_df()
    df = df.loc[:, ~df.isna().all()]
    presence = df.notna()

    if groupby not in adata.obs.columns:
        raise ValueError(f"{groupby!r} not found in adata.obs columns.")
    groups = adata.obs[groupby]

    # Aggregate presence per group (variable is "present" if present in ≥ min_fraction samples in that group)
    grouped_presence = (
        presence.groupby(groups, observed=False)
        .agg(lambda x: x.sum() / len(x) >= min_presence_fraction)
        .T  # transpose to have variables as rows, groups as columns
    )

    if grouped_presence.shape[1] == 0:
        raise ValueError("No variables passed the presence filter for any group.")

    # Convert to UpSet input format
    upset_data = from_indicators(grouped_presence)
    upset = UpSet(upset_data, subset_size="count", sort_by=sort_by, **kwargs)
    upset.plot(fig=fig)

    if return_fig:
        return fig
    elif ax is None:
        plt.show()
    return None

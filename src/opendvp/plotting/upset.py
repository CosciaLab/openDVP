import matplotlib.pyplot as plt
from anndata import AnnData
from upsetplot import UpSet, from_indicators


def upset(
    adata: AnnData,
    groupby: str,
    threshold: float = 0.0,
    min_presence_fraction: float = 0.0,
    sort_by: str = "cardinality",
    show: bool = True
) -> plt.Figure:
    """Generate an UpSet plot from an AnnData object based on variable presence across groups.

    Presence is defined as non-NaN and above a specified threshold. Variables that are
    completely NaN across all samples are excluded. The final UpSet plot shows
    presence/absence of variables across the specified groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (samples) as rows and variables as columns.
    groupby : str
        Column name in `adata.obs` used to group samples before computing presence.
    threshold : float, optional
        Minimum value to consider a variable "present" in a sample. Default is 0.0.
    min_presence_fraction : float, optional
        Minimum fraction of samples (within a group) where a variable must be present
        for that group to consider the variable as "present". Value between 0.0 and 1.0. Default is 0.0.
    sort_by : str, optional
        Sorter for UpSet plot: cardinality, degree, -cardinality, -degree
    figsize : tuple, optional
        Size of the UpSet plot figure. Default is (10, 6).
    show : bool, optional
        Whether to call plt.show(). Default is True.

    Returns:
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the UpSet plot.

    Example:
    -------
    >>> plot_upset_from_adata(adata, groupby="condition", threshold=1000, min_presence_fraction=0.2)
    """
    # Convert to DataFrame
    df = adata.to_df()

    # Exclude variables that are all NaN
    df = df.loc[:, ~df.isna().all()]

    # Compute presence/absence per cell: True if not NaN and > threshold
    presence = df.notna() & (df > threshold)

    # Get group labels
    if groupby not in adata.obs.columns:
        raise ValueError(f"{groupby!r} not found in adata.obs columns.")
    groups = adata.obs[groupby]

    # Aggregate presence per group (variable is "present" if present in ≥ min_fraction samples in that group)
    grouped_presence = (
        presence.groupby(groups, observed=False)
        .agg(lambda x: x.sum() / len(x) >= min_presence_fraction)
        .T  # transpose to have variables as rows, groups as columns
    )

    # Convert to UpSet input format
    upset_data = from_indicators(grouped_presence)
    upset = UpSet(upset_data, subset_size='count', sort_by=sort_by)
    axes_dict = upset.plot()
    fig = list(axes_dict.values())[0].figure

    if show:
        plt.tight_layout()
        plt.show()
    
    return fig
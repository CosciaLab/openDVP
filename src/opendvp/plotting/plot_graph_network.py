from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure


def plot_graph_network(
    w: Any,
    coords: Any,
    threshold: float,
    return_fig: bool = False,
    ax: Any | None = None,
    **kwargs
) -> Figure | None:
    """Plot the graph of connected nodes for a given threshold.

    Parameters
    ----------
    w : libpysal.weights.DistanceBand
        The distance band weights object.
    coords : array-like
        The coordinates of the points.
    threshold : float
        The threshold used to create the DistanceBand object.
    return_fig : bool, optional
        If True, returns the matplotlib Figure object for further customization. If False, shows the plot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to networkx.draw.

    Returns:
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    # Create a network graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(len(coords)))

    # Add edges based on the distance threshold
    for i in range(len(coords)):
        for neighbor in w.neighbors[i]:
            G.add_edge(i, neighbor)

    # Plot the graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    pos = {i: (coords[i][0], coords[i][1]) for i in range(len(coords))}
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=30,
        node_color='blue',
        alpha=0.5,
        edge_color='gray',
        width=0.5,
        ax=ax,
        **kwargs,
    )
    ax.set_title(f"Graph of Connected Nodes at Threshold {threshold}")
    if return_fig:
        return fig
    else:
        plt.show()
        return None
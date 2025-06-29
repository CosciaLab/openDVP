import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import DistanceBand

from opendvp.plotting.plot_graph_network import plot_graph_network
from opendvp.utils import logger


def spatial_hyperparameter_search(
    adata: ad.AnnData,
    x_y: list[str] | None = None,
    threshold_range: np.ndarray | None = None,
    return_df: bool = False,
    plot_network_at: int | None = None
) -> pd.DataFrame | tuple:
    """Perform a hyperparameter search over a range of threshold values.
     
    To determine the number of connected nodes and average neighbors for different threshold values, 
    and optionally plot the network of connected nodes at a given threshold.

    Parameters
    ----------
    adata : AnnData
        Spatially indexed data.
    x_y : list of str, default ['x_centroid', 'y_centroid']
        Column names in adata.obs representing the spatial coordinates.
    threshold_range : np.ndarray, default np.arange(1, 100, 1)
        Range of threshold values to test.
    return_df : bool, default False
        If True, return the DataFrame with threshold statistics along with the plot.
    plot_network_at : Optional[int], default None
        The threshold value at which to plot the network of connected nodes. If None, no plot is generated.

    Returns:
    -------
    If return_df is True:
        tuple[pandas.DataFrame, tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]
            DataFrame with threshold statistics and the plot (figure, axes).
    Else:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The plot (figure, axes).
    """
    # Initialize a list to store the stats for each threshold
    if x_y is None:
        x_y = ["x_centroid", "y_centroid"]
    for col in x_y:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
    if threshold_range is None:
        threshold_range = np.arange(1, 100, 1)
    stats = []

    coords = adata.obs[x_y].to_numpy()
    total_nodes = len(coords)

    for threshold in threshold_range:
        # Compute the spatial weights for each threshold
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)

        # Calculate the number of connected nodes (not islands)
        num_connected_nodes = sum([len(w.neighbors[i]) > 0 for i in range(len(coords))])

        # Calculate the average number of neighbors
        avg_neighbors = np.mean([len(w.neighbors[i]) for i in range(len(coords))])

        stats.append({
            'threshold': threshold,
            'num_connected_nodes': num_connected_nodes,
            'avg_neighbors': avg_neighbors
        })

        logger.info(f"Threshold {threshold}, connected nodes: {num_connected_nodes}, avg neighbors: {avg_neighbors}")

        # Optionally plot the graph network for a particular threshold
        if plot_network_at == threshold:
            plot_graph_network(w, coords, threshold)

    # Convert the stats list to a DataFrame
    threshold_stats = pd.DataFrame(stats)

    # Plot the results (connected nodes as percentage)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot percentage of connected nodes on primary axis
    threshold_stats['connected_percentage'] = (threshold_stats['num_connected_nodes'] / total_nodes) * 100
    ax1.plot(threshold_stats['threshold'], threshold_stats['connected_percentage'], 'b-', label='Connected Nodes (%)')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Percentage of Connected Nodes', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot average number of neighbors on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(threshold_stats['threshold'], threshold_stats['avg_neighbors'], 'r-', label='Avg Neighbors')
    ax2.set_ylabel('Average Number of Neighbors', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set title and show plot
    plt.title("Hyperparameter Search: Threshold vs Connected Nodes (%) and Avg Neighbors")
    fig.tight_layout()

    if return_df:
        return threshold_stats, (fig, ax1)
    return (fig, ax1)
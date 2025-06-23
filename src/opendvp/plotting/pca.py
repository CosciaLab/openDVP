from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scanpy as sc
from anndata import AnnData

#TODO ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
# nbformat not in dependencies, considering removing plotly from functions to simplify package

def pca(
    adata: AnnData,
    color: str | None = None,
    group_colors: dict[str, str] | None = None,
    symbol: str | None = None,
    hoverwith: list[str] | None = None,
    choose_PCs: list[int] = [1, 2],
    multi_scatter: bool = False,
    how_many_PCs: int = 4,
    scatter_3d: bool = False,
    save_path: str | None = None,
    return_fig: bool = False,
    **kwargs: Any
) -> go.Figure | None:
    """Plot PCA of samples in an AnnData object using Plotly.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color : str, optional
        Column in adata.obs to color points by.
    group_colors : dict, optional
        Dictionary mapping group names to colors.
    symbol : str, optional
        Column in adata.obs to use for marker symbol.
    hoverwith : list of str, optional
        Columns in adata.obs to show on hover.
    choose_PCs : list of int, optional
        Principal components to plot (1-based index).
    multi_scatter : bool, optional
        If True, plot a scatter matrix of the first how_many_PCs.
    how_many_PCs : int, optional
        Number of PCs to use in scatter matrix.
    scatter_3d : bool, optional
        If True, plot a 3D scatter plot of the first 3 PCs.
    save_path : str, optional
        Path to save the figure as an image or HTML.
    return_fig : bool, optional
        If True, returns the plotly Figure object for further customization. If False, shows the plot.
    **kwargs
        Additional keyword arguments passed to plotly express functions.

    Returns:
    -------
    fig : plotly.graph_objs.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    if 'pca' not in adata.uns or adata.uns['pca'] is None:
        sc.pp.pca(adata, svd_solver='arpack')
        print("PCA was not found in adata.uns['pca']. It was computed now.")
    
    if hoverwith is None:
        hoverwith = ["sampleid"] if "sampleid" in adata.obs.columns else list(adata.obs.columns)
    
    X_pca = np.asarray(adata.obsm['X_pca'])
    df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=adata.obs.index)
    df = pd.concat([df, adata.obs], axis=1)

    if multi_scatter and scatter_3d:
        raise ValueError("Please choose between multi_scatter and scatter_3d. Not both.")

    if multi_scatter:
        features = [f'PC{i+1}' for i in range(how_many_PCs)]
        labels = {str(i): f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(adata.uns['pca']['variance_ratio']*100)}
        fig = px.scatter_matrix(
            df[features], labels=labels, dimensions=range(how_many_PCs), color=df[color] if color else None, symbol=df[symbol] if symbol else None, **kwargs
        )
        fig.update_traces(diagonal_visible=False, marker={'size': 18, 'opacity': 0.8})
        dimension = how_many_PCs * 500
        fig.update_layout(height=dimension, width=dimension, font=dict(size=20, color='black'))
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido')
        if return_fig:
            return fig

    elif scatter_3d:
        features = [f'PC{i+1}' for i in range(3)]
        fig = px.scatter_3d(
            df, x=features[0], y=features[1], z=features[2],
            color=df[color] if color else None,
            symbol=df[symbol] if symbol else None,
            labels={
                features[0]: f'PC1  {adata.uns["pca"]["variance_ratio"][0]*100:.2f}%',
                features[1]: f'PC2  {adata.uns["pca"]["variance_ratio"][1]*100:.2f}%',
                features[2]: f'PC3  {adata.uns["pca"]["variance_ratio"][2]*100:.2f}%'
            },
            **kwargs
        )
        fig.update_layout(width=1000, height=1000)
        if save_path is not None:
            fig.write_html(save_path)
        if return_fig:
            return fig
    else:
        x_pc = f'PC{choose_PCs[0]}'
        y_pc = f'PC{choose_PCs[1]}'
        fig = px.scatter(
            df,
            x=x_pc,
            y=y_pc,
            color=color,
            symbol=symbol,
            hover_data=hoverwith,
            labels={
                x_pc: f'{x_pc} ({adata.uns["pca"]["variance_ratio"][choose_PCs[0]-1]*100:.2f}%)',
                y_pc: f'{y_pc} ({adata.uns["pca"]["variance_ratio"][choose_PCs[1]-1]*100:.2f}%)',
            },
            color_discrete_map=group_colors,
            **kwargs,
        )
        fig.update_layout(
            title=dict(
                text=f"PCA of samples by {color} and {symbol}",
                font=dict(size=24),
                automargin=False,
                yref='paper',
            ),
            font=dict(size=15, color='black'),
            width=1500,
            height=1000,
        )
        fig.update_traces(marker={'size': 15, 'opacity': 0.8})
        if save_path is not None:
            fig.write_image(save_path, engine='kaleido')
        if return_fig:
            return fig

    if not return_fig:
        fig.show()
        return None
    


# def pca(
#         adata : ad.AnnData, 
#         pc_x : int = 1, 
#         pc_y : int = 2,
#         color : str | None = None, 
#         palette : dict | None = None,
#         symbol : str | None = None, 
#         symbol_dict : bool | dict = True,
#         return_fig : bool = False
#         ):
#     """
#     Create a DataFrame for seaborn scatterplot from AnnData object.
    
#     Parameters:
#         adata: AnnData object
#         key1: str, column in adata.obs for color (hue)
#         key2: str, column in adata.obs for marker (style)
#         pc_x: int, 1-based index for x-axis PCA component
#         pc_y: int, 1-based index for y-axis PCA component
        
#     Returns: 
#         Figure | None
#     """
#     # Convert 1-based to 0-based index
#     x_idx = pc_x - 1
#     y_idx = pc_y - 1
    
#     #create dataframe
#     pca_df = pd.DataFrame({
#         f"PC{pc_x}": adata.obsm["X_pca"][:, x_idx],
#         f"PC{pc_y}": adata.obsm["X_pca"][:, y_idx],
#         color: adata.obs[color].values,
#         symbol: adata.obs[symbol].values
#     })

#     fig,ax = plt.subplots()

#     sns.scatterplot(
#         data=pca_df,
#         x=f"PC{pc_x}",
#         y=f"PC{pc_y}",
#         hue=color,
#         palette=palette,
#         style = symbol,
#         markers = symbol_dict,
#     )

#     if return_fig:
#         return fig
#     else:
#         plt.show()
import pandas as pd
import plotly.express as px
from anndata import AnnData
from typing import Optional, List, Dict, Any
import plotly.graph_objs as go

def plot_boxplots_plotly(
    adata: AnnData,
    x_axis: str = "Phenotype_1",
    y_axis: str = "n_proteins",
    hover_data: Optional[List[str]] = None,
    color_column: str = "Phenotype_1",
    return_fig: bool = False,
    save_path: Optional[str] = None,
    save_df_path: Optional[str] = None,
    **kwargs: Any
) -> Optional[go.Figure]:
    """
    Plot interactive boxplots using Plotly for a given AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_axis : str
        Column in adata.obs to use for the x-axis.
    y_axis : str
        Column in adata.obs to use for the y-axis.
    hover_data : list of str, optional
        Columns in adata.obs to show on hover.
    color_column : str
        Column in adata.obs to use for coloring the boxes.
    return_fig : bool, optional
        If True, returns the plotly Figure object for further customization. If False, shows the plot.
    save_path : str, optional
        Path to save the figure as an image (requires kaleido).
    save_df_path : str, optional
        Path to save the DataFrame used for plotting as CSV.
    **kwargs
        Additional keyword arguments passed to plotly.express.box.

    Returns
    -------
    fig : plotly.graph_objs.Figure or None
        The figure object if return_fig is True, otherwise None.
    """
    adata_copy = adata.copy()
    if hover_data is None:
        hover_data = ["Phenotype_2"] if "Phenotype_2" in adata_copy.obs.columns else []

    df = pd.DataFrame(index=adata_copy.obs.index, data=adata_copy.obs.values, columns=adata_copy.obs_keys())

    fig = px.box(
        df, x=x_axis, y=y_axis,
        points='all', hover_data=hover_data,
        color=color_column, width=1000, height=800,
        color_discrete_sequence=px.colors.qualitative.G10,
        **kwargs
    )

    fig.update_layout(
        title=dict(text="Proteins per sample", font=dict(size=30), automargin=True, yref='paper'),
        font=dict(size=18, color='black'),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255, 255, 255)',
        showlegend=True,
    )

    if save_df_path is not None:
        df.to_csv(save_df_path)
    if save_path is not None:
        fig.write_image(save_path, engine='kaleido')

    if return_fig:
        return fig
    else:
        fig.show()
        return None
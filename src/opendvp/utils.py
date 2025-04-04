# Misc functions
import time
import numpy as np
import matplotlib.pyplot as plt


def export_figure(fig, path, suffix):
    datetime = time.strftime("%Y%m%d_%H%M")
    fig.savefig(fname=f"{path}{datetime}_{suffix}.pdf", format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(fname=f"{path}{datetime}_{suffix}.svg", format="svg", dpi=600, bbox_inches="tight")

def check_link(sdata, shape_element_key, adata, adata_obs_key):
    shape_index = sdata[shape_element_key].index.to_list()
    cell_ids = adata.obs[adata_obs_key].to_list()
    assert shape_index[:5] == cell_ids[:5], "First 5 CellIDs do not match."
    assert shape_index[-5:] == cell_ids[-5:], "Last 5 CellIDs do not match."
    assert sdata[shape_element_key].index.dtype == adata.obs[adata_obs_key].dtype, "Data types do not match."
    print("Success, no problems found")


def ensure_one_based_index(adata, cellid_col="CellID"):
    """
    Ensures the specified CellID column and index are 1-based.
    Converts data to integers if needed.
    
    Parameters:
    - adata: AnnData object
    - cellid_col: str, name of the column with cell IDs (default: "CellID")
    
    Returns:
    - adata: updated AnnData object
    """
    
    # Check if the column exists
    if cellid_col not in adata.obs.columns:
        raise ValueError(f"Column '{cellid_col}' not found in adata.obs.")
    
    # Ensure the CellID column and index are integers
    if not np.issubdtype(adata.obs[cellid_col].dtype, np.integer):
        adata.obs[cellid_col] = adata.obs[cellid_col].astype(int)

    if not np.issubdtype(adata.obs.index.dtype, np.integer):
        adata.obs.index = adata.obs.index.astype(int)
    
    # Check if both are 0-based and increment if needed
    if (adata.obs[cellid_col].min() == 0) and (adata.obs.index.min() == 0):
        adata.obs[cellid_col] += 1
        adata.obs.index += 1
        print(f"✅ Incremented '{cellid_col}' and index to 1-based numbering.")
    else:
        print("⏭️ Skipping increment: CellID or index is not 0-based.")
    
    return adata

def plot_rcn_stacked_barplot(df, phenotype_col, rcn_col, normalize=True):
    """
    Plots a stacked barplot showing phenotype composition per RCN motif.
    
    Parameters:
    df (DataFrame): Input dataframe containing phenotype and RCN columns
    phenotype_col (str): Column name for phenotypes
    rcn_col (str): Column name for RCN motifs
    normalize (bool): If True, normalize frequencies to proportions per motif
    """
    # Count frequencies of each phenotype within each RCN
    count_df = df.groupby([rcn_col, phenotype_col]).size().unstack(fill_value=0)
    
    # Normalize to proportions if requested
    if normalize:
        count_df = count_df.div(count_df.sum(axis=1), axis=0)
    
    # Create the stacked barplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottoms = [0] * len(count_df)
    for phenotype, color in phenotype_colors.items():
        if phenotype in count_df.columns:
            ax.bar(count_df.index, count_df[phenotype],
                   bottom=bottoms, color=color, label=phenotype)
            bottoms = [i + j for i, j in zip(bottoms, count_df[phenotype])]
    
    # Customize plot
    ax.legend(title="Phenotype", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_xlabel("RCN Motif")
    ax.set_title("Phenotype Composition per RCN Motif")
    plt.tight_layout()
    plt.show()


def create_vertical_legend(color_dict, title="Legend"):

    fig, ax = plt.subplots(figsize=(3, len(color_dict) * 0.5))
    ax.set_axis_off()

    patches = [
        plt.Line2D([0], [0], marker='o', color=color, markersize=10, label=label, linestyle='None') 
        for label, color in color_dict.items()
    ]
    
    # Draw legend as a vertical list
    legend = ax.legend(
        handles=patches,
        title=title,
        loc='center left',
        frameon=False,
        bbox_to_anchor=(0, 0.5),
        alignment="left"
    )
    
    return fig
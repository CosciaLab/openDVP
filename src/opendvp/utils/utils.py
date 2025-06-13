# Misc functions
import time

import matplotlib.pyplot as plt
import numpy as np


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def switch_adat_var_index(adata, new_index):
    """Switch the index of adata.var to a new index. Useful for switching between gene names and protein names.
    """
    adata_copy = adata.copy()
    adata_copy.var[adata_copy.var.index.name] = adata_copy.var.index
    adata_copy.var.set_index(new_index, inplace=True)
    adata_copy.var.index.name = new_index
    return adata_copy

def check_link(sdata, shape_element_key, adata, adata_obs_key):
    shape_index = sdata[shape_element_key].index.to_list()
    cell_ids = adata.obs[adata_obs_key].to_list()
    assert shape_index[:5] == cell_ids[:5], "First 5 CellIDs do not match."
    assert shape_index[-5:] == cell_ids[-5:], "Last 5 CellIDs do not match."
    assert sdata[shape_element_key].index.dtype == adata.obs[adata_obs_key].dtype, "Data types do not match."
    print("Success, no problems found")

def ensure_one_based_index(adata, cellid_col="CellID"):
    """Ensures the specified CellID column and index are 1-based.
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


def print_color_dict(dictionary):

    fig, ax = plt.subplots(figsize=(8, len(dictionary) * 0.5))

    for index,(name, hex) in enumerate(dictionary.items()):
        ax.add_patch(plt.Rectangle((0, index), 1, 1, color=hex))
        ax.text(1.1, index + 0.5, name, ha='left', va='center', fontsize=12)

    # Adjust plot limits and aesthetics
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(dictionary))
    ax.axis('off')

    plt.show()
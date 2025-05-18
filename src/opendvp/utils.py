# Misc functions
import time, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from loguru import logger
from itertools import cycle

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def switch_adat_var_index(adata, new_index):
    """
    Switch the index of adata.var to a new index. Useful for switching between gene names and protein names.
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


def parse_color_for_qupath(color_dict, adata, adata_obs_key) -> dict:
    
    logger.info("Parsing colors compatible with QuPath")
    
    if color_dict is None: 
        logger.info("No color_dict found, using defaults")
        default_colors = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189]]
        color_cycle = cycle(default_colors)
        parsed_colors = dict(zip(adata.obs[adata_obs_key].cat.categories.astype(str), color_cycle))
        logger.info(f"color_dict created: {parsed_colors}")
    else:
        logger.info("Custom color dictionary passed, adapting to QuPath color format")  
        parsed_colors = {}
        for name, color in color_dict.items():
            if isinstance(color, tuple) and len(color) == 3:
                # Handle RGB fraction tuples (0-1)
                parsed_colors[name] = list(int(c * 255) for c in color)
            elif isinstance(color, list) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                # Already in [R, G, B] format with values 0-255
                parsed_colors[name] = color
            elif isinstance(color, str) and re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
                # Handle hex codes
                parsed_colors[name] = mcolors.hex2color(color)
                parsed_colors[name] = list(int(c * 255) for c in parsed_colors[name])
            else:
                raise ValueError(f"Invalid color format for '{name}': {color}")
            
    return parsed_colors

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
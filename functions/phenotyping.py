from loguru import logger
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scimap as sm


def read_gates(gates_csv_path, sample_id=None) -> pd.DataFrame:
    """ Read the gates data from a csv file and return a dataframe """
    logger.info(" ---- read_gates : version number 1.1.0 ----")
    time_start = time.time()

    assert gates_csv_path.endswith('.csv'), "The file should be a csv file"
    gates = pd.read_csv(gates_csv_path)
    
    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    assert "gate_value" in gates.columns, "The column gate_value is not present in the csv file"
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")
    logger.info(f"  Markers found: {gates.marker_id.unique()}")
    logger.info(f"  Samples found: {gates.sample_id.unique()}")

    if sample_id is not None:
        assert "sample_id" in gates.columns, "The column sample_id is not present in the csv file"
        gates = gates[gates['sample_id']==sample_id]
        logger.info(f"  Found {gates.shape[0]} valid gates for sample {sample_id}")

    logger.info(f" ---- read_gates is done, took {int(time.time() - time_start)}s  ----")
    return gates


def phenotype_with_gate_change(adata, gates, phenotype_matrix, sample_id, marker, new_gate, adata_in_place=False):
    """ Plot the spatial scatter plot with the new gate value """

    logger.info(" ---- phenotype_with_gate_change : version number 1.0.0 ----")

    adata_copy = adata.copy()
    gates_copy = gates.copy()

    gates_copy.loc[gates_copy['marker_id']==marker, 'gate_value'] = new_gate
    gates_copy.loc[gates_copy['marker_id']==marker, 'log1p_gate_value'] = np.log1p(new_gate)
    processed_gates = process_gates_for_sm(gates_copy, sample_id)

    logger.info(f"processed gate {processed_gates.loc[processed_gates['marker']==marker]}")

    # rescale adata
    adata_copy = sm.pp.rescale(adata_copy, gate=processed_gates, log=True, verbose=False)
    adata_copy = sm.tl.phenotype_cells (adata_copy, phenotype=phenotype_matrix, label="phenotype", verbose=False)

    custom_colours = {
        "Cancer_cells" : "red",
        "CD4_Tcells" : "peru",
        "CD8_Tcells" : "lawngreen",
        "Macrophages" : "yellow",
        "COL1A1_cells" : "deepskyblue",
        "Vimentin_cells" : "orange",
        "B_cells" : "black",
        "Unknown" : "whitesmoke"
    }

    sm.pl.spatial_scatterPlot (adata_copy, colorBy = ['phenotype'],figsize=(12,7), s=1, fontsize=10, customColors=custom_colours)
    
    # Get the current figure
    fig = plt.gcf()
    ax = plt.gca()
    plt.table(cellText=gates_copy.values, colLabels=gates_copy.columns, loc='upper right', cellLoc='center')
    plt.show()

    if adata_in_place:
        logger.info(" phenotyping saved in adata object")
        return adata_copy
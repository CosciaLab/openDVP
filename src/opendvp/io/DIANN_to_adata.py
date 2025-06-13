import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger

datetime = time.strftime("%Y%m%d_%H%M%S")

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")


def DIANN_to_adata(
    DIANN_path: str,
    DIANN_sep: str = "\t",
    metadata_path: str | None = None,
    metadata_sep: str = ",",
    metadata_filepath_header: str = "File.Name",
    filter_contamination: bool = True,
    filter_nan_genes: bool = True,
    n_of_protein_metadata_cols: int = 4
) -> ad.AnnData:
    """Converts DIANN output file and metadata file into an AnnData object.

    Parameters
    ----------
    DIANN_path : str
        Path to DIANN output file.
    DIANN_sep : str, default '\t'
        Delimiter for DIANN output file.
    metadata_path : Optional[str], default None
        Path to metadata file.
    metadata_sep : str, default ','
        Delimiter for metadata file.
    metadata_filepath_header : str, default 'File.Name'
        Name of the column in metadata file that contains the DIANN file paths.
    filter_contamination : bool, default True
        If True, removes Protein.Names labelled with 'Cont_' as a prefix.
    filter_nan_genes : bool, default True
        If True, removes variable rows that contain NaN in the 'Genes' column.
    n_of_protein_metadata_cols : int, default 4
        Number of protein metadata columns at the start of the DIANN file.

    Returns:
    -------
    AnnData
        AnnData object with imported data.
    """
    df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    logger.info(f"Starting DIANN matrix shape {df.shape}")
    if filter_contamination:
        condition_cont = df['Protein.Group'].str.contains("Cont_")
        logger.info(f"Removing {df[condition_cont].shape[0]} proteins considered contaminants")
        df = df[~condition_cont]
    if filter_nan_genes:
        condition_na = df['Genes'].isna()
        logger.info(f"Filtering {df[condition_na].shape[0]} genes that are NaN, {df[condition_na]['Protein.Names'].tolist()}")
        df = df[~condition_na]

    ### numerical data ###
    dft= df.T.copy()
    dft.columns = dft.loc["Protein.Group", :].values
    dft.index.name = "Sample_filepath"
    rawdata = dft.iloc[n_of_protein_metadata_cols:,:]
    logger.info(f" Data comprises {rawdata.shape[0]} samples, and {rawdata.shape[1]} proteins ")

    ### protein metadata ###
    protein_metadata = df.iloc[:,:n_of_protein_metadata_cols]
    protein_metadata['Genes_simplified'] = [gene.split(";")[0] for gene in protein_metadata['Genes'].tolist()]
    protein_metadata.set_index('Genes_simplified', inplace=True)
    logger.info(f"{protein_metadata[protein_metadata['Genes'].str.contains(';')].shape[0]} gene lists (eg 'TMA7;TMA7B') were simplified to their first element ('TMA7').")
    protein_metadata.index.name = "Gene" #changed index name to Gene, instead of Genes to avoid conflicts writing the data

    #load sample metadata
    if metadata_path is not None:
        sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep)
        sample_metadata.set_index(metadata_filepath_header, inplace=True)
    else:
        sample_metadata = pd.DataFrame(index=rawdata.index)

    # check sample_metadata filename_paths are unique, and matches df
    if not set(sample_metadata.index) == set(rawdata.index): 
        logger.warning("unique values from sample metadata and DIANN table do not match")
        logger.warning("consider double checking 'n_of_protein_metadata_cols', it varies per DIANN version")
        raise ValueError("uniques don't match")
    
    if not rawdata.shape[0] == sample_metadata.shape[0]:
        logger.error(f"ERROR: Number of samples in DIANN output {rawdata.shape[0]} and metadata {sample_metadata.shape[0]} do not match. Please check your files.")

    # reindex to match rawdata to sample metadata
    sample_metadata_aligned = sample_metadata.reindex(rawdata.index)

    # create adata object
    adata = ad.AnnData(X=rawdata.values.astype(np.float64), obs=sample_metadata_aligned, var=protein_metadata)
    logger.success("Anndata object has been created :) ")
    return adata
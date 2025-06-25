import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


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
    r"""Converts DIANN output file and metadata file into an AnnData object.

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
    diann_df = pd.read_csv(DIANN_path, sep=DIANN_sep)
    logger.info(f"Starting DIANN matrix shape {diann_df.shape}")
    if filter_contamination:
        condition_cont = diann_df['Protein.Group'].str.contains("Cont_")
        logger.info(f"Removing {diann_df[condition_cont].shape[0]} contaminants")
        diann_df = diann_df[~condition_cont]
    if filter_nan_genes:
        condition_na = diann_df['Genes'].isna()
        logger.info(f"Filtering {diann_df[condition_na].shape[0]} genes that are NaN")
        logger.info(f"{diann_df[condition_na]['Protein.Names'].tolist()}")
        diann_df = diann_df[~condition_na]

    ### numerical data ###
    diann_dft= diann_df.T.copy()
    diann_dft.columns = diann_dft.loc["Protein.Group", :].to_numpy()
    diann_dft.index.name = "Sample_filepath"
    rawdata = diann_dft.iloc[n_of_protein_metadata_cols:,:]
    logger.info(f"{rawdata.shape[0]} samples, and {rawdata.shape[1]} proteins")

    ### protein metadata ###
    protein_metadata = diann_df.iloc[:,:n_of_protein_metadata_cols]
    protein_metadata['Genes_simplified'] = [
        gene.split(";")[0]
        for gene in protein_metadata['Genes'].tolist()
    ]
    protein_metadata = protein_metadata.set_index('Genes_simplified')
    n_simple = protein_metadata[protein_metadata['Genes'].str.contains(';')].shape[0]
    logger.info(f"{n_simple} gene lists (eg 'TMA7;TMA7B') were simplified to ('TMA7').")
    protein_metadata.index.name = "Gene"

    #load sample metadata
    if metadata_path is not None:
        sample_metadata = pd.read_csv(metadata_path, sep=metadata_sep)
        sample_metadata = sample_metadata.set_index(metadata_filepath_header)
    else:
        sample_metadata = pd.DataFrame(index=rawdata.index)

    # check sample_metadata filename_paths are unique, and matches df
    if set(sample_metadata.index) != set(rawdata.index): 
        logger.warning("uniques from sample metadata and DIANN table do not match")
        logger.warning("check n_of_protein_metadata_cols, it varies per DIANN version")
        raise ValueError("uniques don't match")
    
    if rawdata.shape[0] != sample_metadata.shape[0]:
        logger.error("Number of samples in DIANN output and metadata do not match")

    # reindex to match rawdata to sample metadata
    sample_metadata_aligned = sample_metadata.reindex(rawdata.index)

    # create adata object
    adata = ad.AnnData(
        X=rawdata.to_numpy().astype(np.float64), 
        obs=sample_metadata_aligned, 
        var=protein_metadata)
    
    if adata.var.index.has_duplicates:
        logger.info("Duplicate genes found from different protein groups")
        logger.info(f"{adata.var.index[adata.var.index.duplicated()].unique().tolist()}")
        logger.info("duplicates will be made unique by adding suffix (eg, -1,-2,-3)")
        adata.var_names_make_unique()

    logger.success("Anndata object has been created :) ")
    return adata
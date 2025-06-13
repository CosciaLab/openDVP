import os
import pandas as pd
import numpy as np
import anndata as ad
from datetime import datetime
from typing import Optional

def adata_to_perseus(
    adata: ad.AnnData,
    path_to_dir: str,
    suffix: str,
    obs_key: Optional[str] = None
) -> None:
    """
    Export an AnnData object to Perseus-compatible text files (expression data and metadata).

    Parameters
    ----------
    adata : AnnData
        AnnData object to export.
    path_to_dir : str
        Directory where output files will be saved.
    suffix : str
        Suffix for output file names.
    obs_key : Optional[str], default None
        Column in adata.obs to use as row index for the data file. If None, uses adata.obs_names.

    Returns
    -------
    None
        This function saves files to disk and does not return a value.
    """
    os.makedirs(path_to_dir, exist_ok=True)  # Ensure output directory exists

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Prepare file paths
    data_file = os.path.join(path_to_dir, f"{timestamp}_data_{suffix}.txt")
    metadata_file = os.path.join(path_to_dir, f"{timestamp}_metadata_{suffix}.txt")

    # Export expression data
    data = np.asarray(adata.X)
    index = adata.obs[obs_key] if obs_key is not None else adata.obs_names
    df = pd.DataFrame(data=data, columns=adata.var_names, index=index)
    df.index.name = "Name"  # Perseus requires this
    df.to_csv(data_file, sep="\t")

    # Export metadata
    metadata = adata.obs.copy()
    metadata.set_index(obs_key, inplace=True)
    metadata.index.name = "Name"
    metadata.to_csv(metadata_file, sep="\t")

    print(f"Success: files saved as\n- {data_file}\n- {metadata_file}")
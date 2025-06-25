import os
from pathlib import Path

import anndata as ad
import pandas as pd

from opendvp.utils import logger
from opendvp.utils.utils import get_datetime


def export_adata(
    adata: ad.AnnData,
    path_to_dir: str,
    checkpoint_name: str,
    export_as_cvs : bool = False,
    metadata_cols : list | None = None,
    metadata_index : str | None = None, 
    parquet : bool = False,
    perseus : bool = False,
) -> None:
    """Save an AnnData object as both .h5ad and optionally .parquet, .csv, or Perseus files in a checkpoint directory.

    Parameters:
    ----------
    adata : AnnData
        AnnData object to save.
    path_to_dir : str
        Directory where the checkpoint folder will be created.
    checkpoint_name : str
        Name for the checkpoint folder and file prefix.
    export_as_cvs : bool, optional
        If True, exports the data and metadata as .csv (tab-delimited) or .parquet files. Default is False.
    metadata_cols : list, optional
        List of columns from adata.obs to include in the metadata file. If None, all columns are included.
    metadata_index : str, optional
        Column from adata.obs to use as index in the exported files. If None, uses the default index.
    parquet : bool, optional
        If True, exports as .parquet files instead of .csv. Only used if export_as_cvs is True.
    perseus : bool, optional
        If True, exports Perseus-compatible files in a subfolder. Default is False.

    Returns:
    -------
    None
        This function saves files to disk and does not return a value.

    Example:
    -------
    >>> from opendvp.io.export_adata import export_adata
    >>> import anndata as ad
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = np.random.rand(10, 5)
    >>> obs = pd.DataFrame({'celltype': ['A']*5 + ['B']*5}, index=[f'cell{i}' for i in range(10)])
    >>> var = pd.DataFrame(index=[f'gene{i}' for i in range(5)])
    >>> adata = ad.AnnData(X=X, obs=obs, var=var)
    >>> export_adata(adata, path_to_dir='checkpoints', checkpoint_name='test', export_as_cvs=True, perseus=True)
    """
    try:    
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_dir,checkpoint_name), exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create folder, permission problem likely: {e}")
        return
    
    basename = f"{os.path.join(path_to_dir,checkpoint_name)}/{get_datetime()}_{checkpoint_name}_adata"

    # Save h5ad file
    try:
        logger.info("Writing h5ad")
        adata.write_h5ad(filename=Path(basename + ".h5ad"))
        logger.success("Wrote h5ad file")
    except (OSError, ValueError) as e:
        logger.error(f"Could not write h5ad file: {e}")
        return

    if export_as_cvs:
        X_array = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X # type: ignore
        df_index = adata.obs[metadata_index] if metadata_index else adata.obs.index
        adata_obs_cols = metadata_cols if metadata_cols else adata.obs.columns.tolist()

        data = pd.DataFrame(data=X_array, columns=adata.var_names, index=df_index) # type: ignore
        metadata = pd.DataFrame(data=adata.obs[adata_obs_cols], index=df_index)

        if parquet:
            logger.info("Writing parquet")
            data.to_parquet(path=Path(basename + "data.parquet"))
            metadata.to_parquet(path=Path(basename + "metadata.parquet"))
            logger.success("Wrote parquet file")
        else:
            logger.info("Writing csvs")
            data.to_csv(path_or_buf=Path(basename + "data.txt"), sep="\t")
            metadata.to_csv(path_or_buf=Path(basename + "metadata.txt"), sep="\t")
            logger.success("Wrote csv files")

    if perseus:
        perseus_dir = os.path.join(path_to_dir, checkpoint_name, "perseus")
        os.makedirs(perseus_dir, exist_ok=True)
        logger.info(f"Exporting Perseus files to {perseus_dir}")
        # --- Begin inlined adata_to_perseus logic ---
        timestamp = get_datetime()
        data_file = os.path.join(perseus_dir, f"{timestamp}_data_{checkpoint_name}.txt")
        metadata_file = os.path.join(perseus_dir, f"{timestamp}_metadata_{checkpoint_name}.txt")
        # Export expression data
        data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X # type: ignore
        index = adata.obs[metadata_index] if metadata_index is not None else adata.obs_names
        expression_df = pd.DataFrame(data=data, columns=adata.var_names, index=index) # type: ignore
        expression_df.index.name = "Name"  # Perseus requires this
        expression_df.to_csv(data_file, sep="\t")
        # Export metadata
        metadata = adata.obs.copy()
        if metadata_index is not None:
            metadata = metadata.set_index(metadata_index)
        metadata.index.name = "Name"
        metadata.to_csv(metadata_file, sep="\t")
        logger.success(f"Wrote Perseus files: {data_file}, {metadata_file}")
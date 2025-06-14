import os
from pathlib import Path

import anndata as ad

from opendvp.utils import logger
from opendvp.utils.utils import get_datetime


def export_adata(
    adata: ad.AnnData,
    path_to_dir: str,
    checkpoint_name: str
) -> None:
    """Save an AnnData object as both .h5ad and .parquet files in a checkpoint directory.

    Parameters
    ----------
    adata : AnnData
        AnnData object to save.
    path_to_dir : str
        Directory where the checkpoint folder will be created.
    checkpoint_name : str
        Name for the checkpoint folder and file prefix.

    Returns:
    -------
    None
        This function saves files to disk and does not return a value.
    """
    try:    
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_dir,checkpoint_name), exist_ok=True)
    except Exception as e:
        logger.error(f"Unexpected error in save_adata_checkpoint: {e}")
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

    # Save CSV file
    try:
        logger.info("Writing parquet")
        adata.to_df().to_parquet(path=Path(basename + ".parquet"))
        logger.success("Wrote parquet file")
    except (OSError, ValueError) as e:
        logger.error(f"Could not write parquet file: {e}")
import os

from opendvp.logger import logger
from opendvp.utils import get_datetime


def save_adata_checkpoint(adata, path_to_dir, checkpoint_name):
    try:    
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(os.path.join(path_to_dir,checkpoint_name), exist_ok=True)
        basename = f"{os.path.join(path_to_dir,checkpoint_name)}/{get_datetime()}_{checkpoint_name}_adata"
    except Exception as e:
        logger.error(f"Unexpected error in save_adata_checkpoint: {e}")
    
    # Save h5ad file
    try:
        logger.info("Writing h5ad")
        adata.write_h5ad(filename = basename + ".h5ad")
        logger.success("Wrote h5ad file")
    except (OSError, ValueError) as e:
            logger.error(f"Could not write h5ad file: {e}")
            return
    
    # Save CSV file
    try:
        logger.info("Writing parquet")
        adata.to_df().to_parquet(path=basename + ".parquet")
        logger.success("Wrote parquet file")
    except (OSError, ValueError) as e:
        logger.error(f"Could not write parquet file: {e}")
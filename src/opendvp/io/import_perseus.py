import anndata as ad
from perseuspy import pd

from opendvp.utils import logger


def import_perseus(
    path_to_perseus_txt: str,
    n_var_metadata_rows: int = 4
) -> ad.AnnData:
    """Convert a Perseus text file to an AnnData object.

    Parameters
    ----------
    path_to_perseus_txt : str
        Path to the Perseus text file.
    n_var_metadata_rows : int, default 4
        Number of metadata rows at the bottom of the columns to use as var headers.

    Returns:
    -------
    AnnData
        AnnData object with imported data.
    """
    logger.info(f"Reading Perseus file from: {path_to_perseus_txt}")
    perseus_df = pd.read_perseus(path_to_perseus_txt) # type: ignore
    logger.info(f"Perseus DataFrame shape: {perseus_df.shape}")
    # get obs headers
    obs_headers = list(perseus_df.columns.names)
    logger.debug(f"Observation headers: {obs_headers}")
    # get obs contents
    obs = [col for col in perseus_df.columns.values] #tuples
    obs = pd.DataFrame(obs)
    logger.debug(f"Observation DataFrame shape before cleaning: {obs.shape}")
    # var headers configurable
    var_headers = obs.iloc[-n_var_metadata_rows:,0].values.tolist()
    logger.debug(f"Variable headers: {var_headers}")
    #remove rows with empty strings
    obs = obs[obs != '']
    obs = obs.dropna()
    logger.debug(f"Observation DataFrame shape after cleaning: {obs.shape}")
    #rename headers
    obs.columns = obs_headers
    #var 
    var = perseus_df[var_headers]
    var.columns = var_headers
    logger.debug(f"Variable DataFrame shape: {var.shape}")
    #get data
    data = perseus_df.iloc[:,:-(len(var_headers))].values.T
    logger.info(f"Data matrix shape: {data.shape}")
    #to prevent implicit modification
    obs.index = obs.index.astype(str)
    var.index = var.index.astype(str)

    try:
        adata = ad.AnnData(X=data, obs=obs, var=var)
    except Exception as e:
        logger.error(
            f"Failed to convert Perseus file to AnnData. Error: {e}\n"
            f"Tip: Double-check the `n_var_metadata_rows` parameter. "
            f"Try lowering or increasing it depending on your file structure."
        )
        raise

    logger.success("AnnData object created from Perseus file.")
    return adata
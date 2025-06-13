import sys
import time

import anndata as ad
from loguru import logger

datetime = time.strftime("%Y%m%d_%H%M%S")

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def perseus_to_anndata(
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
    from perseuspy import pd
    df = pd.read_perseus(path_to_perseus_txt)
    # get obs headers
    obs_headers = list(df.columns.names)
    # get obs contents
    obs = [col for col in df.columns.values] #tuples
    obs = pd.DataFrame(obs)
    # var headers configurable
    var_headers = obs.iloc[-n_var_metadata_rows:,0].values.tolist()
    #remove rows with empty strings
    obs = obs[obs != '']
    obs.dropna(inplace=True)
    #rename headers
    obs.columns = obs_headers
    #var 
    var = df[var_headers]
    var.columns = var_headers
    #get data
    data = df.iloc[:,:-(len(var_headers))].values.T
    adata = ad.AnnData(X=data, obs=obs, var=var)
    return adata
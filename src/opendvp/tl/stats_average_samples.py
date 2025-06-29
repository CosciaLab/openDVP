import anndata as ad
import numpy as np
import pandas as pd

from opendvp.utils import logger


def stats_average_samples(
    adata: ad.AnnData,
    categories: list[str]
) -> ad.AnnData:
    """Average samples based on specified categories in adata.obs.

    This function groups cells by unique combinations of the provided `categories`
    and computes the mean expression for each feature within each group. The result
    is a new, smaller AnnData object where each observation corresponds to a
    unique category combination.

    The original, pre-averaged AnnData object is stored in the `.uns` attribute
    of the returned object under the key 'pre_averaged_adata'.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix to be averaged.
    categories : list of str
        A list of column names in `adata.obs` to group by for averaging.

    Returns:
    -------
    AnnData
        A new AnnData object where observations are the unique category combinations
        and variables are the averaged features.

    Raises:
    ------
    ValueError
        If any of the specified categories are not in `adata.obs.columns`.
    """
    logger.info(f"Averaging samples by categories: {categories}")

    # --- 1. Validation ---
    missing_cats = [cat for cat in categories if cat not in adata.obs.columns]
    if missing_cats:
        raise ValueError(f"Categories not found in adata.obs: {missing_cats}")

    # --- 2. Data Preparation ---
    # It's more efficient to work with a pandas DataFrame for groupby operations.
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    df = pd.DataFrame(X_dense, columns=adata.var_names, index=adata.obs.index)

    # Add the grouping categories to the DataFrame for the groupby operation
    df_with_groups = pd.concat([df, adata.obs[categories]], axis=1)

    # --- 3. Group and Average ---
    avg_df = df_with_groups.groupby(categories).mean()

    # --- 4. Create New AnnData Object ---
    new_obs = avg_df.index.to_frame(index=False)
    new_X = avg_df.to_numpy()
    adata_res = ad.AnnData(X=new_X, obs=new_obs, var=adata.var.copy())

    # --- 5. Store Original Data for Provenance ---
    logger.info("Storing original AnnData object in `.uns['pre_averaged_adata']`.")
    adata_res.uns['pre_averaged_adata'] = adata.copy()

    logger.success(f"Averaging complete. New AnnData shape: {adata_res.shape}")

    return adata_res
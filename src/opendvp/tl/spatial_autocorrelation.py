from collections.abc import Sequence

import anndata as ad
import esda
import numpy as np
import pandas as pd
from libpysal.weights import KNN, DistanceBand
from tqdm import tqdm

from opendvp.utils import logger


def spatial_autocorrelation(
    adata: ad.AnnData,
    method: str = "moran",
    x_y: Sequence[str] = ("x_centroid", "y_centroid"),
    k : int = 8,
    threshold : int | float = 10.0,
    island_threshold : float = 0.1
) -> None:
    """Compute spatial autocorrelation statistics (Moran's I or Geary's C) for each gene in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix where observations are cells and variables are genes.
    method : {'moran', 'geary'}, default 'moran'
        Spatial statistic to compute: 'moran' or 'geary'.
    x_y : Sequence[str], default ("x_centroid", "y_centroid")
        Names of columns in `adata.obs` containing spatial coordinates.
    k : int, default 8
        Number of neighbors for Moran's I (ignored for Geary's C).
    threshold : int or float, default 10.0
        Distance threshold for neighbors for Geary's C (ignored for Moran's I).
    island_threshold : float, default 0.1 (10%)
        If more than this fraction of samples are islands (no neighbors), raises error.

    Returns:
    -------
    None
        Results are added to adata.var in-place.

    Raises:
    ------
    ValueError
        If method is not 'moran' or 'geary'.
    RuntimeError
        If too many islands are detected in Geary's C mode.
    """
    logger.info(f"Starting spatial autocorrelation: {method.upper()}")
    logger.info(f"adata shape: obs={adata.n_obs}, var={adata.n_vars}")
    coords = adata.obs[list(x_y)].to_numpy()

    # Build spatial weights
    if method.lower() == "moran":
        logger.info(f"Building KNN graph with k={k}")
        w = KNN.from_array(coords, k=k)
        w.transform = "r"
    elif method.lower() == "geary":
        logger.info(f"Building DistanceBand graph with threshold={threshold}")
        w = DistanceBand.from_array(coords, threshold=threshold, binary=True)

        # Check islands
        n_islands = sum(1 for neighbors in w.neighbors.values() if len(neighbors) == 0)
        frac_islands = n_islands / w.n
        logger.info(f"Detected {n_islands} islands ({frac_islands:.2%} of samples)")

        if frac_islands > island_threshold:
            logger.error(
                f"Too many islands (> {island_threshold:.0%}). "
                f"Consider adjusting the threshold or coordinates."
            )
            raise RuntimeError(
                f"Too many islands ({frac_islands:.2%} > {island_threshold:.0%}) in DistanceBand graph. "
                f"Try increasing the threshold or check your coordinates."
            )
    else:
        logger.error("Method must be 'moran' or 'geary'.")
        raise ValueError("Method must be 'moran' or 'geary'.")

    results = []
    failed_genes = []

    logger.info(f"Starting calculation for {adata.n_vars} genes")
    for gene in tqdm(adata.var.index, desc=f"Running {method.title()}", leave=True):
        feature_values = np.asarray(adata[:, gene].X).flatten()

        # Pre-computation check for invalid values that would cause issues.
        if not np.all(np.isfinite(feature_values)):
            results.append(np.nan)
            failed_genes.append(gene)
            logger.warning(f"Skipping gene {gene}: contains NaN or Inf values.")
            continue
        if np.var(feature_values) == 0:
            results.append(np.nan)
            failed_genes.append(gene)
            logger.warning(f"Skipping gene {gene}: has zero variance (all values are constant).")
            continue

        try:
            if method.lower() == "moran":
                result = esda.moran.Moran(feature_values, w)
            elif method.lower() == "geary":
                result = esda.geary.Geary(feature_values, w)
            else:
                raise ValueError("Method must be 'moran' or 'geary'.")
            results.append(result)
        except Exception as e:
            results.append(np.nan)
            failed_genes.append(gene)
            logger.warning(f"Failed processing gene {gene}: {e}")

    if failed_genes:
        logger.warning(f"{len(failed_genes)} genes failed during {method.upper()} calculation")

    if method.lower() == "moran":
        adata.var[f'Moran_I_k{k}'] = pd.Series(
            [getattr(r, 'I', np.nan) if pd.notna(r) else np.nan for r in results],
            index=adata.var.index
        )
        adata.var[f'Moran_p_sim_k{k}'] = pd.Series(
            [getattr(r, 'p_sim', np.nan) if pd.notna(r) else np.nan for r in results],
            index=adata.var.index
        )
        adata.var[f'Moran_Zscore_k{k}'] = pd.Series(
            [getattr(r, 'z_sim', np.nan) if pd.notna(r) else np.nan for r in results],
            index=adata.var.index
        )
    elif method.lower() == "geary":
        adata.var[f'Geary_C_threshold{threshold}'] = pd.Series(
            [getattr(r, 'C', np.nan) if pd.notna(r) else np.nan for r in results],
            index=adata.var.index
        )
        adata.var[f'Geary_p_sim_threshold{threshold}'] = pd.Series(
            [getattr(r, 'p_sim', np.nan) if pd.notna(r) else np.nan for r in results],
            index=adata.var.index
        )

    logger.success(f"Finished spatial autocorrelation ({method.upper()}) computation.")
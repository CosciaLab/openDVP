import time

import anndata as ad
import pandas as pd

from opendvp.logger import logger


def filter_by_abs_value(adata, marker, value=None, quantile=None, keep='above', plot=False) -> ad.AnnData:
    """ 
    Filter cells by absolute value 
    Args:
        adata: anndata object
        marker: name of the marker to filter, string present in adata.var_names
        value: value to use as threshold
        quantile: quantile to use as threshold
        keep: 'above' or 'below', denoting which cells are kept
    """
    logger.info(" ---- filter_by_abs_value : version number 1.1.0 ----")
    time_start = time.time()

    # checks
    assert type(adata) is ad.AnnData, "adata should be an AnnData object"
    assert marker in adata.var_names, f"Marker {marker} not found in adata.var_names"
    # value or quantile
    if value is not None:
        assert quantile is None, "Only one of value or quantile should be provided"
        assert isinstance(value, (int, float)), "Value should be a number"
    elif quantile is not None:
        assert value is None, "Only one of value or quantile should be provided"
        assert isinstance(quantile, float), "Quantile should be a float"
        assert 0 < quantile < 1, "Quantile should be between 0 and 1"
    else:
        raise ValueError("Either value or quantile should be provided")
    # keep
    assert keep in ['above', 'below'], "keep should be either 'above' or 'below'"

    # set objects up 
    adata_copy = adata.copy()
    df = pd.DataFrame(data=adata_copy.X, columns=adata_copy.var_names)

    # calculate threshold
    if value is None:
        threshold = df[marker].quantile(quantile)
    else:
        threshold = value

    # Filter
    label = f"{marker}_{keep}_{threshold}"
    operator = '>' if keep == 'above' else '<'
    df[label] = df.eval(f"{marker} {operator} @threshold")
    adata_copy.obs[label] = df[label].values
    logger.info(f"Number of cells with {marker} {keep} {threshold}: {sum(df[label])}")

    # if plot:
    #     sns.histplot(df[marker], bins=500)
    #     plt.yscale('log')
    #     plt.xscale('log')
    #     plt.title(f'{marker} distribution')
    #     plt.axvline(threshold, color='black', linestyle='--', alpha=0.5)

    #     if keep == 'above':
    #         plt.text(threshold + 10, 1000, f"cells with {marker} > {threshold}", fontsize=9, color='black')
    #     elif keep == 'below':
    #         plt.text(threshold - 10, 1000, f"cells with {marker} < {threshold}", fontsize=9, color='black', horizontalalignment='right')
    #     plt.show()

    logger.info(f" ---- filter_by_abs_value is done, took {int(time.time() - time_start)}s  ----")
    return adata_copy
import time

import anndata as ad
import pandas as pd

from opendvp.logger import logger


def filter_by_ratio(adata, end_cycle, start_cycle, label="DAPI", min_ratio=0.5, max_ratio=1.05) -> ad.AnnData:
    """Filter cells by ratio"""
    logger.info(" ---- filter_by_ratio : version number 1.1.0 ----")
    #adapt to use with adata
    time_start = time.time()

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data=adata.X, columns=adata.var_names)
    df[f'{label}_ratio'] = df[end_cycle] / df[start_cycle]
    df[f'{label}_ratio_pass_nottoolow'] = df[f'{label}_ratio'] > min_ratio
    df[f'{label}_ratio_pass_nottoohigh'] = df[f'{label}_ratio'] < max_ratio
    df[f'{label}_ratio_pass'] = df[f'{label}_ratio_pass_nottoolow'] & df[f'{label}_ratio_pass_nottoohigh']

    # Pass to adata object
    adata.obs[f'{label}_ratio'] = df[f'{label}_ratio'].values
    adata.obs[f'{label}_ratio_pass_nottoolow']     = df[f'{label}_ratio_pass_nottoolow'].values
    adata.obs[f'{label}_ratio_pass_nottoohigh']    = df[f'{label}_ratio_pass_nottoohigh'].values
    adata.obs[f'{label}_ratio_pass']            = adata.obs[f'{label}_ratio_pass_nottoolow'] & adata.obs[f'{label}_ratio_pass_nottoohigh']

    # print out statistics
    logger.info(f"Number of cells with {label} ratio < {min_ratio}: {sum(df[f'{label}_ratio'] < min_ratio)}")
    logger.info(f"Number of cells with {label} ratio > {max_ratio}: {sum(df[f'{label}_ratio'] > max_ratio)}")
    logger.info(f"Number of cells with {label} ratio between {min_ratio} and {max_ratio}: {sum(df[f'{label}_ratio_pass'])}")
    logger.info(f"Percentage of cells filtered out: {round(100 - sum(df[f'{label}_ratio_pass'])/len(df)*100,2)}%")

    # plot histogram

    # fig, ax = plt.subplots()

    # sns.histplot(df[f'{label}_ratio'], color='blue')
    # plt.yscale('log')

    # plt.axvline(min_ratio, color='black', linestyle='--', alpha=0.5)
    # plt.axvline(max_ratio, color='black', linestyle='--', alpha=0.5)
    # plt.text(max_ratio + 0.05, 650, f"cells that gained >{int(max_ratio*100-100)}% {label}", fontsize=9, color='black')
    # plt.text(min_ratio - 0.05, 650, f"cells that lost >{int(min_ratio*100-100)}% {label}", fontsize=9, color='black', horizontalalignment='right')

    # plt.ylabel('cell count')
    # plt.xlabel(f'{label} ratio (last/cycle)')
    # plt.xlim(min_ratio-1, max_ratio+1)

    # plt.show()

    logger.info(f" ---- filter_by_ratio is done, took {int(time.time() - time_start)}s  ----")

    return adata
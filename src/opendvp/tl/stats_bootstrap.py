from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from opendvp.metrics import coefficient_of_variation
from opendvp.utils import logger


def stats_bootstrap(
    dataframe : pd.DataFrame,
    n_bootstrap : int = 100,
    subset_sizes : list | None = None,
    summary_func : Callable | Literal["count_above_threshold"] = np.mean,
    replace : bool = True,
    return_raw : bool = False,
    return_summary : bool = True,
    plot : bool =True,
    random_seed : int = 42,
    nan_policy : str = "omit",
    cv_threshold : float | None = None,
):
    """Evaluate the variability of feature-level coefficient of variation (CV) via bootstrapping.

    This function samples subsets from the input DataFrame and computes the CV (standard deviation divided by mean)
    of each feature (column) for each bootstrap replication. For each subset size, the function aggregates the CVs
    across bootstraps and then summarizes them with a user-specified statistic (e.g., mean, median). Optionally,
    the function can generate a violin plot of the summarized CVs across different subset sizes, and it returns the
    bootstrapped raw CVs and/or the summarized results.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data (features in columns, samples in rows).
    n_bootstrap : int, optional (default=100)
        Number of bootstrap replicates to perform for each subset size.
    subset_sizes : list of int, optional (default=[10, 50, 100])
        List of subset sizes (number of rows to sample) to use during the bootstrapping.
    summary_func : callable or 'count_above_threshold', optional (default=np.mean)
        Function to aggregate the per-feature CVs across bootstraps. For example, np.mean, np.median, etc.
        If set to "count_above_threshold", counts the number of CVs above `cv_threshold` for each feature.
    replace : bool, optional (default=True)
        Whether to sample with replacement (True, standard bootstrapping) or without (False, subsampling).
    cv_threshold : float, optional (default=None)
        Threshold for counting CVs above this value when summary_func is "count_above_threshold".
    return_raw : bool, optional (default=True)
        If True, returns the raw bootstrapped CVs in long format.
    return_summary : bool, optional (default=True)
        If True, returns a summary DataFrame where the per-feature bootstrapped CVs have been aggregated using
        `summary_func` for each subset size.
    plot : bool, optional (default=True)
        If True, displays a violin plot of the summarized CVs (one aggregated value per feature) across subset sizes.
    random_seed : int or None, optional (default=42)
        Seed for the random number generator, ensuring reproducibility.
    nan_policy : {'omit', 'raise', 'propagate'}, optional (default="omit")
        How to handle NaN values. Options are:
            - "omit": ignore NaNs during calculations,
            - "raise": raise an error if NaNs are encountered,
            - "propagate": allow NaNs to propagate in the output.

    Returns:
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        Depending on the flags `return_raw` and `return_summary`, the function returns:
            - If both are True: a tuple (raw_df, summary_df)
              * raw_df: DataFrame in long format with columns "feature", "cv", "subset_size", and "bootstrap_id".
              * summary_df: DataFrame with the aggregated CV (using `summary_func`) per feature and subset size,
                with columns "subset_size", "feature", and "cv_summary".
            - If only one of the flags is True, only that DataFrame is returned.
            - If neither is True, returns None.

    Raises:
    ------
    ValueError
        If any of the specified subset sizes is larger than the number of rows in `dataframe`.

    Examples:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(np.random.randn(100, 5))  # 100 samples, 5 features
    >>> raw_results, summary_results = bootstrap_variability(df, subset_sizes=[10, 20, 50])
    >>> summary_results.head()
         subset_size feature  cv_summary
    0           10       A    0.123456
    1           10       B    0.098765
    2           20       A    0.110987
    3           20       B    0.102345
    4           50       A    0.095432
    """
    # Safety checks
    logger.info("Starting bootstrap analysis...")
    if subset_sizes is None:
        subset_sizes = [10, 50, 100]
    if not replace and max(subset_sizes) > dataframe.shape[0]:
        raise ValueError("A subset size is larger than the number of rows in the dataframe when sampling without replacement.")

    rng = np.random.default_rng(seed=random_seed)
    all_results = []

    for size in tqdm(subset_sizes, desc="Subset sizes"):
        for i in tqdm(range(n_bootstrap), desc=f"Bootstraps (n={size})", leave=False):
            # Use a new random state for each sample to ensure independence
            subset = dataframe.sample(n=size, replace=replace, random_state=rng.integers(0, int(1e9)))
            cv = coefficient_of_variation(subset, axis=0, nan_policy=nan_policy)  # Series
            
            # Create a small long-format df for this iteration
            iter_df = cv.reset_index()
            iter_df.columns = ['feature', 'cv']
            iter_df['subset_size'] = size
            iter_df['bootstrap_id'] = i + 1
            all_results.append(iter_df)

    # Combine all subset sizes
    if not all_results:
        logger.warning("No bootstrap results were generated. Returning empty dataframes.")
        empty_df = pd.DataFrame(columns=['feature', 'cv', 'subset_size', 'bootstrap_id'])
        empty_summary = pd.DataFrame(columns=['subset_size', 'feature', 'cv_summary'])
        return (empty_df, empty_summary) if return_raw and return_summary else (empty_summary if return_summary else (empty_df if return_raw else None))

    results_df = pd.concat(all_results, ignore_index=True)

    # Summarize
    logger.info("Summarizing bootstrap results...")
    if summary_func == "count_above_threshold":
        if cv_threshold is None:
            raise ValueError("cv_threshold must be set when using 'count_above_threshold' as summary_func.")
        summary_df = (
            results_df.groupby(['subset_size', 'feature'])['cv']
            .apply(lambda x: (x > cv_threshold).sum())
            .reset_index()
            .rename(columns={'cv': 'cv_count_above_threshold'})
        )
    else:
        summary_df = (
            results_df.groupby(['subset_size', 'feature'])['cv']
            .agg(summary_func)
            .reset_index()
            .rename(columns={'cv': 'cv_summary'})
        )

    if plot:
        logger.info("Generating plot...")
        plt.figure(figsize=(8, 5))
        if summary_func == "count_above_threshold":
            sns.violinplot(data=summary_df, x="subset_size", y="cv_count_above_threshold")
            plt.ylabel(f"Count of CV > {cv_threshold} per feature")
        else:
            sns.violinplot(data=summary_df, x="subset_size", y="cv_summary")
            plt.ylabel(f"Aggregated CV per feature ({summary_func.__name__})")
        plt.title("Bootstrap variability across subset sizes")
        plt.xlabel("Subset size")
        plt.tight_layout()
        plt.show()

    if return_raw and return_summary:
        return results_df, summary_df
    elif return_summary:
        return summary_df
    elif return_raw:
        return results_df
    else:
        logger.success("Bootstrap analysis complete.")
        return None
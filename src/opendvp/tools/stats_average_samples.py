
import anndata as ad
import numpy as np
import pandas as pd


def calculate_category_averages(
    adata: ad.AnnData,
    categories: list[str]
) -> ad.AnnData:
    """Calculate averages for all permutations of given categories in adata.obs.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    categories : list of str
        List of categories (column names in adata.obs) to calculate averages for.

    Returns:
    -------
    AnnData
        AnnData object containing category combinations and their corresponding averages.
    """
    #LEGACY CODE
    print(f" --- --- --- Calculating averages for {categories} --- --- --- ")

    adata_copy = adata.copy()
    # Get the unique values for each category
    unique_values = [adata_copy.obs[cat].unique() for cat in categories]
    
    # Generate all possible combinations of category values
    combinations = pd.MultiIndex.from_product(unique_values, names=categories)
    
    # Create an empty DataFrame to store averages
    avg_df = pd.DataFrame(index=combinations, columns=adata_copy.var_names)
    
    # Loop through each category combination
    for combination in combinations:

        # Select cells that match the current category combination
        mask = np.all(np.vstack([adata_copy.obs[cat] == val for cat, val in zip(categories, combination, strict=False)]), axis=0)
        selected_cells = np.asarray(adata.X)[mask]
        
        # Calculate average for the selected cells and store it in the DataFrame
        avg_values = np.mean(selected_cells, axis=0)
        avg_df.loc[combination] = avg_values
    
    df_reset = avg_df.reset_index()

    adata_res = ad.AnnData(X=df_reset.iloc[:,2:].values, 
                            obs=df_reset[categories], 
                            var=adata_copy.var)

    return adata_res
import time

import decoupler as dc
import numpy as np
import scanpy as sc


# helper functions
def get_datetime():
    return time.strftime("%Y%m%d_%H%M%S")

def plot_ORA_heatmap(
        adata, 
        msigdb, 
        collection, 
        groupby_analysis, 
        groupby_plot, 
        n_pathways, 
        return_adata=False,
        return_acts=False,
        **kwargs): 
    """Description:
        Perform an Over-Representation Analysis (ORA) using the Decoupler package and plot the results as a heatmap.
    Parameters:
        adata: AnnData object
            Annotated data matrix.
        msigdb: DataFrame
            A DataFrame with the gene sets from the Molecular Signatures Database (MSigDB).
    """
    #TODO add option to save pathways as list_of_strings

    print("version 1.0.0")

    adata_copy = adata.copy()
    msigdb_collection = msigdb[msigdb['collection']==collection]
    msigdb_collection = msigdb_collection[~msigdb_collection.duplicated(['geneset', 'genesymbol'], keep='first')]
    print(f"Collection dataframe shape {msigdb_collection.shape}")
    #print 5 unique genesets
    print(f"{msigdb_collection['geneset'].nunique()} unique genesets")
    # print(f"{msigdb_collection['geneset'].value_counts().head()}")

    print("Running ORA")
    dc.run_ora(
        mat=adata_copy,
        net=msigdb_collection,
        source='geneset',
        target='genesymbol',
        verbose=True,
        use_raw=False,
    )

    acts = dc.get_acts(adata_copy, obsm_key="ora_estimate")

    if np.isinf(acts.X).sum() > 0:
        print("Infinite values found, replacing with max value")
        acts_v = acts.X.ravel()
        max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
        acts.X[~np.isfinite(acts.X)] = max_e

    print("Ranking top ", n_pathways, " pathways")
    pathways = dc.rank_sources_groups(adata=acts, groupby=groupby_analysis, reference='rest', method='t-test_overestim_var')
    source_markers = pathways.groupby('group').head(n_pathways).groupby('group')[
        'names'
    ].apply(lambda x: list(x)).to_dict()

    print("Plotting heatmap using scanpy")
    sc.pl.matrixplot(adata=acts,
                var_names=source_markers, 
                groupby=groupby_plot, 
                dendrogram=True,
                cmap='coolwarm',
                **kwargs
    )

    if return_adata and return_acts:
        print("returning adata object and acts")
        return adata_copy, acts
    elif return_adata and not return_acts:
        print("returning adata object")
        return adata_copy
    elif not return_adata and return_acts:
        return acts
    else:
        print("Done")
        return None
#!/usr/bin/env python3
# Created on Mon Oct 12 17:03:56 2020
# @author: Ajit Johnson Nirmal
"""Cluster cells by expression profile or other features.

Adapted from `scimap <https://github.com/labsyspharm/scimap>`_ (``sm.tl.cluster``).

Clusters cells within the dataset to identify distinct cell populations based on their
expression profiles or other relevant features. Two algorithms are supported:

- **kmeans**: a partitioning method that divides the dataset into ``k`` clusters, each
  represented by the centroid of the data points in the cluster. Suitable for identifying
  spherical clusters in the feature space.
- **leiden**: refines the cluster partitioning by optimizing a modularity score, detecting
  highly connected communities. Known for uncovering fine-grained, highly cohesive clusters.

Each algorithm has its own parameters and assumptions, making some more suitable than others
for particular dataset characteristics.
"""

# Import library
import pathlib
import sys

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans


def scimap_spatial_cluster(
    adata,
    method="kmeans",
    layer="log",
    subset_genes=None,
    sub_cluster=False,
    sub_cluster_column="phenotype",
    sub_cluster_group=None,
    k=10,
    n_pcs=None,
    resolution=1,
    phenograph_clustering_metric="euclidean",
    nearest_neighbors=30,
    use_raw=True,
    log=True,
    random_state=0,
    collapse_labels=False,
    label=None,
    verbose=True,
    output_dir=None,
):
    # Load the andata object
    if isinstance(adata, str):
        imid = str(adata.rsplit("/", 1)[-1])
        adata = anndata.read_h5ad(adata)
    else:
        adata = adata

    # dynamically adapt the number of neighbours
    if nearest_neighbors > adata.shape[0]:
        nearest_neighbors = adata.shape[0] - 3

    # prepare data to be clustered

    # Leiden clustering
    def leiden_clustering(pheno, adata, nearest_neighbors, n_pcs, resolution):
        # subset the data to be clustered
        if pheno is not None:
            cell_subset = adata.obs[adata.obs[sub_cluster_column] == pheno].index
        else:
            cell_subset = adata.obs.index

        if use_raw == True:
            data_subset = adata[cell_subset]
            if log is True:
                data_subset.X = np.log1p(data_subset.raw.X)
            else:
                data_subset.X = data_subset.raw.X
        else:
            data_subset = adata[cell_subset]

        # clustering
        if pheno is not None:
            if verbose:
                print("Leiden clustering " + str(pheno))
        else:
            if verbose:
                print("Leiden clustering")

        sc.tl.pca(data_subset)
        if n_pcs is None:
            n_pcs = len(adata.var)
        sc.pp.neighbors(data_subset, n_neighbors=nearest_neighbors, n_pcs=n_pcs)
        sc.tl.leiden(data_subset, resolution=resolution, random_state=random_state)

        # Rename the labels
        cluster_labels = list(map(str, list(data_subset.obs["leiden"])))
        if pheno is not None:
            cluster_labels = list(map(lambda orig_string: pheno + "-" + orig_string, cluster_labels))

        # Make it into a dataframe
        cluster_labels = pd.DataFrame(cluster_labels, index=data_subset.obs.index)

        # return labels
        return cluster_labels

    # Kmeans clustering
    def k_clustering(pheno, adata, k, sub_cluster_column, use_raw, random_state):
        # subset the data to be clustered
        if pheno is not None:
            cell_subset = adata.obs[adata.obs[sub_cluster_column] == pheno].index
        else:
            cell_subset = adata.obs.index

        # Usage of scaled or raw data
        if use_raw == True:
            if log is True:
                data_subset = pd.DataFrame(
                    np.log1p(adata.raw[cell_subset].X),
                    columns=adata[cell_subset].var.index,
                    index=adata[cell_subset].obs.index,
                )
            else:
                data_subset = pd.DataFrame(
                    adata.raw[cell_subset].X,
                    columns=adata[cell_subset].var.index,
                    index=adata[cell_subset].obs.index,
                )
        else:
            data_subset = pd.DataFrame(
                adata[cell_subset].X,
                columns=adata[cell_subset].var.index,
                index=adata[cell_subset].obs.index,
            )

        # K-means clustering
        if pheno is not None:
            if verbose:
                print("Kmeans clustering " + str(pheno))
        else:
            if verbose:
                print("Kmeans clustering")

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(data_subset)

        # Rename the labels
        cluster_labels = list(map(str, kmeans.labels_))
        if pheno is not None:
            cluster_labels = list(map(lambda orig_string: pheno + "-" + orig_string, cluster_labels))

        # Make it into a
        cluster_labels = pd.DataFrame(cluster_labels, index=data_subset.index)

        # return labels
        return cluster_labels

    # Use user defined genes for clustering
    if subset_genes is not None:
        bdata = adata[:, subset_genes]
        bdata.raw = bdata[:, subset_genes]
    else:
        bdata = adata.copy()

    # IF sub-cluster is True
    # What cells to run the clustering on?
    if sub_cluster is True:
        if sub_cluster_group is not None:
            if isinstance(sub_cluster_group, list):
                pheno = sub_cluster_group
            else:
                pheno = [sub_cluster_group]
        else:
            # Make sure number of clusters is not greater than number of cells available
            if method == "kmeans":
                pheno = (bdata.obs[sub_cluster_column].value_counts() > k + 1).index[
                    bdata.obs[sub_cluster_column].value_counts() > k + 1
                ]
            if method == "phenograph":
                pheno = (bdata.obs[sub_cluster_column].value_counts() > nearest_neighbors + 1).index[
                    bdata.obs[sub_cluster_column].value_counts() > nearest_neighbors + 1
                ]
            if method == "leiden":
                pheno = (bdata.obs[sub_cluster_column].value_counts() > 1).index[
                    bdata.obs[sub_cluster_column].value_counts() > 1
                ]

    # Run the specified method
    if method == "kmeans":
        if sub_cluster == True:
            # Apply the Kmeans function
            r_k_clustering = lambda x: k_clustering(
                pheno=x,
                adata=bdata,
                k=k,
                sub_cluster_column=sub_cluster_column,
                use_raw=use_raw,
                random_state=random_state,
            )  # Create lamda function
            all_cluster_labels = list(map(r_k_clustering, pheno))  # Apply function
        else:
            all_cluster_labels = k_clustering(
                pheno=None,
                adata=bdata,
                k=k,
                sub_cluster_column=sub_cluster_column,
                use_raw=use_raw,
                random_state=random_state,
            )

    if method == "leiden":
        if sub_cluster == True:
            r_leiden_clustering = lambda x: leiden_clustering(
                pheno=x,
                adata=bdata,
                nearest_neighbors=nearest_neighbors,
                n_pcs=n_pcs,
                resolution=resolution,
            )  # Create lamda function
            all_cluster_labels = list(map(r_leiden_clustering, pheno))  # Apply function
        else:
            all_cluster_labels = leiden_clustering(
                pheno=None,
                adata=bdata,
                nearest_neighbors=nearest_neighbors,
                n_pcs=n_pcs,
                resolution=resolution,
            )

    # Merge all the labels into one and add to adata
    if sub_cluster == True:
        sub_clusters = pd.concat(all_cluster_labels, axis=0, sort=False)
    else:
        sub_clusters = all_cluster_labels

    # Merge with all cells
    # sub_clusters = pd.DataFrame(bdata.obs[sub_cluster_column]).merge(sub_clusters, how='outer', left_index=True, right_index=True)
    sub_clusters = pd.DataFrame(bdata.obs).merge(sub_clusters, how="outer", left_index=True, right_index=True)

    # Transfer labels
    if collapse_labels is False and sub_cluster is True:
        sub_clusters = pd.DataFrame(sub_clusters[0].fillna(sub_clusters[sub_cluster_column]))

    # Get only the required column
    sub_clusters = sub_clusters[0]

    # re index the rows
    sub_clusters = sub_clusters.reindex(adata.obs.index)

    # Append to adata
    if label is None:
        adata.obs[method] = sub_clusters
    else:
        adata.obs[label] = sub_clusters

    # Save data if requested
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        adata.write(output_dir / imid)
    else:
        # Return data
        return adata


# Command line compatible
def main(argv=sys.argv):
    cluster(**vars(args))

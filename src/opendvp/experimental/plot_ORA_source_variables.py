import time

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# helper functions
def get_datetime():
    return time.strftime("%Y%m%d_%H%M%S")


def plot_ORA_source_variables(
    adata: ad.AnnData,
    msigdb: pd.DataFrame,
    geneset: str,
    groupby_analysis: str,
) -> None:
    """Description:
    Plots gene expression by group with median-centered scaling and sorts genes by the mean difference in expression between groups.
    Adds a second subplot with the combined expression of all genes per group.
    """
    #TODO max number of genes to plot, dynamic number of genes to plot, too many and switch to heatmap??
    #TODO summary plot should be violin plot, boxplot loses information about density

    print("version 1.2.0 from 14.11.24")
    adata_copy = adata.copy()

    # assert adata is ready
    assert adata_copy.var.index.name == "Genes", "adata.var.index should be 'Genes'"
    assert adata_copy.var.index.isna().sum() == 0, "adata.var.index should not have any NA values"
    assert not any(";" in str(i) for i in adata_copy.var.index), (
        "adata.var.index contains ';' characters, gene list should be cleaned"
    )
    assert "ora_estimate" in adata_copy.obsm, "ora_estimate not found in adata.obsm."

    # assert msigdb is ready
    assert geneset in msigdb['geneset'].values, f"geneset {geneset} not found in msigdb"

    # Remove duplicates and get list of genes in the specified geneset
    msigdb = msigdb[~msigdb.duplicated(['geneset', 'genesymbol'], keep='first')]
    pathway_genes = msigdb[msigdb['geneset'] == geneset]['genesymbol'].tolist()

    # Check how many genes from geneset are present in adata.var.index
    pathway_genes_present = [gene for gene in pathway_genes if gene in adata_copy.var.index]
    print(f"{len(pathway_genes_present)} genes from {len(pathway_genes)} from {geneset} are present in adata")

    # Create dataframe from adata to plot
    expression_data = adata_copy[:, pathway_genes_present].to_df()
    # Add group labels to the data
    expression_data['Group'] = adata_copy.obs[groupby_analysis]

    # Apply median-centered scaling
    medians = expression_data[pathway_genes_present].median()
    expression_data[pathway_genes_present] = expression_data[pathway_genes_present].subtract(medians, axis=1)

    # Calculate mean expression per gene per group
    mean_expression = expression_data.groupby('Group').mean().T
    # Calculate mean difference between the two groups
    mean_expression['Difference'] = mean_expression.iloc[:, 0] - mean_expression.iloc[:, 1]
    # Sort genes by the mean difference
    sorted_genes = mean_expression['Difference'].sort_values().index.tolist()

    # Melt into long form for the first subplot
    df_long = expression_data.melt(id_vars='Group', value_vars=pathway_genes_present, var_name='Gene', value_name='Expression')
    # Set categorical order for genes based on mean difference sorting
    df_long['Gene'] = pd.Categorical(df_long['Gene'], categories=sorted_genes, ordered=True)

    # Prepare figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [7, 1]})

    # First subplot: Gene-wise boxplot
    sns.boxplot(x='Gene', y='Expression', hue='Group', data=df_long, dodge=True, width=0.6, ax=axes[0])
    sns.stripplot(
        data=df_long, x='Gene', y='Expression', hue='Group',
        dodge=True, marker='o', palette='dark:black', alpha=0.6, size=3, ax=axes[0],
    )
    axes[0].set_title(f"{geneset} by Group (Sorted by Mean Difference)")
    axes[0].set_ylabel("Median-Centered Expression")
    axes[0].tick_params(axis='x', rotation=45)

    # Adjust legend to avoid double legend due to hue
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[0:2], labels[0:2], title="Group")

    # Second subplot: Combined boxplot of all genes by group
    sns.boxplot(x='Group', y='Expression', hue='Group', data=df_long, ax=axes[1], width=0.3, showfliers=False)

    sns.stripplot(x='Group', y='Expression', data=df_long, color='black', alpha=0.6, size=3, ax=axes[1])
    axes[1].set_title("")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")

    # Match y-axis limits between subplots
    y_min, y_max = axes[0].get_ylim()
    axes[1].set_ylim(y_min, y_max)
    axes[1].axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
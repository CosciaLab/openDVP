def plot_rcn_stacked_barplot(df, phenotype_col, rcn_col, normalize=True):
    """
    Plots a stacked barplot showing phenotype composition per RCN motif.
    
    Parameters:
    df (DataFrame): Input dataframe containing phenotype and RCN columns
    phenotype_col (str): Column name for phenotypes
    rcn_col (str): Column name for RCN motifs
    normalize (bool): If True, normalize frequencies to proportions per motif
    """
    # Count frequencies of each phenotype within each RCN
    count_df = df.groupby([rcn_col, phenotype_col]).size().unstack(fill_value=0)
    
    # Normalize to proportions if requested
    if normalize:
        count_df = count_df.div(count_df.sum(axis=1), axis=0)
    
    # Create the stacked barplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottoms = [0] * len(count_df)
    for phenotype, color in phenotype_colors.items():
        if phenotype in count_df.columns:
            ax.bar(count_df.index, count_df[phenotype],
                   bottom=bottoms, color=color, label=phenotype)
            bottoms = [i + j for i, j in zip(bottoms, count_df[phenotype])]
    
    # Customize plot
    ax.legend(title="Phenotype", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_xlabel("RCN Motif")
    ax.set_title("Phenotype Composition per RCN Motif")
    plt.tight_layout()
    plt.show()
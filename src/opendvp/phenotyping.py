from loguru import logger
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
# import scimap as sm

# for rescale
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import json


def read_gates(gates_csv_path, sample_id=None) -> pd.DataFrame:
    """ Read the gates data from a csv file and return a dataframe """
    logger.info(" ---- read_gates : version number 1.1.0 ----")
    time_start = time.time()

    assert gates_csv_path.endswith('.csv'), "The file should be a csv file"
    gates = pd.read_csv(gates_csv_path)
    
    logger.info("   Filtering out all rows with value 0.0 (assuming not gated)")
    assert "gate_value" in gates.columns, "The column gate_value is not present in the csv file"
    gates = gates[gates.gate_value != 0.0]
    logger.info(f"  Found {gates.shape[0]} valid gates")
    logger.info(f"  Markers found: {gates.marker_id.unique()}")
    logger.info(f"  Samples found: {gates.sample_id.unique()}")

    if sample_id is not None:
        assert "sample_id" in gates.columns, "The column sample_id is not present in the csv file"
        gates = gates[gates['sample_id']==sample_id]
        logger.info(f"  Found {gates.shape[0]} valid gates for sample {sample_id}")

    logger.info(f" ---- read_gates is done, took {int(time.time() - time_start)}s  ----")
    return gates
    
def process_gates_for_sm(gates:pd.DataFrame, sample_id) -> pd.DataFrame:
    """ Process gates dataframe to be in log1p scale """
    logger.info(" ---- process_gates_for_sm : version number 1.2.0 ----")
    time_start = time.time()

    df = gates.copy()

    df['log1p_gate_value'] = np.log1p(gates.gate_value)
    gates_for_scimap = df[['marker_id', 'log1p_gate_value']]
    gates_for_scimap.rename(columns={'marker_id': 'markers', 'log1p_gate_value': sample_id}, inplace=True)

    logger.info(f" ---- process_gates_for_sm is done, took {int(time.time() - time_start)}s  ----")
    return gates_for_scimap

def negate_var_by_ann(adata, target_variable, target_annotation_column , quantile_for_imputation=0.05) -> ad.AnnData:

    assert quantile_for_imputation >= 0 and quantile_for_imputation <= 1, "Quantile should be between 0 and 1"
    assert target_variable in adata.var_names, f"Variable {target_variable} not found in adata.var_names"
    assert target_annotation_column in adata.obs.columns, f"Annotation column {target_annotation_column} not found in adata.obs.columns"

    adata_copy = adata.copy()

    target_var_idx = adata_copy.var_names.get_loc(target_variable)
    target_rows = adata_copy.obs[target_annotation_column].values
    value_to_impute = np.quantile(adata_copy[:, target_var_idx].X.toarray(), quantile_for_imputation)
    logger.info(f"Imputing with {quantile_for_imputation}% percentile value = {value_to_impute}")

    adata_copy.X[target_rows, target_var_idx] = value_to_impute
    return adata_copy

def rescale(
    adata,
    gate=None,
    log=True,
    imageid='imageid',
    failed_markers=None,
    method='all',
    verbose=True,
    random_state=0,
    gmm_components=3,
):
    """
    Parameters:
        adata (AnnData Object, required):
            An annotated data object that contains single-cell expression data.

        gate (DataFrame, optional):
            A pandas DataFrame where the first column lists markers, and subsequent columns contain gate values
            for each image in the dataset. Column names must correspond to unique `imageid` identifiers, and the marker column must be named "markers".
            If a single column of gate values is provided for a dataset with multiple images, the same gate will be uniformly applied to all images.
            In this case, ensure that the columns are named exactly "markers" and "gates".
            If no gates are provided for specific markers, the function attempts to automatically determine gates using a Gaussian Mixture Model (GMM).

            Note: If you have used `napariGater()`, the gates are stored within `adata.uns['gates']`.
            You can directly pass `adata.uns['gates']` to use these pre-defined gates.

        log (bool, optional):
            If `True`, the data in `adata.raw.X` will be log-transformed (using log1p) before gate application. This transformation is recommended when automatic gate identification through GMM is performed, as it helps in normalizing data distributions.

        imageid (str, optional):
            The name of the column in `adata` that contains Image IDs. This is necessary for matching manual gates specified in the `gate` DataFrame to their respective images.

        failed_markers (dict, optional):
            A dictionary mapping `imageid` to markers that failed quality control. This allows for the exclusion of specific markers from the analysis based on prior visual inspection or other criteria. The dictionary can use `all` as a key to specify markers that failed across all images.

        method (str, optional):
            Specifies the gating strategy: `all` to pool data from all images for GMM application, or `by_image` to apply GMM separately for each image. `all` may introduce batch effects, while `by_image` requires sufficient variation within each image to distinguish negative from positive populations effectively.

        random_state (int, optional):
            The seed used by the random number generator for GMM. Ensures reproducibility of results.

        verbose (bool, optional):
            If `True`, detailed progress updates and diagnostic messages will be printed during the function's execution.

        gmm_components (int, optional):
            Number of components to use in the Gaussian Mixture Model for automatic gating.
            Must be at least 2. Gate will be placed between the highest two components.
            Default is 3.

    Returns:
        Modified AnnData Object (AnnData):
            Returns the input `adata` object with updated expression data (`adata.X`) after rescaling. The gates applied, either provided manually or determined automatically, are stored within `adata.uns['gates']`.

    Example:
        ```python

        # Example with manual gates
        manual_gate = pd.DataFrame({'marker': ['CD3D', 'KI67'], 'gate': [7, 8]})
        adata = sm.pp.rescale(adata, gate=manual_gate, failed_markers={'all': ['CD20', 'CD21']})

        # Importing gates from a CSV
        manual_gate = pd.read_csv('manual_gates.csv')
        adata = sm.pp.rescale(adata, gate=manual_gate, failed_markers={'all': ['CD20', 'CD21']})

        # Running without manual gates to use GMM for automatic gate determination
        adata = sm.pp.rescale(adata, gate=None, failed_markers={'all': ['CD20', 'CD21']})

        ```

    """

    # log=True; imageid='imageid'; failed_markers=None; method='all'; random_state=0

    # make a copy to raw data if raw is none
    if adata.raw is None:
        adata.raw = adata

    # Mapping between markers and gates in the given dataset
    dataset_markers = adata.var.index.tolist()
    dataset_images = adata.obs[imageid].unique().tolist()
    m = pd.DataFrame(index=dataset_markers, columns=dataset_images).reset_index()
    m = pd.melt(m, id_vars=[m.columns[0]])
    m.columns = ['markers', 'imageid', 'gate']

    # Manipulate m with and without provided manual gates
    if gate is None:
        gate_mapping = m.copy()
    else:
        # Check overlap between gate columns and dataset images
        matching_images = set(gate.columns) & set(dataset_images)

        # link to make sure index name is markers as we use reset_index later
        if gate.index.name != 'markers' and 'markers' not in gate.columns:
            gate.index.name = 'markers'

        if len(matching_images) == 0 and len(gate.columns) > 0:
            # Case 1: No matching images and single value column - apply globally
            gate = gate.reset_index()  # Convert index to column
            gate_mapping = m.copy()
            gate_mapping.gate = gate_mapping.gate.fillna(
                gate_mapping.markers.map(
                    dict(
                        zip(gate['markers'], gate['gates'])
                    )  # these columns are hardcoded in CSV
                )
            )
        else:
            # Case 2: handles both if all imageid matches with gate columns or if they partially match
            gate = gate.reset_index()
            manual_m = pd.melt(gate, id_vars=gate[['markers']])
            manual_m.columns = ['markers', 'imageid', 'm_gate']
            gate_mapping = pd.merge(
                m,
                manual_m,
                how='left',
                left_on=['markers', 'imageid'],
                right_on=['markers', 'imageid'],
            )
            gate_mapping['gate'] = gate_mapping['gate'].fillna(gate_mapping['m_gate'])
            gate_mapping = gate_mapping.drop(columns='m_gate')

    # Addressing failed markers
    def process_failed(adata_subset, foramted_failed_markers):
        if verbose:
            print(
                'Processing Failed Marker in '
                + str(adata_subset.obs[imageid].unique()[0])
            )
        # prepare data
        data_subset = pd.DataFrame(
            adata_subset.raw.X,
            columns=adata_subset.var.index,
            index=adata_subset.obs.index,
        )
        if log is True:
            data_subset = np.log1p(data_subset)

        # subset markers in the subset
        fm_sub = foramted_failed_markers[adata_subset.obs[imageid].unique()].dropna()

        def process_failed_internal(fail_mark, data_subset):
            return data_subset[fail_mark].max()

        r_process_failed_internal = lambda x: process_failed_internal(
            fail_mark=x, data_subset=data_subset
        )
        f_g = list(map(r_process_failed_internal, [x[0] for x in fm_sub.values]))
        subset_gate = pd.DataFrame(
            {
                'markers': [x[0] for x in fm_sub.values],
                'imageid': adata_subset.obs[imageid].unique()[0],
                'gate': f_g,
            }
        )
        # return
        return subset_gate

    # Identify the failed markers
    if failed_markers is not None:
        # check if failed marker is a dict
        if isinstance(failed_markers, dict) is False:
            raise ValueError(
                '`failed_markers` should be a python dictionary, please refer documentation'
            )
        # create a copy
        fm = failed_markers.copy()
        # seperate all from the rest
        if 'all' in failed_markers:
            all_failed = failed_markers['all']
            if isinstance(all_failed, str):
                all_failed = [all_failed]
            failed_markers.pop('all', None)

            df = pd.DataFrame(columns=adata.obs[imageid].unique())
            for i in range(len(all_failed)):
                df.loc[i] = np.repeat(all_failed[i], len(df.columns))
            # for i in  range(len(df.columns)):
            #    df.loc[i] = all_failed[i]
        # rest of the failed markers
        # fail = pd.DataFrame.from_dict(failed_markers)
        fail = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in failed_markers.items()])
        )
        # merge
        if 'all' in fm:
            foramted_failed_markers = pd.concat([fail, df], axis=0)
        else:
            foramted_failed_markers = fail

        # send the adata objects that need to be processed
        # Check if any image needs to pass through the GMM protocol
        adata_list = [
            adata[adata.obs[imageid] == i] for i in foramted_failed_markers.columns
        ]
        # apply the process_failed function
        r_process_failed = lambda x: process_failed(
            adata_subset=x, foramted_failed_markers=foramted_failed_markers
        )
        failed_gates = list(map(r_process_failed, adata_list))
        # combine the results and merge with gate_mapping
        result = []
        for i in range(len(failed_gates)):
            result.append(failed_gates[i])
        result = pd.concat(result, join='outer')
        # use this to merge with gate_mapping
        x1 = gate_mapping.set_index(['markers', 'imageid'])['gate']
        x2 = result.set_index(['markers', 'imageid'])['gate']
        x1.update(x2)
        gate_mapping = x1.reset_index()

    # trim the data before applying GMM
    def clipping(x):
        clip = x.clip(
            lower=np.percentile(x, 0.01), upper=np.percentile(x, 99.99)
        ).tolist()
        return clip

    # Find GMM based gates
    def gmm_gating(marker, data, gmm_components):
        """Internal function to identify gates using GMM

        Parameters:
            marker: marker name
            data: expression data
            gmm_components: number of components for GMM (minimum 2)
        """
        # Ensure minimum of 2 components
        gmm_components = max(2, gmm_components)

        # Prepare data for GMM
        data_gm = data[marker].values.reshape(-1, 1)
        data_gm = data_gm[~np.isnan(data_gm), None]

        # Fit GMM with gmm_components
        gmm = GaussianMixture(
            n_components=gmm_components, random_state=random_state
        ).fit(data_gm)

        # Sort components by their means
        means = gmm.means_.flatten()
        sorted_idx = np.argsort(means)
        sorted_means = means[sorted_idx]

        # Calculate gate as midpoint between the second-to-last and last components
        gate = np.mean([sorted_means[-2], sorted_means[-1]])

        return gate

    # Running gmm_gating on the dataset
    def gmm_gating_internal(adata_subset, gate_mapping, method):
        return result

    # Create a list of image IDs that need to go through the GMM
    gmm_images = gate_mapping[gate_mapping.gate.isnull()].imageid.unique()

    # Check if any image needs to pass through the GMM protocol
    if len(gmm_images) > 0:
        # Create a list of adata that need to go through the GMM
        if method == 'all':
            adata_list = [adata]
        else:
            adata_list = [adata[adata.obs[imageid] == i] for i in gmm_images]
        # run function
        r_gmm_gating_internal = lambda x: gmm_gating_internal(
            adata_subset=x, gate_mapping=gate_mapping, method=method
        )
        all_gates = list(map(r_gmm_gating_internal, adata_list))

        # combine the results and merge with gate_mapping
        result = []
        for i in range(len(all_gates)):
            result.append(all_gates[i])
        result = pd.concat(result, join='outer')
        # use this to merge with gate_mapping
        gate_mapping.gate = gate_mapping.gate.fillna(
            gate_mapping.markers.map(dict(zip(result.markers, result.gate)))
        )

    # Rescaling function
    def data_scaler(adata_subset, gate_mapping):
        if verbose:
            print('\nScaling Image: ' + str(adata_subset.obs[imageid].unique()[0]))
        # Organise data
        data_subset = pd.DataFrame(
            adata_subset.raw.X,
            columns=adata_subset.var.index,
            index=adata_subset.obs.index,
        )
        if log is True:
            data_subset = np.log1p(data_subset)
        # subset markers in the subset
        gate_mapping_sub = gate_mapping[
            gate_mapping['imageid'] == adata_subset.obs[imageid].unique()[0]
        ]

        # organise gates
        def data_scaler_internal(marker, gate_mapping_sub):
            if verbose:
                gate_value = gate_mapping_sub[gate_mapping_sub.markers == marker][
                    'gate'
                ].values[0]
                print(f'Scaling {marker} (gate: {gate_value:.3f})')
            # find the gate
            moi = gate_mapping_sub[gate_mapping_sub.markers == marker]['gate'].values[0]

            # Find the closest value to the gate
            absolute_val_array = np.abs(data_subset[marker].values - float(moi))
            # throw error if the array has nan values
            if np.isnan(absolute_val_array).any():
                raise ValueError(
                    "An exception occurred: " + str(marker) + ' has nan values'
                )
            # smallest diff
            smallest_difference_index = absolute_val_array.argmin()
            closest_element = data_subset[marker].values[smallest_difference_index]

            # rescale the data based on the identified gate
            marker_study = data_subset[marker]
            marker_study = marker_study.sort_values(axis=0)
            # Find the index of the gate
            # account for 0
            if all(marker_study == 0):
                gate_index = pd.DataFrame(marker_study).tail(2).index[0]
            else:
                gate_index = marker_study.index[marker_study == closest_element][0]
            # Split into high and low groups
            high = marker_study[gate_index:]
            low = marker_study[:gate_index]
            # Prepare for scaling the high and low dataframes
            scaler_high = MinMaxScaler(feature_range=(0.5, 1))
            scaler_low = MinMaxScaler(feature_range=(0, 0.5))
            # Scale it
            h = pd.DataFrame(
                scaler_high.fit_transform(high.values.reshape(-1, 1)), index=high.index
            )
            l = pd.DataFrame(
                scaler_low.fit_transform(low.values.reshape(-1, 1)), index=low.index
            )
            # Merge the high and low and resort it
            scaled_data = pd.concat([l, h])
            scaled_data = scaled_data.loc[~scaled_data.index.duplicated(keep='first')]
            scaled_data = scaled_data.reindex(data_subset.index)
            # scaled_data[scaled_data > 0.5].count(axis=1).sum()
            # return
            return scaled_data

        # run internal function
        r_data_scaler_internal = lambda x: data_scaler_internal(
            marker=x, gate_mapping_sub=gate_mapping_sub
        )
        scaled_subset = list(
            map(r_data_scaler_internal, gate_mapping_sub.markers.values)
        )

        # combine the results and merge with gate_mapping
        scaled_subset_result = []
        for i in range(len(scaled_subset)):
            scaled_subset_result.append(scaled_subset[i])
        scaled_subset_result = pd.concat(scaled_subset_result, join='outer', axis=1)
        scaled_subset_result.columns = gate_mapping_sub.markers.values
        # scaled_subset_result[scaled_subset_result['CD3E'] > 0.5]['CD3E'].count(axis=1).sum()

        # return
        return scaled_subset_result

    # pass each dataset seperately
    adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]

    # Run the scaler function
    r_data_scaler = lambda x: data_scaler(adata_subset=x, gate_mapping=gate_mapping)
    scaled_subset = list(map(r_data_scaler, adata_list))

    # combine the results and merge with gate_mapping
    final_result = []
    for i in range(len(scaled_subset)):
        final_result.append(scaled_subset[i])
    final_result = pd.concat(final_result, join='outer')

    # reindex the final_results
    final_result = final_result.reindex(adata.obs.index)

    # save final gates
    adata.uns['gates'] = gate_mapping.pivot_table(
        index=['markers'], columns=['imageid']
    ).droplevel(
        0, axis=1
    )  # .reset_index()

    # add to the anndata
    adata.X = final_result

    # return adata
    return adata


def phenotype_cells (adata, 
                     phenotype, 
                     gate = 0.5, 
                     label="phenotype", 
                     imageid='imageid',
                     pheno_threshold_percent=None, 
                     pheno_threshold_abs=None,
                     verbose=True
                     ):
    """
    
Parameters:
    adata (anndata.AnnData):  
        The input AnnData object containing single-cell data for phenotyping.

    phenotype (pd.DataFrame):  
        A DataFrame specifying the gating strategy for cell phenotyping. It should outline the workflow for phenotype classification based on marker expression levels. An example workflow is available at [this GitHub link](https://github.com/ajitjohnson/scimap/blob/master/scimap/tests/_data/phenotype_workflow.csv).
        
    gate (float, optional):  
        The threshold value for determining positive cell classification based on scaled data. By convention, values above this threshold are considered to indicate positive cells. 
        
    label (str):  
        The name of the column in `adata.obs` where the final phenotype classifications will be stored. This label will be used to access the phenotyping results within the `AnnData` object.
        
    imageid (str, optional):  
        The name of the column in `adata.obs` that contains unique image identifiers. This is crucial for analyses that require differentiation of data based on the source image, especially when using phenotype threshold parameters (`pheno_threshold_percent` or `pheno_threshold_abs`).
        
    pheno_threshold_percent (float, optional):  
        A threshold value (between 0 and 100) specifying the minimum percentage of cells that must exhibit a particular phenotype for it to be considered valid. Phenotypes not meeting this threshold are reclassified as 'unknown'. This parameter is useful for minimizing the impact of low-frequency false positives. 
        
    pheno_threshold_abs (int, optional):  
        Similar to `pheno_threshold_percent`, but uses an absolute cell count instead of a percentage. Phenotypes with cell counts below this threshold are reclassified as 'unknown'. This can help in addressing rare phenotype classifications that may not be meaningful. 
    
    verbose (bool):  
        If set to `True`, the function will print detailed messages about its progress and the steps being executed.

Returns:
    adata (anndata.AnnData):  
        The input AnnData object, updated to include the phenotype classifications for each cell. The phenotyping results can be found in `adata.obs[label]`, where `label` is the name specified by the user for the phenotype column.

Example:    
    ```python
    
    # Load the phenotype workflow CSV file
    phenotype = pd.read_csv('path/to/csv/file/')  
    
    # Apply phenotyping to cells based on the specified workflow
    adata = sm.tl.phenotype_cells(adata, phenotype=phenotype, gate=0.5, label="phenotype")
    
    ```

    """
    # Create a dataframe from the adata object
    data = pd.DataFrame(adata.X, columns = adata.var.index, index= adata.obs.index)

    # Function to calculate the phenotype scores
    def phenotype_cells (data,phenotype,gate,group):

        # Subset the phenotype based on the group
        phenotype = phenotype[phenotype.iloc[:,0] == group]

        # Parser to parse the CSV file into four categories
        def phenotype_parser (p, cell):
            # Get the index and subset the phenotype row being passed in
            location = p.iloc[:,1] == cell
            idx = [i for i, x in enumerate(location) if x][0]
            phenotype = p.iloc[idx,:]
            # Calculate
            pos = phenotype[phenotype == 'pos'].index.tolist()
            neg = phenotype[phenotype == 'neg'].index.tolist()
            anypos = phenotype[phenotype == 'anypos'].index.tolist()
            anyneg = phenotype[phenotype == 'anyneg'].index.tolist()
            allpos = phenotype[phenotype == 'allpos'].index.tolist()
            allneg = phenotype[phenotype == 'allneg'].index.tolist()
            return {'pos': pos, 'neg': neg ,'anypos': anypos, 'anyneg': anyneg, 'allpos': allpos, 'allneg': allneg}
            #return pos, neg, anypos, anyneg

        # Run the phenotype_parser function on all rows
        p_list = phenotype.iloc[:,1].tolist()
        r_phenotype = lambda x: phenotype_parser(cell=x, p=phenotype) # Create lamda function
        all_phenotype = list(map(r_phenotype, p_list)) # Apply function
        all_phenotype = dict(zip(p_list, all_phenotype)) # Name the lists

        # Define function to check if there is any marker that does not satisfy the gate
        def gate_satisfation_lessthan (marker, data, gate):
            fail = np.where(data[marker] < gate, 1, 0) # 1 is fail
            return fail
        # Corresponding lamda function
        r_gate_satisfation_lessthan = lambda x: gate_satisfation_lessthan(marker=x, data=data, gate=gate)

        # Define function to check if there is any marker that does not satisfy the gate
        def gate_satisfation_morethan (marker, data, gate):
            fail = np.where(data[marker] > gate, 1, 0)
            return fail
        # Corresponding lamda function
        r_gate_satisfation_morethan = lambda x: gate_satisfation_morethan(marker=x, data=data, gate=gate)

        def prob_mapper (data, all_phenotype, cell, gate):
            
            if verbose:
                print("Phenotyping " + str(cell))

            # Get the appropriate dict from all_phenotype
            p = all_phenotype[cell]

            # Identiy the marker used in each category
            pos = p.get('pos')
            neg = p.get('neg')
            anypos = p.get('anypos')
            anyneg = p.get('anyneg')
            allpos = p.get('allpos')
            allneg = p.get('allneg')

            # Perform computation for each group independently
            # Positive marker score
            if len(pos) != 0:
                pos_score = data[pos].mean(axis=1).values
                pos_fail = list(map(r_gate_satisfation_lessthan, pos)) if len(pos) > 1 else []
                pos_fail = np.amax(pos_fail, axis=0) if len(pos) > 1 else []
            else:
                pos_score = np.repeat(0, len(data))
                pos_fail = []

            # Negative marker score
            if len(neg) != 0:
                neg_score = (1-data[neg]).mean(axis=1).values
                neg_fail = list(map(r_gate_satisfation_morethan, neg)) if len(neg) > 1 else []
                neg_fail = np.amax(neg_fail, axis=0) if len(neg) > 1 else []
            else:
                neg_score = np.repeat(0, len(data))
                neg_fail = []

            # Any positive score
            anypos_score = np.repeat(0, len(data)) if len(anypos) == 0 else data[anypos].max(axis=1).values

            # Any negative score
            anyneg_score = np.repeat(0, len(data)) if len(anyneg) == 0 else (1-data[anyneg]).max(axis=1).values

            # All positive score
            if len(allpos) != 0:
                allpos_score = data[allpos]
                allpos_score['score'] = allpos_score.max(axis=1)
                allpos_score.loc[(allpos_score < gate).any(axis = 1), 'score'] = 0
                allpos_score = allpos_score['score'].values + 0.01 # A small value is added to give an edge over the matching positive cell
            else:
                allpos_score = np.repeat(0, len(data))


            # All negative score
            if len(allneg) != 0:
                allneg_score = 1- data[allneg]
                allneg_score['score'] = allneg_score.max(axis=1)
                allneg_score.loc[(allneg_score < gate).any(axis = 1), 'score'] = 0
                allneg_score = allneg_score['score'].values + 0.01
            else:
                allneg_score = np.repeat(0, len(data))


            # Total score calculation
            # Account for differences in the number of categories used for calculation of the final score
            number_of_non_empty_features = np.sum([len(pos) != 0,
                                                len(neg) != 0,
                                                len(anypos) != 0,
                                                len(anyneg) != 0,
                                                len(allpos) != 0,
                                                len(allneg) != 0])

            total_score = (pos_score + neg_score + anypos_score + anyneg_score + allpos_score + allneg_score) / number_of_non_empty_features

            return {cell: total_score, 'pos_fail': pos_fail ,'neg_fail': neg_fail}
            #return total_score, pos_fail, neg_fail


        # Apply the fuction to get the total score for all cell types
        r_prob_mapper = lambda x: prob_mapper (data=data, all_phenotype=all_phenotype, cell=x, gate=gate) # Create lamda function
        final_scores = list(map(r_prob_mapper, [*all_phenotype])) # Apply function
        final_scores = dict(zip([*all_phenotype], final_scores)) # Name the lists

        # Combine the final score to annotate the cells with a label
        final_score_df = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i][i])
            final_score_df= pd.concat([final_score_df, df], axis=1)
        # Name the columns
        final_score_df.columns = [*final_scores]
        final_score_df.index = data.index
        # Add a column called unknown if all markers have a value less than the gate (0.5)
        unknown = group + str('-rest')
        final_score_df[unknown] = (final_score_df < gate).all(axis=1).astype(int)

        # Name each cell
        labels = final_score_df.idxmax(axis=1)

        # Group all failed instances (i.e. when multiple markers were given
        # any one of the marker fell into neg or pos zones of the gate)
        pos_fail_all = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i]['pos_fail'])
            df.columns = [i] if len(df) != 0 else []
            pos_fail_all= pd.concat([pos_fail_all, df], axis=1)
        pos_fail_all.index = data.index if len(pos_fail_all) != 0 else []
        # Same for Neg
        neg_fail_all = pd.DataFrame()
        for i in [*final_scores]:
            df = pd.DataFrame(final_scores[i]['neg_fail'])
            df.columns = [i] if len(df) != 0 else []
            neg_fail_all= pd.concat([neg_fail_all, df], axis=1)
        neg_fail_all.index = data.index if len(neg_fail_all) != 0 else []


        # Modify the labels with the failed annotations
        if len(pos_fail_all) != 0:
            for i in pos_fail_all.columns:
                labels[(labels == i) & (pos_fail_all[i] == 1)] = 'likely-' + i
        # Do the same for negative
        if len(neg_fail_all) != 0:
            for i in neg_fail_all.columns:
                labels[(labels == i) & (neg_fail_all[i] == 1)] = 'likely-' + i

        # Retun the labels
        return labels

    # Create an empty dataframe to hold the labeles from each group
    phenotype_labels = pd.DataFrame()

    # Loop through the groups to apply the phenotype_cells function
    for i in phenotype.iloc[:,0].unique():

        if phenotype_labels.empty:
            phenotype_labels = pd.DataFrame(phenotype_cells(data = data, group = i, phenotype=phenotype, gate=gate))
            phenotype_labels.columns = [i]

        else:
            # Find the column with the cell-type of interest
            column_of_interest = [] # Empty list to hold the column name
            try:
                column_of_interest = phenotype_labels.columns[phenotype_labels.eq(i).any()]
            except:
                pass
            # If the cell-type of interest was not found just add NA
            if len(column_of_interest) == 0:
                phenotype_labels[i] = np.nan
            else:
                #cells_of_interest = phenotype_labels[phenotype_labels[column_of_interest] == i].index
                cells_of_interest = phenotype_labels[phenotype_labels[column_of_interest].eq(i).any(axis=1)].index
                d = data.loc[cells_of_interest]
                if verbose:
                    print("-- Subsetting " + str(i))
                phenotype_l = pd.DataFrame(phenotype_cells(data = d, group = i, phenotype=phenotype, gate=gate), columns = [i])
                phenotype_labels = phenotype_labels.merge(phenotype_l, how='outer', left_index=True, right_index=True)

    # Rearrange the rows back to original
    phenotype_labels = phenotype_labels.reindex(data.index)
    phenotype_labels = phenotype_labels.replace('-rest', np.nan, regex=True)

    if verbose:
        print("Consolidating the phenotypes across all groups")
    phenotype_labels_Consolidated = phenotype_labels.fillna(method='ffill', axis = 1)
    phenotype_labels[label] = phenotype_labels_Consolidated.iloc[:,-1].values

    # replace nan to 'other cells'
    phenotype_labels[label] = phenotype_labels[label].fillna('Unknown')

    # Apply the phenotype threshold if given
    if pheno_threshold_percent or pheno_threshold_abs is not None:
        p = pd.DataFrame(phenotype_labels[label])
        q = pd.DataFrame(adata.obs[imageid])
        p = q.merge(p, how='outer', left_index=True, right_index=True)

        # Function to remove phenotypes that are less than the given threshold
        def remove_phenotype(p, ID, pheno_threshold_percent, pheno_threshold_abs):
            d = p[p[imageid] == ID]
            x = pd.DataFrame(d.groupby([label]).size())
            x.columns = ['val']
            # FInd the phenotypes that are less than the given threshold
            if pheno_threshold_percent is not None:
                fail = list(x.loc[x['val'] < x['val'].sum() * pheno_threshold_percent/100].index)
            if pheno_threshold_abs is not None:
                fail = list(x.loc[x['val'] < pheno_threshold_abs].index)
            d[label] = d[label].replace(dict(zip(fail, ['Unknown'] * len(fail) )))
            # Return
            return d

        # Apply function to all images
        r_remove_phenotype = lambda x: remove_phenotype (p=p, ID=x,
                                                         pheno_threshold_percent=pheno_threshold_percent,
                                                         pheno_threshold_abs=pheno_threshold_abs) # Create lamda function
        final_phrnotypes= list(map(r_remove_phenotype, list(p[imageid].unique()))) # Apply function

        final_phrnotypes = pd.concat(final_phrnotypes, join='outer')
        phenotype_labels = final_phrnotypes.reindex(adata.obs.index)


    # Return to adata
    adata.obs[label] = phenotype_labels[label]

    #for i in phenotype_labels.columns:
    #    adata.obs[i] = phenotype_labels[i]

    return adata
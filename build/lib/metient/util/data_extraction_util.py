import csv
import numpy as np
import os
import torch
from collections import OrderedDict
import pandas as pd
import sys
from metient.util.globals import *

# TODO: make more assertions on uniqueness and completeness of input csvs

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def get_adjacency_matrix_from_txt_edge_list(txt_file):
    edges = []
    max_idx = -1
    with open(txt_file) as f:
        for line in f:
            s = line.strip().split(" ")
            edge = (int(s[0]), int(s[1]))
            edges.append(edge)
            max_idx = max(edge[0], max_idx)
            max_idx = max(edge[1], max_idx)

    T = torch.zeros((max_idx+1,max_idx+1))
    for edge in edges:
        T[edge[0], edge[1]] = 1

    return T

def get_mut_to_cluster_map_from_pyclone_output(pyclone_cluster_fn, min_mut_thres=0):
    clstr_id_to_muts, mutation_names = load_pyclone_clusters(pyclone_cluster_fn, min_mut_thres=min_mut_thres)
    mut_name_to_clstr_id = {}
    clstr_id_to_name = {}
    for cid in clstr_id_to_muts:
        for mut in clstr_id_to_muts[cid]:
            mut_name_to_clstr_id[mut] = cid
        clstr_id_to_name[cid] = ";".join(clstr_id_to_muts[cid])
    return mut_name_to_clstr_id, clstr_id_to_name, mutation_names

def _validate_combo(df, tsv_fn, col1, col2, message):
    grouped = df.groupby(col1)[col2].unique()
    valid_mapping = grouped.apply(lambda x: len(x) == 1).all()
    if not valid_mapping:
        raise ValueError(f"{message}. Issue in: {tsv_fn}")

def validate_unpooled_or_pooled_df(df, tsv_fn, index_cols):
    '''
    Validation needed for either unpooled or pooled tsv
    '''

    assert(set(df['site_category'])==set(['primary', 'metastasis']))

    # Check if all index columns are whole numbers, and convert to int if needed
    for index_col in index_cols:
        assert df[index_col].dtype in ['int64', 'float64'], f"{index_col} column is not int or float"
        if df[index_col].dtype == 'float64':
            is_whole_number = df[index_col].apply(lambda x: x.is_integer())
            has_decimal = not is_whole_number.all()
            if has_decimal:
                raise ValueError(f"{index_col} column must contain whole numbers only. Issue in: {tsv_fn}")
            
    df['cluster_index'] = df['cluster_index'].astype(int)
    
    # Check that each anatomical site index maps to only one anatomical site label
    _validate_combo(df, tsv_fn, 'anatomical_site_index', 'anatomical_site_label', "Each anatomical site index must correspond to the same anatomical site label.")
    # Check that each anatomical site label maps to only one anatomical site index
    _validate_combo(df, tsv_fn, 'anatomical_site_label','anatomical_site_index', "Each anatomical site label must correspond to the same anatomical site index.")
    # Check that each anatomical site index maps to only one site category
    _validate_combo(df, tsv_fn, 'anatomical_site_index', 'site_category', "Each unique anatomical site index must correspond to the same site_category.")
    # Check that all index columns go from 0 to max
    assert set(df['cluster_index']) == set(range(len(df['cluster_index'].unique()))), f"cluster_index values do not go from 0 to max. Issue in: {tsv_fn}"
    assert set(df['anatomical_site_index']) == set(range(len(df['anatomical_site_index'].unique()))), f"anatomical_site_index values do not go from 0 to max, Issue in: {tsv_fn}"

def validate_prepooled_tsv(tsv_fn):
    # Validate the input tsv
    df = pd.read_csv(tsv_fn, delimiter="\t", index_col=False)  
    index_cols = ['anatomical_site_index', 'cluster_index']
    validate_unpooled_or_pooled_df(df, tsv_fn, index_cols)

def pool_input_tsv(tsv_fn, output_dir, run_name):
    '''
    Pool reads from the same anatomical site index and cluster index
    '''
    df = pd.read_csv(tsv_fn, delimiter="\t", index_col=False)  
    validate_unpooled_or_pooled_df(df, tsv_fn, ['anatomical_site_index', 'cluster_index', 'character_index'])
    df['cluster_index'] = df['cluster_index'].astype(int)
    # Check that each character index maps to only one character label
    _validate_combo(df, tsv_fn, 'character_index', 'character_label', "Each character index must correspond to the same character_label.")
    # Check that all var_read_probs are between [0.0,1.0]
    valid_var_read_probs = (df['var_read_prob'] >= 0.0) & (df['var_read_prob'] <= 1.0)
    if not valid_var_read_probs.all():
        raise ValueError(f"All values in the var_read_prob column must be between 0.0 and 1.0 inclusive. Issue in: {tsv_fn}")
    
    _, output_fn = write_pooled_tsv_from_clusters(df, {}, output_dir, run_name)
    
    return output_fn

def write_pooled_tsv_from_clusters(df, aggregation_rules, output_dir, patient_id):
    '''
    After clustering with any clustering algorithm, prepares tsvs by pooling mutations belonging 
    to the same cluster and anatomical site index

    Args:
        df: pandas DataFrame with reqiured columns: [character_label, 
        anatomical_site_index, anatomical_site_label, ref, var, var_read_prob, site_category]

        mut_idx_to_cluster_id: dictionary mapping mutation index ('character_index' in input df)
        to a cluster id

        cluster_id_to_cluster_name: dictionary mapping a cluster id to the name in the output tsv

        aggregation_rules: dictionary indicating how to aggregate any extra columns that are not in 
        the required columns (specificied above). e.g. "first" aggregation rules can be used for columns 
        shared by rows with the same anatomical_site_label, since we are aggregating within an
        anatomical_site_label. 

        output_dir: where to save clustered tsv

        patient_id: name of patient used in tsv filename
    
    Outputs:

        Saves pooled tsv at {output_dir}/{patient_id}_clustered_SNVs.tsv

    '''

    required_cols = ["anatomical_site_index", "anatomical_site_label", "cluster_index",
                     "character_index", "character_label", "ref","var", 
                     "var_read_prob", "site_category"]
    
    all_cols = required_cols + list(aggregation_rules.keys())

    if not (set(required_cols).issubset(df.columns)):
        missing_cols = set(required_cols) - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not set(all_cols) == set(df.columns):
        missing_columns = set(df.columns) - set(all_cols)
        df.drop(columns=missing_columns, inplace=True)
        print(f"WARNING: dropping extra columns without aggregation rules: {missing_columns}")

    # 1. Fix the value of each variant's total read count to account for the var_read_prob of each variant,
    # and then we can set var_read_prob to 0.5 (see PairTree's supplement, end of section S3.8)
    # First calculate individual variant read probabilities
    df['total_reads_corrected'] = df.apply(lambda row: 2*row['var_read_prob']*(row['ref']+row['var']), axis=1)
    df['var'] = df.apply(lambda row: min(row['var'], row['total_reads_corrected']), axis=1)
    df['ref'] = df.apply(lambda row: row['total_reads_corrected']-row['var'], axis=1)
    df['var_read_prob'] = 0.5
    df = df.dropna(subset=['cluster_index'])

    # Save the number of mutations in each cluster before pooling
    cluster_id_to_num_muts = df.groupby('cluster_index')['character_index'].nunique().to_dict()
    cluster_id_to_mut_names = df.groupby('cluster_index')['character_label'].unique().apply(list).to_dict()

    # 2. Pool reference and variant allele counts from all mutations within a cluster
    pooled_df = df.drop(['character_label','character_index'], axis=1) # we're going to add this back in later

    ref_var_rules = {'ref': np.sum, 'var': np.sum,'total_reads_corrected': np.sum, "var_read_prob": 'first', 
                     'site_category':'first', 'anatomical_site_label':lambda x: ';'.join(set(x)),}

    pooled_df = pooled_df.groupby(['cluster_index', 'anatomical_site_index'], as_index=False).agg({**ref_var_rules, **aggregation_rules})
    # 3. Add indices for mutations, samples and anatomical sites as needed for input format
    pooled_df['character_index'] = pooled_df['cluster_index'].tolist()
    pooled_df['anatomical_site_index'] = pooled_df.apply(lambda row: list(pooled_df['anatomical_site_label'].unique()).index(row["anatomical_site_label"]), axis=1)
    
    # 4. Do some post-processing, e.g. adding number of mutations and shortening character label for display
    pooled_df['num_mutations'] = pooled_df.apply(lambda row:cluster_id_to_num_muts[int(row['character_index'])], axis=1)
    pooled_df['character_label'] = pooled_df.apply(lambda row:cluster_id_to_mut_names[row['cluster_index']], axis=1)
    pooled_df['total_reads_corrected'] = pooled_df['total_reads_corrected'].round(0).astype(int)
    pooled_df['var'] = pooled_df['var'].round(0).astype(int)
    pooled_df['ref'] = pooled_df.apply(lambda row: row['total_reads_corrected']-row['var'], axis=1)
    all_cols.append('num_mutations')

    pooled_df = pooled_df[all_cols]

    # Save
    output_fn = os.path.join(output_dir, f"{patient_id}_clustered_SNVs.tsv")
    with open(output_fn, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(pooled_df.columns)
        for _, row in pooled_df.iterrows():
            writer.writerow(row)

    return pooled_df, output_fn

def calc_var_read_prob(major_cn, minor_cn, purity, diploid=True):
    major_cn = int(major_cn)
    minor_cn = int(minor_cn)
    p = float(purity)
    factor = 2 if diploid else 1
    x = (p*(major_cn+minor_cn)+factor*(1-p))
    var_read_prob = (p*major_cn)/x
    return var_read_prob

def get_idx_to_cluster_label(cluster_filepath):
    '''
    Args:
        cluster_filepath: path to cluster file for MACHINA simulated data in the format:
        0
        1
        3;15;17;22;24;29;32;34;53;56
        69;78;80;81

        where each semi-colon separated number represents a mutation that belongs
        to cluster i where i is the file line number.

    Returns:
        (1) a dictionary mapping cluster number to cluster name
        for e.g. for the file above, this would return:
            {0: '0', 1: '69;78;80;81'}
    '''
    idx_to_cluster_label = OrderedDict()
    with open(cluster_filepath) as f:
        i = 0
        for line in f:
            label = line.strip()
            idx_to_cluster_label[i] = label
            i += 1
    return idx_to_cluster_label

def get_ref_var_omega_matrices(tsv_filepaths):
    '''
    tsv_filepaths: List of paths to tsvs (one for each patient). Expects columns:

    ['anatomical_site_index', 'anatomical_site_label', 'character_index', 'character_label', 
    'ref', 'var', 'var_read_prob']
    Returns:
    (1) list of R matrices (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) list of V matrices (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) list of omega matrices (num_samples x num_clusters) with the variant read probability of the mutation/mutation cluster in the sample,
    (4) list of list of unique anatomical sites from the patient's data,
    (5) list of dictionaries mapping index to character_label (based on input tsv, gives the index for each mutation name,
    (6) list of dictionaries mapping index to number of mutations (if 'num_mutations' is passed, else None)
    where these indices are used in R matrix, V matrix

    For each of the returned lists, index 0 represents the patient at the first tsv_filepath, etc.
    '''
    if not isinstance(tsv_filepaths, list):
        return get_ref_var_omega_matrix(tsv_filepaths)

    assert(len(tsv_filepaths) >= 1)

    ref_matrices, var_matrices, omega_matrices, ordered_sites, idx_to_label_dicts = [], [], [], [], []
    idx_to_num_muts = None

    for tsv_filepath in tsv_filepaths:
        ref, var, omega, sites, idx_to_label_dict, idx_to_num_mut_dict =  get_ref_var_omega_matrix(tsv_filepath)
        ref_matrices.append(ref)
        var_matrices.append(var)
        omega_matrices.append(omega)
        ordered_sites.append(sites)
        idx_to_label_dicts.append(idx_to_label_dict)
        if idx_to_num_mut_dict != None:
            if idx_to_num_muts == None:
                idx_to_num_muts = []
            idx_to_num_muts.append(idx_to_num_mut_dict)


    return ref_matrices, var_matrices, omega_matrices, ordered_sites, idx_to_label_dicts, idx_to_num_muts

def get_primary_sites(tsv_filepath):
    # For tsvs with very large fields
    csv.field_size_limit(sys.maxsize)

    df = pd.read_csv(tsv_filepath, sep="\t")
    primary_sites = list(df[df['site_category']=='primary']['anatomical_site_label'].unique())
    if len(primary_sites) < 1:
        raise ValueError("No primary found in site_category column")

    return primary_sites

import ast

def get_ref_var_omega_matrix(tsv_filepath):
    '''
    tsv_filepath: path to tsv with columns:

    ['anatomical_site_index', 'anatomical_site_label', 'character_index', 'character_label', 
    'ref', 'var', 'var_read_prob', 'site_category']

    Optional additional flags: 'num_mutations'

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) omega matrix (num_samples x num_clusters) with the variant read probability of the mutation/mutation cluster in the sample,
    (4) unique anatomical sites from the patient's data,
    (5) dictionary mapping index to mut_names (based on input tsv, gives the index for each mutation name,
    where these indices are used in R matrix, V matrix)
    (6) dictionary mapping index to number of mutations (if 'num_mutations' is passed, else None)
    '''

    df = pd.read_csv(tsv_filepath, delimiter="\t", index_col=False)  

    num_sites = df['anatomical_site_index'].max() + 1
    num_clusters = df['cluster_index'].max() + 1
    # Build R and V matrices
    idx_to_mut_names = dict()
    R = np.zeros((num_sites, num_clusters))
    V = np.zeros((num_sites, num_clusters))
    omega = np.zeros((num_sites, num_clusters))
    unique_sites = [""]*num_sites
    idx_to_num_muts = dict()
    for _, row in df.iterrows():
        x = int(row['anatomical_site_index'])
        y = int(row['character_index'])
        R[x,y] = int(float(row['ref']))
        V[x,y] = int(float(row['var']))
        omega[x,y] = float(row['var_read_prob'])
        # collect additional metadata
        unique_sites[x] = row['anatomical_site_label']
        # pandas won't let you save a list very easily, so it converts to string...this is a workaround
        # to turn it back into a string
        idx_to_mut_names[y] = ast.literal_eval(row['character_label'])
        if y not in idx_to_num_muts:
            idx_to_num_muts[y] = float(row['num_mutations'])
        
    return torch.tensor(R, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), torch.tensor(omega, dtype=torch.float32), list(unique_sites), idx_to_mut_names, idx_to_num_muts

def extract_info_from_observed_clone_tsv(tsv_filename):
    '''
    When observed clones are inputted, we don't need to get ref or var information
    '''
    df = pd.read_csv(tsv_filename, delimiter="\t", index_col=False)  

    num_sites = df['anatomical_site_index'].max() + 1
    idx_to_label = dict()
    idx_to_sites_present = dict()
    unique_sites = [""]*num_sites
    idx_to_num_muts = dict()
    for _, row in df.iterrows():
        x = int(row['anatomical_site_index'])
        y = int(row['cluster_index'])
        # collect additional metadata
        unique_sites[x] = row['anatomical_site_label']
        idx_to_label[y] = [row['cluster_label']]
        if y not in idx_to_num_muts:
            idx_to_num_muts[y] = float(row['num_mutations'])
        if row['present'] == 1:
            if y not in idx_to_sites_present:
                idx_to_sites_present[y] = []
            idx_to_sites_present[y].append(row['anatomical_site_index'])
    return unique_sites, idx_to_label, idx_to_num_muts, idx_to_sites_present

def extract_ordered_sites(tsv_filepaths):
    def _extract_ordered_sites_from_single_tsv(tsv_filename):
        df = pd.read_csv(tsv_filename, delimiter="\t", index_col=False)  
        sorted_df = df.sort_values(by='anatomical_site_index')
        # Get unique labels in the order of index
        unique_site_labels_in_order = sorted_df['anatomical_site_label'].unique().tolist()
        return unique_site_labels_in_order
    
    if not isinstance(tsv_filepaths, list):
        return _extract_ordered_sites_from_single_tsv(tsv_filepaths)

    assert(len(tsv_filepaths) >= 1)
    ordered_sites = []
    for fn in tsv_filepaths:
        ordered_sites.append(_extract_ordered_sites_from_single_tsv(fn))
    return ordered_sites

def extract_matrices_from_tsv(tsv_fn, estimate_observed_clones, T):
    if estimate_observed_clones:    
        ref, var, omega, ordered_sites, node_idx_to_label, idx_to_num_mutations = get_ref_var_omega_matrix(tsv_fn)
        if not torch.is_tensor(ref):
            ref = torch.tensor(ref, dtype=torch.float32)
        if not torch.is_tensor(var):
            var = torch.tensor(var, dtype=torch.float32)
        idx_to_observed_sites = None # needs to be estimated later
    else:
        ref, var, omega = None, None, None
        ordered_sites, node_idx_to_label, idx_to_num_mutations, idx_to_observed_sites = extract_info_from_observed_clone_tsv(tsv_fn)

    G = None
    # If genetic distance info is given, load into genetic distance matrix
    if idx_to_num_mutations != None:
        G = get_genetic_distance_matrix_from_adj_matrix(T, idx_to_num_mutations)

    return ref, var, omega, ordered_sites, node_idx_to_label, idx_to_observed_sites, G

def _get_adj_matrix_from_spruce_tree(tree_edges, idx_to_character_label, remove_unseen_nodes=True):
    '''
    Args:
        tree_edges: list of tuples where each tuple is an edge in the tree
        idx_to_character_label: dictionary mapping index to character_label (machina
        uses colors to represent subclones, so this would map n to 'pink', if pink
        is the nth node in the adjacency matrix).
        remove_unseen_nodes: if True, removes nodes that
        appear in the machina tsv file but do not appear in the reported tree

    Returns:
        T: adjacency matrix where Tij = 1 if there is a path from i to j
        idx_to_character_label: a pruned idx_to_character_label where nodes that
        appear in the machina tsv file but do not appear in the reported tree are removed
    '''
    character_label_to_idx = {v:k for k,v in idx_to_character_label.items()}
    num_internal_nodes = len(character_label_to_idx)
    T = np.zeros((num_internal_nodes, num_internal_nodes))
    seen_nodes = set()
    for edge in tree_edges:
        node_i, node_j = edge[0], edge[1]
        seen_nodes.add(node_i)
        seen_nodes.add(node_j)
        # don't include the leaf/extant nodes (determined from U)
        if node_i in character_label_to_idx and node_j in character_label_to_idx:
            T[character_label_to_idx[node_i], character_label_to_idx[node_j]] = 1

    unseen_nodes = list(set(character_label_to_idx.keys()) - seen_nodes)
    pruned_character_label_to_idx = OrderedDict()
    if remove_unseen_nodes:
        unseen_node_indices = [character_label_to_idx[unseen_node] for unseen_node in unseen_nodes]
        T = np.delete(T, unseen_node_indices, 0)
        T = np.delete(T, unseen_node_indices, 1)

        i = 0
        for char_label in character_label_to_idx:
            if char_label not in unseen_nodes:
                pruned_character_label_to_idx[char_label] = i
                i += 1
        if len(unseen_nodes) > 0:
            #print("Removing unseen nodes:", unseen_nodes, pruned_character_label_to_idx)
            pass

    ret_dict = pruned_character_label_to_idx if remove_unseen_nodes else character_label_to_idx
    ret_dict = {v:k for k,v in ret_dict.items()}
    return T, ret_dict

def get_adj_matrices_from_spruce_mutation_trees(mut_trees_filename, idx_to_character_label, is_sim_data=False):
    '''
    When running MACHINA's generatemutationtrees executable (SPRUCE), it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of tuples, each containing (T, pruned_idx_to_character_label) for each
    tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - idx_to_character_label: a dict mapping indices of the adj matrix T to character
        labels 
    '''
    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_data = []
        for i, line in enumerate(f):
            if i < 3: continue
            # This marks the beginning of a tree
            if "#edges, tree" in line:
                adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_spruce_tree(tree_data, idx_to_character_label)
                out.append((adj_matrix, pruned_idx_to_label))
                tree_data = []
            else:
                nodes = line.strip().split()
                # fixes incompatibility with naming b/w cluster file (uses ";" separator)
                # and the ./generatemutationtrees output (uses "_" separator)
                if is_sim_data:
                    tree_data.append((";".join(nodes[0].split("_")), ";".join(nodes[1].split("_"))))
                else:
                    tree_data.append((nodes[0], nodes[1]))

        adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_spruce_tree(tree_data, idx_to_character_label)
        out.append((adj_matrix, pruned_idx_to_label))
    return out

def get_adj_matrices_from_all_conipher_trees(mut_trees_filename):
    '''
    Extracts trees after running CONIPHER (https://github.com/McGranahanLab/CONIPHER-wrapper)

    Returns a list of adjacency matrices, where Tij = 1 if there is a path from i to j
    '''

    def _get_adj_matrix_from_edges(edges):
        nodes = set([node for edge in edges for node in edge])
        T = np.zeros((len(nodes), len(nodes)))
        for edge in edges:
            T[edge[0], edge[1]] = 1
        return torch.tensor(T, dtype = torch.float32)

    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_edges = []
        for i, line in enumerate(f):
            if i < 2: continue
            # This marks the beginning of a tree
            if "# tree" in line:
                adj_matrix = _get_adj_matrix_from_edges(tree_edges)
                out.append(adj_matrix)
                tree_edges = []
            else:
                nodes = line.strip().split()
                tree_edges.append((int(nodes[0]), int(nodes[1])))

        adj_matrix = _get_adj_matrix_from_edges(tree_edges)
        out.append(adj_matrix)
    return out


def get_genetic_distance_matrices_from_adj_matrices(adj_matrices, idx_to_character_labels, split_char, normalize=True):
    '''
    Get the genetic distances between nodes by counting the number of mutations between
    parent and child. idx_to_character_label's keys are expected to be mutation/cluster indices with
    the mutation or mutations in that cluster (e.g. 'ENAM:4:71507837_DLG1:3:196793590'). split_char
    indicates what the mutations in the cluster name are split by (if it's a cluster).

    If a single adj_matrix or idx_to_character_label is inputted, a single genetic distance matrix is returned. 
    Otherwise, a list of genetic distance matrices is returned.
    '''
    if not isinstance(adj_matrices, list):
        return get_genetic_distance_matrix_from_adj_matrix(adj_matrices, idx_to_character_labels, split_char, normalize=normalize)

    assert(len(adj_matrices) >= 1 and (len(adj_matrices) == len(idx_to_character_labels)))

    gen_dist_matrices = []
    for adj_matrix, idx_to_character_label in zip(adj_matrices, idx_to_character_labels):
        gen_dist_matrix = get_genetic_distance_matrix_from_adj_matrix(adj_matrix, idx_to_character_label, split_char, normalize=normalize)
        gen_dist_matrices.append(gen_dist_matrix)

    return gen_dist_matrices

def get_genetic_distance_matrix_from_adj_matrix(adj_matrix, idx_to_num_muts, normalize=True):
    '''
    Get the genetic distances between nodes by using the number of mutations between parent and child. 
    The number of mutations in a node is given in idx_to_num_muts.
    '''
    G = np.zeros(adj_matrix.shape)

    for i, adj_row in enumerate(adj_matrix):
        for j, val in enumerate(adj_row):
            if val == 1:
                # This is the number of mutations the child node has accumulated compared to its parent
                num_mutations = idx_to_num_muts[j]
                G[i][j] = num_mutations

    if normalize:
        G = G / np.sum(G)
    return torch.tensor(G, dtype = torch.float32)

def get_organotropism_matrix_from_msk_met(ordered_sites, cancer_type, frequency_csv, site_to_msk_met_map=None):
    '''
    Args:
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        cancer_type: cancer type name, which is used as the 
        frequency_csv: csv with frequency of metastasis by cancer type
        site_to_msk_met_map: dictionary mapping site names to MSK-MET site names. if
        not provided, the names used in the ordered_sites array are used

    Returns:
        array of size len(ordered_sites) with the frequency with which
        primary cancer type seeds site i
    '''

    freq_df = pd.read_csv(frequency_csv)
    if (cancer_type not in list(freq_df['Cancer Type'])):
        raise ValueError(f"{cancer_type} is not in MSK-MET data as a primary cancer type")

    if site_to_msk_met_map != None:
        for site in ordered_sites:
            if (site not in site_to_msk_met_map):
                raise ValueError(f"{site} not in provided site_to_msk_met_map")
        mapped_sites = [site_to_msk_met_map[site] for site in ordered_sites]
    else:
        mapped_sites = ordered_sites

    for site in mapped_sites:
        if (site not in list(freq_df.columns)):
            raise ValueError(f"{site} not in MSK-MET metastatic sites")

    organotrop_arr = np.zeros(len(ordered_sites))
    met_freqs = freq_df[freq_df['Cancer Type']==cancer_type]

    for i, metastatic_site in enumerate(mapped_sites):
        organotrop_arr[i] = met_freqs[metastatic_site].item()

    return torch.tensor(organotrop_arr, dtype = torch.float32)

def load_pyclone_clusters(loci_fn, min_mut_thres=0):
    '''
    Returns (1) map of PyClone cluster ID to mutation names, 
    and (2) list of all mutation names used in PyClone clustering

    Filters clusters with less than min_mut_thres number of mutations
    '''

    # Load map from mutation name to PyClone cluster
    cluster_id_to_mut_names = {}
    cluster_df = pd.read_csv(loci_fn, sep="\t")
    for i, row in cluster_df.iterrows():
        cluster_id = row['cluster_id']
        mutation_id = row['mutation_id']
        if cluster_id not in cluster_id_to_mut_names:
            cluster_id_to_mut_names[cluster_id] = set()
        else:
            cluster_id_to_mut_names[cluster_id].add(mutation_id)
        
    # Filter clusters with less than min_mut_thres mutations
    final_cluster_id_to_mut_names = {}
    mutations = set()
    ix = 0
    for _, v in cluster_id_to_mut_names.items():
        if len(v) >= min_mut_thres:
            mutations.update(v)
            final_cluster_id_to_mut_names[ix] = list(v)
            ix += 1

    return final_cluster_id_to_mut_names, list(mutations)

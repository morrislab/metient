import csv
import numpy as np
import os
import torch
import copy
from collections import OrderedDict
import pandas as pd
import sys

# TODO: make more assertions on uniqueness and completeness of input csvs

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def write_pooled_tsv_from_clusters(df, mut_name_to_cluster_id, cluster_id_to_cluster_name,
                                   aggregation_rules, output_dir, patient_id):
    '''
    After clustering with any clustering algorithm, prepares tsvs by pooling mutations belonging 
    to the same cluster

    Args:
        df: pandas DataFrame with reqiured columns: [#sample_index, sample_label, 
        character_index, character_label, anatomical_site_index, anatomical_site_label, 
        ref, var]

        mut_name_to_cluster_id: dictionary mapping mutation name ('character_label' in input df)
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

    required_cols = ["#sample_index", "sample_label", "anatomical_site_index", "anatomical_site_label", 
                     "character_index", "character_label", "ref","var"]
    
    all_cols = required_cols + list(aggregation_rules.keys())

    if not (set(required_cols).issubset(df.columns)):
        raise ValueError(f"Input tsv needs required columns: {required_cols}")

    if not set(all_cols) == set(df.columns):
        missing_columns = set(df.columns) - set(all_cols)
        raise ValueError(f"Aggregation rules are required for all columns, missing rules for: {missing_columns}")

    df['cluster'] = df.apply(lambda row: mut_name_to_cluster_id[row['character_label']] if row['character_label'] in mut_name_to_cluster_id else np.nan, axis=1)
    df.dropna(subset=['cluster'])

    # Pool reference and variant allele counts from all mutations within a cluster
    pooled_df = df.drop(['character_label', 'character_index', '#sample_index', 'anatomical_site_index'], axis=1)

    ref_var_rules = {'ref': np.sum, 'var': np.sum, 'sample_label': lambda x: ';'.join(set(x))}

    pooled_df = pooled_df.groupby(['cluster', 'anatomical_site_label'], as_index=False).agg({**ref_var_rules, **aggregation_rules})
    
    pooled_df['character_label'] = pooled_df.apply(lambda row: cluster_id_to_cluster_name[row['cluster']], axis=1)
    
    # Add indices for mutations, samples and anatomical sites as needed for input format
    pooled_df['character_index'] = pooled_df.apply(lambda row: list(pooled_df['character_label'].unique()).index(row["character_label"]), axis=1)
    pooled_df['anatomical_site_index'] = pooled_df.apply(lambda row: list(pooled_df['anatomical_site_label'].unique()).index(row["anatomical_site_label"]), axis=1)
    pooled_df['#sample_index'] = pooled_df.apply(lambda row: list(pooled_df['sample_label'].unique()).index(row["sample_label"]), axis=1)    
    
    pooled_df = pooled_df[all_cols]
    output_fn = os.path.join(output_dir, f"{patient_id}_clustered_SNVs.tsv")
    pooled_df.to_csv(output_fn, sep="\t")

def write_pooled_tsv_from_pyclone_clusters(input_data_tsv_fn, clusters_tsv_fn, 
                                           aggregation_rules, output_dir, patient_id,
                                           cluster_sep=";"):

    '''
    After clustering with PyClone (see: https://github.com/Roth-Lab/pyclone),
    prepares tsvs by pooling mutations belonging to the same cluster

    Args:
        tsv_fn: path to tsv with reqiured columns: [#sample_index, sample_label, 
        character_index, character_label, anatomical_site_index, anatomical_site_label, 
        ref, var]

        clusters_tsv_fn: PyClone results tsv that maps each mutation to a cluster id
        
        aggregation_rules: dictionary indicating how to aggregate any extra columns that are not in 
        the required columns (specificied above). e.g. "first" aggregation rules can be used for columns 
        shared by rows with the same anatomical_site_label, since we are aggregating within an
        anatomical_site_label. 

        output_dir: where to save clustered tsv

        patient_id: name of patient used in tsv filename

        cluster_sep: string that separates names of mutations when creating a cluster name.
        e.g. cluster name for mutations ABC:4:3 and DEF:1:2 with cluster_sep=";" will be 
        "ABC:4:3;DEF:1:2"


    Outputs:

        Saves pooled tsv at {output_dir}/{patient_id}_clustered_SNVs.tsv

    '''

    df = pd.read_csv(input_data_tsv_fn, delimiter="\t", index_col=0)
    pyclone_df = pd.read_csv(clusters_tsv_fn, delimiter="\t")
    mut_name_to_cluster_id = dict()
    cluster_id_to_mut_names = dict()
    # 1. Get mapping between mutation names and PyClone cluster ids
    for _, row in df.iterrows():
        mut_items = row['character_label'].split(":")
        cluster_id = pyclone_df[(pyclone_df['CHR']==int(mut_items[1]))&(pyclone_df['POS']==int(mut_items[2]))&(pyclone_df['REF']==mut_items[3])]['treeCLUSTER'].unique()
        assert(len(cluster_id) <= 1)
        if len(cluster_id) == 1:
            cluster_id = cluster_id.item()
            mut_name_to_cluster_id[row['character_label']] = cluster_id
            if cluster_id not in cluster_id_to_mut_names:
                cluster_id_to_mut_names[cluster_id] = set()
            else:
                cluster_id_to_mut_names[cluster_id].add(row['character_label'])
    # 2. Set new names for clustered mutations
    cluster_id_to_cluster_name = {k:cluster_sep.join(list(v)) for k,v in cluster_id_to_mut_names.items()}

    # 3. Pool mutations and write to file
    write_pooled_tsv_from_clusters(df, mut_name_to_cluster_id, cluster_id_to_cluster_name,
                                   aggregation_rules, output_dir, patient_id)

def is_resolved_polytomy_cluster(cluster_label):
    '''
    In MACHINA simulated data, cluster labels with non-numeric components (e.g. M2_1
    instead of 1;3;4) represent polytomies
    '''
    is_polytomy = False
    for mut in cluster_label.split(";"):
        if not mut.isnumeric() and (mut.startswith('M') or mut.startswith('P')):
            is_polytomy = True
    return is_polytomy

def is_leaf(cluster_label):
    '''
    In MACHINA simulated data, cluster labels that have underscores (e.g. 3;4_M2)
    represent leaves
    '''
    return "_" in cluster_label and not is_resolved_polytomy_cluster(cluster_label)

def get_idx_to_cluster_label(cluster_filepath, ignore_polytomies):
    '''
    Args:
        cluster_filepath: path to cluster file for MACHINA simulated data in the format:
        0
        1
        3;15;17;22;24;29;32;34;53;56
        69;78;80;81

        where each semi-colon separated number represents a mutation that belongs
        to cluster i where i is the file line number.

        ignore_polytomies: whether to include resolved polytomies (which were found by
        running PMH-TR) in the returned dictionary

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
            if is_resolved_polytomy_cluster(label) and ignore_polytomies:
                continue
            idx_to_cluster_label[i] = label
            i += 1
    return idx_to_cluster_label

# TODO: remove polytomy stuff?
def get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_idx_to_cluster_label, T):
    '''
    tsv_filepath: path to tsv for machina simulated data (generated from create_conf_intervals_from_reads.py)

    tsv is expected to have columns: ['#sample_index', 'sample_label', 'anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'f_lb', 'f_ub', 'ref', 'var']

    pruned_idx_to_cluster_label:  dictionary mapping the cluster index to label, where 
    index corresponds to col index in the R matrix and V matrix returned. This isn't 1:1 
    with the 'character_label' to 'character_index' mapping in the tsv because we only keep the
    nodes which appear in the mutation tree, and re-index after removing unseen nodes
    (see _get_adj_matrix_from_machina_tree)

    T: adjacency matrix of the internal nodes.

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data
    '''

    assert(pruned_idx_to_cluster_label != None)
    assert(T != None)

    pruned_cluster_label_to_idx = {v:k for k,v in pruned_idx_to_cluster_label.items()}
    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        # Take a pass over the tsv to collect some metadata
        num_samples = 0 # S
        for i, row in enumerate(tsv):
            # Get the position of columns in the csvs
            if i == 3:
                sample_idx = row.index('#sample_index')
                site_label_idx = row.index('anatomical_site_label')
                cluster_label_idx = row.index('character_label')
                ref_idx = row.index('ref')
                var_idx = row.index('var')

            if i > 3:
                num_samples = max(num_samples, int(row[sample_idx]))
        # 0 indexing
        num_samples += 1

    num_clusters = len(pruned_cluster_label_to_idx.keys())

    R = np.zeros((num_samples, num_clusters))
    V = np.zeros((num_samples, num_clusters))
    unique_sites = []
    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        for i, row in enumerate(tsv):
            if i < 4: continue
            if row[cluster_label_idx] in pruned_cluster_label_to_idx:
                mut_cluster_idx = pruned_cluster_label_to_idx[row[cluster_label_idx]]
                R[int(row[sample_idx]), mut_cluster_idx] = int(row[ref_idx])
                V[int(row[sample_idx]), mut_cluster_idx] = int(row[var_idx])

            # collect additional metadata
            # doing this as a list instead of a set so we preserve the order
            # of the anatomical site labels in the same order as the sample indices
            if row[site_label_idx] not in unique_sites:
                unique_sites.append(row[site_label_idx])

    # Fill the columns in R and V with the resolved polytomies' parents data
    # (if there are resolved polytomies)
    for cluster_label in pruned_cluster_label_to_idx:
        if is_resolved_polytomy_cluster(cluster_label):
            res_polytomy_idx = pruned_cluster_label_to_idx[cluster_label]
            parent_idx = np.where(T[:,res_polytomy_idx] == 1)[0][0]
            R[:, res_polytomy_idx] = R[:, parent_idx]
            V[:, res_polytomy_idx] = V[:, parent_idx]

    return torch.tensor(R, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), list(unique_sites)

def shorten_cluster_names(idx_to_full_cluster_label, split_char):
    idx_to_cluster_label = dict()
    for ix in idx_to_full_cluster_label:
        og_label_muts = idx_to_full_cluster_label[ix].split(split_char) # e.g. CUL3:2:225371655:T;TRPM6:9:77431650:C
        idx_to_cluster_label[ix] = og_label_muts[0]
    return idx_to_cluster_label

def get_ref_var_matrices(tsv_filepaths, split_char=None):
    '''
    tsv_filepaths: List of paths to tsvs (one for each patient). Expects columns:

    ['#sample_index', 'sample_label', '#anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'ref', 'var']

    Returns:
    (1) list of R matrices (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) list of V matrices (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) list of list of unique anatomical sites from the patient's data,
    (4) list of dictionaries mapping index to character_label (based on input tsv, gives the index for each mutation name,
    where these indices are used in R matrix, V matrix

    For each of the returned lists, index 0 represents the patient at the first tsv_filepath, etc.
    '''
    if not isinstance(tsv_filepaths, list):
        return get_ref_var_matrix(tsv_filepaths)

    assert(len(tsv_filepaths) >= 1)

    ref_matrices, var_matrices, ordered_sites, idx_to_label_dicts = [], [], [], []

    for tsv_filepath in tsv_filepaths:
        ref, var, sites, idx_to_label_dict =  get_ref_var_matrix(tsv_filepath, split_char=split_char)
        ref_matrices.append(ref)
        var_matrices.append(var)
        ordered_sites.append(sites)
        idx_to_label_dicts.append(idx_to_label_dict)

    return ref_matrices, var_matrices, ordered_sites, idx_to_label_dicts 

def get_ref_var_matrix(tsv_filepath, split_char=None):
    '''
    tsv_filepath: path to tsv with columns:

    ['#sample_index', 'sample_label', '#anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'ref', 'var']

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data,
    (4) dictionary mapping index to character_label (based on input tsv, gives the index for each mutation name,
    where these indices are used in R matrix, V matrix
    '''

    # For tsvs with very large fields
    csv.field_size_limit(sys.maxsize)
    # Get metadata
    header_row_idx = -1
    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        # Take a pass over the tsv to collect some metadata
        num_clusters = 0
        num_samples = 0

        for i, row in enumerate(tsv):
            if '#sample_index' in row: # is header row
                header_row_idx = i
                sample_idx = row.index('#sample_index')
                ref_idx = row.index('ref')
                var_idx = row.index('var')
                mut_cluster_idx = row.index('character_index')
                site_label_idx = row.index('anatomical_site_label')
                character_label_idx = row.index('character_label')
                break

        assert (header_row_idx != -1), "No header provided in csv file"

        for i, row in enumerate(tsv):
            num_clusters = max(num_clusters, int(row[mut_cluster_idx]))
            num_samples = max(num_samples, int(row[sample_idx]))
        # 0 indexing
        num_clusters += 1
        num_samples += 1


    # Build R and V matrices
    character_label_to_idx = dict()
    R = np.zeros((num_samples, num_clusters))
    V = np.zeros((num_samples, num_clusters))
    unique_sites = []
    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        for i, row in enumerate(tsv):
            if i <= header_row_idx: continue
            R[int(float(row[sample_idx])), int(float(row[mut_cluster_idx]))] = int(float(row[ref_idx]))
            V[int(float(row[sample_idx])), int(float(row[mut_cluster_idx]))] = int(float(row[var_idx]))

            # collect additional metadata
            # doing this as a list instead of a set so we preserve the order
            # of the anatomical site labels in the same order as the sample indices
            if row[site_label_idx] not in unique_sites:
                unique_sites.append(row[site_label_idx])
            if row[character_label_idx] not in character_label_to_idx:
                character_label_to_idx[row[character_label_idx]] = int(row[mut_cluster_idx])

    idx_to_character_label = {v:k for k,v in character_label_to_idx.items()}
    if split_char != None:
        idx_to_character_label = shorten_cluster_names(idx_to_character_label, split_char)
    return torch.tensor(R, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), list(unique_sites), idx_to_character_label

def _get_adj_matrix_from_machina_tree(tree_edges, idx_to_character_label, remove_unseen_nodes=True, skip_polytomies=False):
    '''
    Args:
        tree_edges: list of tuples where each tuple is an edge in the tree
        idx_to_character_label: dictionary mapping index to character_label (machina
        uses colors to represent subclones, so this would map n to 'pink', if pink
        is the nth node in the adjacency matrix).
        remove_unseen_nodes: if True, removes nodes that
        appear in the machina tsv file but do not appear in the reported tree
        skip_polyomies: if True, checks for polytomies and skips over them. For example
        if the tree is 0 -> polytomy -> 1, returns 0 -> 1. If the tree is 0 -> polytomy
        returns 0.

    Returns:
        T: adjacency matrix where Tij = 1 if there is a path from i to j
        idx_to_character_label: a pruned idx_to_character_label where nodes that
        appear in the machina tsv file but do not appear in the reported tree are removed
    '''
    character_label_to_idx = {v:k for k,v in idx_to_character_label.items()}
    num_internal_nodes = len(character_label_to_idx)
    T = np.zeros((num_internal_nodes, num_internal_nodes))
    seen_nodes = set()
    # dict of { child_label : parent_label } needed to skip over polytomies
    child_to_parent_map = {}
    for edge in tree_edges:
        node_i, node_j = edge[0], edge[1]
        seen_nodes.add(node_i)
        seen_nodes.add(node_j)
        # don't include the leaf/extant nodes (determined from U)
        if node_i in character_label_to_idx and node_j in character_label_to_idx:
            T[character_label_to_idx[node_i], character_label_to_idx[node_j]] = 1

        # we don't want to include the leaf nodes, only internal nodes
        if not is_leaf(node_j) and node_i != "GL":
            child_to_parent_map[node_j] = node_i

    # Fix missing connections
    if skip_polytomies:
        for child_label in child_to_parent_map:
            parent_label = child_to_parent_map[child_label]
            if is_resolved_polytomy_cluster(parent_label) and parent_label in child_to_parent_map:
                # Connect the resolved polytomy's parent to the resolved polytomy's child
                res_poly_parent = child_to_parent_map[parent_label]
                if res_poly_parent in character_label_to_idx and child_label in character_label_to_idx:
                    T[character_label_to_idx[res_poly_parent], character_label_to_idx[child_label]] = 1

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
                adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
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

        adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
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

# TODO: take out skip polytomies functionality?
def get_adj_matrix_from_machina_tree(character_label_to_idx, tree_filename, remove_unseen_nodes=True, skip_polytomies=False):
    '''
    character_label_to_idx: dictionary mapping character_label to index (machina
    uses colors to represent subclones, so this would map 'pink' to n, if pink
    is the nth node in the adjacency matrix).
    tree_filename: path to .tree file
    remove_unseen_nodes: if True, removes nodes that
    appear in the machina tsv file but do not appear in the reported tree
    skip_polyomies: if True, checks for polytomies and skips over them. For example
    if the tree is 0 -> polytomy -> 1, returns 0 -> 1. If the tree is 0 -> polytomy
    returns 0.

    Returns:
        T: adjacency matrix where Tij = 1 if there is a path from i to j
        character_label_to_idx: a pruned character_label_to_idx where nodes that
        appear in the machina tsv file but do not appear in the reported tree are removed
    '''
    edges = []
    with open(tree_filename, 'r') as f:
        for line in f:
            nodes = line.strip().split()
            node_i, node_j = nodes[0], nodes[1]
            edges.append((node_i, node_j))
    return _get_adj_matrix_from_machina_tree(edges, character_label_to_idx, remove_unseen_nodes, skip_polytomies)

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

def get_genetic_distance_matrix_from_adj_matrix(adj_matrix, idx_to_character_label, split_char, normalize=True):
    '''
    Get the genetic distances between nodes by counting the number of mutations between
    parent and child. idx_to_character_label's keys are expected to be mutation/cluster indices with
    the mutation or mutations in that cluster (e.g. 'ENAM:4:71507837_DLG1:3:196793590'). split_char
    indicates what the mutations in the cluster name are split by (if it's a cluster).
    '''
    G = np.zeros(adj_matrix.shape)

    for i, adj_row in enumerate(adj_matrix):
        for j, val in enumerate(adj_row):
            if val == 1:
                # This is the number of mutations the child node has accumulated compared to its parent
                if split_char == None: # not mutation clusters, just single mutations
                    num_mutations = 1
                else:
                    num_mutations = idx_to_character_label[j].count(split_char) + 1
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

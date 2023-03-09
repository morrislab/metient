import csv
import numpy as np
import os
import torch
import copy
from collections import OrderedDict
import pandas as pd

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


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

def get_cluster_label_to_idx(cluster_filepath, ignore_polytomies):
    '''
    cluster_filepath: path to cluster file for MACHINA simulated data in the format:
    0
    1
    3;15;17;22;24;29;32;34;53;56
    69;78;80;81

    where each semi-colon separated number represents a mutation that belongs
    to cluster i where i is the file line number.

    ignore_polytomies: whether to include resolved polytomies (which were found by
    running PMH-TR) in the returned dictionary

    returns
    (1) a dictionary mapping cluster name to cluster number
    for e.g. for the file above, this would return:
        {'0': 0, '1': 1, '3;15;17;22;24;29;32;34;53;56': 2, '69;78;80;81': 3}
    '''
    cluster_label_to_idx = OrderedDict()
    with open(cluster_filepath) as f:
        i = 0
        for line in f:
            label = line.strip()
            if is_resolved_polytomy_cluster(label) and ignore_polytomies:
                continue
            cluster_label_to_idx[label] = i
            i += 1
    return cluster_label_to_idx

# TODO: remove polytomy stuff?
def get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_cluster_label_to_idx, T):
    '''
    tsv_filepath: path to tsv for machina simulated data (generated from create_conf_intervals_from_reads.py)

    tsv is expected to have columns: ['#sample_index', 'sample_label', '#anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'f_lb', 'f_ub', 'ref', 'var']

    pruned_cluster_label_to_idx:  dictionary mapping the cluster label to index which corresponds to
    col index in the R matrix and V matrix returned. This isn't 1:1 with the
    'character_label' to 'character_index' mapping in the tsv because we only keep the
    nodes which appear in the mutation tree, and re-index after removing unseen nodes
    (see _get_adj_matrix_from_machina_tree)

    T: adjacency matrix of the internal nodes.

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data,
    # (4) dictionary mapping character_label to index (machina uses colors to represent
    # subclones, so this would map 'pink' to n, if pink is the nth node in the adjacency matrix)
    '''

    assert(pruned_cluster_label_to_idx != None)
    assert(T != None)

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

def get_ref_var_matrices_from_real_data(tsv_filepath):
    '''
    tsv_filepath: path to tsv for hoadley tsv data

    tsv is expected to have columns: ['#sample_index', 'sample_label', '#anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'ref', 'var']

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data,
    (4) dictionary mapping character_label to index (machina uses diff labels for each dataset to represent
    mutation clusters, so this would map 'pink' to n, if pink is the nth node in the adjacency matrix)
    '''

    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        # Take a pass over the tsv to collect some metadata
        num_clusters = 0
        num_samples = 0
        for i, row in enumerate(tsv):
            if i == 3:
                sample_idx = row.index('#sample_index')
                ref_idx = row.index('ref')
                var_idx = row.index('var')
                mut_cluster_idx = row.index('character_index')
                site_label_idx = row.index('anatomical_site_label')
                character_label_idx = row.index('character_label')

            if i > 3:
                num_clusters = max(num_clusters, int(row[mut_cluster_idx]))
                num_samples = max(num_samples, int(row[sample_idx]))
        # 0 indexing
        num_clusters += 1
        num_samples += 1

    character_label_to_idx = dict()
    R = np.zeros((num_samples, num_clusters))
    V = np.zeros((num_samples, num_clusters))
    unique_sites = []
    with open(tsv_filepath) as f:
        tsv = csv.reader(f, delimiter="\t", quotechar='"')
        for i, row in enumerate(tsv):
            if i < 4: continue
            R[int(row[sample_idx]), int(row[mut_cluster_idx])] = int(row[ref_idx])
            V[int(row[sample_idx]), int(row[mut_cluster_idx])] = int(row[var_idx])

            # collect additional metadata
            # doing this as a list instead of a set so we preserve the order
            # of the anatomical site labels in the same order as the sample indices
            if row[site_label_idx] not in unique_sites:
                unique_sites.append(row[site_label_idx])
            if row[character_label_idx] not in character_label_to_idx:
                character_label_to_idx[row[character_label_idx]] = int(row[mut_cluster_idx])

    return torch.tensor(R, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), list(unique_sites), character_label_to_idx

def _get_adj_matrix_from_machina_tree(tree_edges, character_label_to_idx, remove_unseen_nodes=True, skip_polytomies=False):
    '''
    Args:
        tree_edges: list of tuples where each tuple is an edge in the tree
        character_label_to_idx: dictionary mapping character_label to index (machina
        uses colors to represent subclones, so this would map 'pink' to n, if pink
        is the nth node in the adjacency matrix).
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

    return T, pruned_character_label_to_idx if remove_unseen_nodes else character_label_to_idx

def get_adj_matrices_from_all_mutation_trees(mut_trees_filename, character_label_to_idx, is_sim_data=False):
    '''
    When running MACHINA's generatemutationtrees executable, it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of tuples, each containing (T, character_label_to_idx) for each
    tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - character_label_to_idx: a pruned character_label_to_idx where nodes that
        appear in the machina tsv file but do not appear in the reported tree are removed
    '''

    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_data = []
        for i, line in enumerate(f):
            if i < 3: continue
            # This marks the beginning of a tree
            if "#edges, tree" in line:
                adj_matrix, pruned_char_label_to_idx = _get_adj_matrix_from_machina_tree(tree_data, character_label_to_idx)
                out.append((adj_matrix, pruned_char_label_to_idx))
                tree_data = []
            else:
                nodes = line.strip().split()
                # fixes incompatibility with naming b/w cluster file (uses ";" separator)
                # and the ./generatemutationtrees output (uses "_" separator)
                if is_sim_data:
                    tree_data.append((";".join(nodes[0].split("_")), ";".join(nodes[1].split("_"))))
                else:
                    tree_data.append((nodes[0], nodes[1]))

        adj_matrix, pruned_char_label_to_idx = _get_adj_matrix_from_machina_tree(tree_data, character_label_to_idx)
        out.append((adj_matrix, pruned_char_label_to_idx))
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

def get_genetic_distance_tensor_from_sim_adj_matrix(adj_matrix, character_label_to_idx, split_char):
    '''
    Get the genetic distances between nodes by counting the number of mutations between
    parent and child. character_label_to_idx's keys are expected to be cluster names with
    the mutations in that cluster (e.g. 'ENAM:4:71507837_DLG1:3:196793590'). split_char
    indicates what the mutations in the cluster name are split by.
    '''

    G = np.zeros(adj_matrix.shape)
    idx_to_char_label = { v:k for k,v in character_label_to_idx.items() }

    for i, adj_row in enumerate(adj_matrix):
        for j, val in enumerate(adj_row):
            if val == 1:
                # This is the number of mutations the child node has accumulated compared to its parent
                num_mutations = idx_to_char_label[j].count(split_char) + 1
                G[i][j] = num_mutations
    return torch.tensor(G, dtype = torch.float32)


def get_organotropism_matrix(ordered_sites, site_to_msk_met_map, msk_met_fn):
    '''
    Args:
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        site_to_msk_met_map: dictionary mapping site names to MSK-MET site names

    Returns:
        matrix of size len(ordered_sites) x len(ordered_sites) with the frequency with which
        site i seeds site j (as taken from MSK-MET data)
    '''

    freq_df = pd.read_csv(msk_met_fn)

    organotrop_mat = np.zeros((len(ordered_sites), len(ordered_sites)))

    # TODO: handle site i and site j differently since those are diff labels in MSK-MET
    for site in ordered_sites:
        assert(site in site_to_msk_met_map)

    mapped_sites = [site_to_msk_met_map[site] for site in ordered_sites]
    for i, start in enumerate(mapped_sites):
        for j, dest in enumerate(mapped_sites):
            missing_val = False
            if start not in list(freq_df['Primary Tumor Site']):
                print(f"{start} not in MSK-MET as primary tumor")
                missing_val = True
            if dest not in list(freq_df.columns):
                print(f"{dest} not in MSK-MET as metastatic site")
                missing_val = True
            if missing_val or (freq_df[freq_df['Primary Tumor Site']==start][dest].item() == 0.0):
                organotrop_mat[i,j] = -1.0 # TODO: what do we do when primary or met are not in db
            else:
                organotrop_mat[i,j] = freq_df[freq_df['Primary Tumor Site']==start][dest].item()

    return torch.tensor(organotrop_mat, dtype = torch.float32)

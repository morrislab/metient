import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from networkx.drawing.nx_pydot import to_pydot
from PIL import Image as PILImage
from IPython.display import Image, display
import io
import matplotlib.gridspec as gridspec
import math
import matplotlib.font_manager
from matplotlib import rcParams
import os
import pickle
import pygraphviz as pgv
from collections import deque
import gzip
import re
import copy
import tracemalloc

import metient.util.vertex_labeling_util as vutil 
import metient.util.data_extraction_util as dutil
from metient.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

FONT = "Arial"

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def is_cyclic(G):
    """
    returns True if graph contains cycles
    """
    num_nodes = G.size(0)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    stack = torch.zeros(num_nodes, dtype=torch.bool)

    def dfs(node):
        visited[node] = True
        stack[node] = True

        for neighbor in range(num_nodes):
            if G[node, neighbor] >= 1:
                if not visited[neighbor]:
                    if dfs(neighbor):
                        return True
                elif stack[neighbor]:
                    return True

        stack[node] = False
        return False

    for node in range(num_nodes):
        if not visited[node]:
            if dfs(node):
                return True

    return False

def site_clonality_with_G(G):
    """
    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    """
    if torch.all(G == 0):
        return "n/a"
    return "polyclonal" if ((G > 1).any()) else  "monoclonal"

def site_clonality(V, A):
    """
    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    """
    V, A = prep_V_A_inputs(V, A)
    G = migration_graph(V, A)
    return site_clonality_with_G(G)

def genetic_clonality(V, A, idx_to_label):
    """
    Returns monoclonal if every site is seeded by the *same* clone,
    else returns polyclonal.
    """
    V, A = prep_V_A_inputs(V, A)
    all_seeding_clusters = seeding_clusters(V, A, idx_to_label)
    if len(all_seeding_clusters) == 0:
        return "n/a"
    monoclonal = True if len(all_seeding_clusters) == 1 else False
    return "monoclonal" if monoclonal else "polyclonal"

def seeding_pattern_with_G(G):

    # Determine if single-source seeding (all incoming edges to a site in G 
    # originate from the same site) OR multi-source seeding (at least one site is 
    # seeded from multiple other sites) OR (R) reseeding (at least one site seeds 
    # its originating site)
    non_zero = torch.where(G > 0)
    source_sites = non_zero[0]
    binarized_G = (G != 0).to(torch.int)

    col_sums = torch.sum(binarized_G, axis=0)
    # single-source means that each site is only seeded by ONE other site 
    # (not that seeding site is 1)
    is_single_source = torch.all(col_sums <= 1).item()
    unique_source_sites = torch.unique(source_sites)

    if len(unique_source_sites) == 0:
        return "no seeding"
    elif is_cyclic(G):
        return "reseeding"
    elif len(unique_source_sites) == 1:
        return "primary single-source"
    elif is_single_source:
        return "single-source"
    return "multi-source"

def seeding_pattern(V, A):
    """
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    returns: verbal description of the seeding pattern, one of:
    {primary single-source, single-source, multi-source, reseeding}
    """
    G = migration_graph(V, A)
    return seeding_pattern_with_G(G)
    

def remove_migration_edges_to_sites_sparse(Y, V, sites_to_keep):
    Y = Y.coalesce()
    indices = Y.indices()
    values = Y.values()

    # Get destination sites for all edges in one operation
    dest_sites = torch.argmax(V[:, indices[1]], dim=0)
    
    # Create mask for edges to keep
    sites_tensor = torch.tensor(list(sites_to_keep), device=dest_sites.device)
    keep_mask = torch.isin(dest_sites, sites_tensor)
    
    # Filter indices and values in one step
    Y = torch.sparse_coo_tensor(indices[:, keep_mask], values[keep_mask], Y.size())
    return Y

def migration_edges(V, A, sites=None):
    """
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    sites: optional, set of indices in range [0,num_sites) that you restrict migration edges *to*
    Returns:
        Returns a matrix where Yij = 1 if there is a migration edge from node i to node j
    """
    X = V.T @ V 
    Y = torch.mul(A, (1-X))
    
    # Remove migration edges to sites that we're not interested in 
    if sites != None:
        if Y.is_sparse:
            return remove_migration_edges_to_sites_sparse(Y, V, sites)
        for i,j in vutil.tree_iterator(Y):
            k = (V[:,j] == 1).nonzero()[0][0].item()
            if k not in sites:
                Y[i,j] = 0
    return Y

def seeding_cluster_sparse(Y, A, node_idx_to_label, sites=None):
    """
    returns: list of nodes whose child is a different color
    """

    Y = Y.coalesce()
    nonzero_indices = Y.indices()

    # Select the indices where Y is not 0
    seeding_clusters = nonzero_indices[1][Y.values() != 0]
    # seeding_clusters = (Y == 1).nonzero(as_tuple=True)[1]
    # Check if it's a scalar (0D tensor)
    if seeding_clusters.dim() == 0:
        # Convert to a 1D tensor (vector)
        seeding_clusters = seeding_clusters.unsqueeze(0)
    seeding_clusters = [int(x) for x in seeding_clusters]

    if isinstance(node_idx_to_label, vutil.MigrationHistoryNodeCollection):
        node_collection = node_idx_to_label
    else:
        # TODO this is way too slow on large trees
        node_collection = vutil.MigrationHistoryNodeCollection.from_dict(node_idx_to_label)

    # Special case for if the node w/ diff color parent is a witness node or polytomy resolver node,
    # the seeding cluster is the parent node (since no new mutations)
    all_seeding_clusters = set()
    for x in seeding_clusters:
        node = node_collection.get_node(x)
        if node.is_witness or node.is_polytomy_resolver_node:
            all_seeding_clusters.add(vutil.get_parent(A, node.idx))
        else:
            all_seeding_clusters.add(x)
    return list(all_seeding_clusters)

def seeding_clusters(V, A, node_idx_to_label, sites=None):
    """
    returns: list of nodes whose parent is a different color
    """
    V, A = prep_V_A_inputs(V, A)
    Y = migration_edges(V,A, sites)

    if Y.is_sparse:
        return seeding_cluster_sparse(Y, A, node_idx_to_label, sites)
    
    seeding_clusters = torch.nonzero(Y.any(dim=1)).squeeze()
    # Check if it's a scalar (0D tensor)
    if seeding_clusters.dim() == 0:
        # Convert to a 1D tensor (vector)
        seeding_clusters = seeding_clusters.unsqueeze(0)
    seeding_clusters = [int(x) for x in seeding_clusters]

    return seeding_clusters
        
def find_tree_trunk(adj_matrix):
    n = len(adj_matrix)  # Number of nodes in the matrix
    root = vutil.get_root_index(adj_matrix)  # Assuming the root node is 0
    
    # Function to find children of a given node
    def children(node):
        return [i for i in range(n) if adj_matrix[node][i] == 1]
    
    # Start from the root
    current_node = root
    trunk = [current_node]  # Initialize the trunk with the root node

    # Continue until a node has more than one child
    while True:
        children = children(current_node)
        if len(children) != 1:  # More than one child or no children
            break
        current_node = children[0]  # Move to the next node in the trunk
        trunk.append(current_node)
    
    return trunk

def is_valid_path(path, S):
    return all(node in path for node in S)

def mrca(adj_matrix, nodes_to_check):
    """
    Gets the most recent common ancestor of nodes in nodes_to_check
    """
    start_node = vutil.get_root_index(adj_matrix)
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes

    queue = deque()
    queue.append(start_node)
    visited[start_node] = True

    while queue:
        current_node = queue.popleft()
        if current_node in nodes_to_check:
            return current_node

        for neighbor, connected in enumerate(adj_matrix[current_node]):
            if connected and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                
def hamiltonian_paths(adj_matrix, path, visited, n, S):
    if is_valid_path(path, S):
        return True

    current_node = path[-1]
    for next_node in range(n):
        if adj_matrix[current_node][next_node] == 1 and not visited[next_node]:
            visited[next_node] = True
            path.append(next_node)
            
            if hamiltonian_paths(adj_matrix, path, visited, n, S):
                return True

            path.pop()
            visited[next_node] = False
            
    return False

def has_hamiltonian_path_with_set(adj_matrix, nodes_to_check):
    n = len(adj_matrix)
    highest_node = mrca(adj_matrix, nodes_to_check)
    visited = [False] * n
    path = [highest_node]
    visited[highest_node] = True

    if hamiltonian_paths(adj_matrix, path, visited, n, nodes_to_check):
        return True
    
    return False
                 
def phyleticity(V, A, idx_to_label, sites=None):
    """
    If all nodes can be reached from the top level node in the seeding clusters,
    returns monophyletic, else polyphyletic
    """
    
    V, A = prep_V_A_inputs(V, A)
    clonality = genetic_clonality(V, A, idx_to_label)
    all_seeding_clusters = seeding_clusters(V, A, idx_to_label, sites)
    
    if len(all_seeding_clusters) == 0: # no seeding
        return "n/a"
    
    if "monoclonal" in clonality:
        return "monophyletic"

    num_nodes = len(A)
    visited = [False] * num_nodes
    highest_node = mrca(A, all_seeding_clusters)
    
    # Do a single DFS from highest_node to build reachable set
    num_nodes = len(A)
    visited = [False] * num_nodes
    reachable = set()
    
    # Stack-based DFS is faster than recursive
    stack = [highest_node]
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            reachable.add(node)
            # Add unvisited neighbors to stack
            for neighbor, connected in enumerate(A[node]):
                if connected and not visited[neighbor]:
                    stack.append(neighbor)
    
    # Check if all seeding clusters are reachable
    if all(node in reachable for node in all_seeding_clusters):
        return "monophyletic"
    return "polyphyletic"

def tracerx_phyleticity(V, A, idx_to_label):
    """
    Looking at the seeding clones, is there a single path that connects all seeding clusters
    or multiple possible paths? If singular path, monophyletic, if not, polyphyletic

    This is to implement TRACERx's definition of phyleticity:
    "the origin of the seeding clusters was determined as monophyletic if all 
    clusters appear along a single branch, and polyphyletic if clusters were
    spread across multiple branches of the phylogenetic tree. Thus, if a 
    metastasis was defined as monoclonal, the origin was necessarily monophyletic. 
    For polyclonal metastases, the clusters were mapped to branches of the 
    evolutionary tree. If multiple branches were found, the origin was determined 
    to be polyphyletic, whereas, if only a single branch gave rise to all shared 
    clusters, the origin was defined as monophyletic."
    (https://www.nature.com/articles/s41586-023-05729-x#Sec7)
    """
    V, A = prep_V_A_inputs(V, A)
    clonality = genetic_clonality(V, A, idx_to_label)
    all_seeding_clusters = seeding_clusters(V, A, idx_to_label)

    is_hamiltonian = has_hamiltonian_path_with_set(A, all_seeding_clusters)
    if "monoclonal" in clonality:
        return "monophyletic"
    phyleticity = "monophyletic" if is_hamiltonian else "polyphyletic"
    return phyleticity
    
def tracerx_seeding_pattern(V, A, idx_to_label):
    """
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Monoclonal if only one clone seeds met(s), else polyclonal
    Monophyletic if there is a Hamiltonian path connecting all seeding clusters
    returns: one of {no seeding, monoclonal monophyletic, polyclonal polyphyletic, polyclonal monophyletic}
    """
    V, A = prep_V_A_inputs(V, A)
    G = migration_graph(V, A)
    non_zero = torch.where(G > 0)
    source_sites = non_zero[0]
    if len(torch.unique(source_sites)) == 0:
        return "no seeding"

    # 1) determine if monoclonal (only one clone seeds met(s)), else polyclonal
    clonality = genetic_clonality(V,A,idx_to_label)
    
    # 2) determine if monophyletic or polyphyletic
    phyleticity = tracerx_phyleticity(V, A,idx_to_label)

    return clonality + phyleticity

def min_max_normalize(x):
    x = np.array(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def losses_to_probabilities(messy_losses, temperature=0.5):
    """
    Converts a messy list of losses (in NumPy arrays) to probabilities using the softmax function
    with temperature scaling.

    Args:
    - messy_losses (list): List of NumPy arrays or scalars containing loss values.
    - temperature (float): Temperature parameter for softmax. Lower = sharper distribution.

    Returns:
    - probabilities (numpy array): Probabilities corresponding to each loss.
    """
    # Flatten and extract numerical values from the messy list
    cleaned_losses = np.array([loss.flatten() if hasattr(loss, 'flatten') else loss for loss in messy_losses])
    neg_losses = -np.array([loss.item() if hasattr(loss, 'item') else loss for loss in cleaned_losses])  # Ensure scalar
    neg_losses = min_max_normalize(neg_losses)
    # Apply temperature scaling
    scaled_losses = neg_losses / temperature

    # Subtract the max to prevent overflow
    max_scaled_loss = np.max(scaled_losses)
    exp_scaled = np.exp(scaled_losses - max_scaled_loss)

    # Normalize to get probabilities
    probabilities = exp_scaled / np.sum(exp_scaled)

    return probabilities
    
def get_soln_probabilities(loss_dicts):
    """
    Calculate probability weights for each solution based on loss values.
    
    Args:
        loss_dicts (list[dict]): List of dictionaries containing loss values and metrics for each solution
        thetas (list[float] | None, optional): List of 3 theta parameters for weighting migration, comigration, 
            and seeding site metrics. If None, uses raw loss values. Defaults to None.
            
    Returns:
        np.ndarray: List of probabilities for each solution
    """
    # Calculate losses for each solution
    losses = []
    for loss_dict in loss_dicts:
        losses.append(loss_dict[FULL_LOSS_KEY])
    
    probabilities = losses_to_probabilities(losses)
    return probabilities

def weighted_classification(losses, classifications):
    """
    Performs weighted classification by summing probabilities for each class.

    Args:
        losses (list[float]): List of loss values/weights for each classification
        classifications (list[str]): List of classification labels corresponding to each loss

    Returns:
        str: The classification label with the highest total weighted probability
    """
    # Initialize a dictionary to hold the weighted sum of probabilities for each class
    weighted_probs = {c: 0 for c in classifications}

    # Sum the weighted probabilities for each attribute
    for loss, c in zip(losses, classifications):
        weighted_probs[c] += loss
    
    # Return the classification with the highest total weight
    return max(weighted_probs, key=weighted_probs.get)

def _get_weighted_classification_data(pkl):
    """Helper function to extract and prepare data for weighted classification.

    Args:
        pkl (dict): Pickle file containing tree data

    Returns:
        tuple: (probabilities, Vs, As, node_infos) containing solution probabilities and tree data
    """
    # Get data from pickle
    loss_dicts = pkl[OUT_LOSS_DICT_KEY]
    parents = pkl[OUT_PARENTS_KEY]
    As = [dutil.adjacency_matrix_from_parents(p) for p in parents]
    Vs = pkl[OUT_LABElING_KEY]
    node_infos = [vutil.MigrationHistoryNodeCollection.from_dict(x) for x in pkl[OUT_IDX_LABEL_KEY]]

    probabilities = get_soln_probabilities(loss_dicts)
    
    return probabilities, Vs, As, node_infos

def compute_depths(A,P):
    """
    Returns:
    dict: Dictionary where keys are nodes and values are their depth in the tree.
    """
    root = vutil.get_root_index(A)
    n = P.shape[0]
    depths = {}
    
    for node in range(n):
        if node == root:
            depths[node] = 0  # Root has depth 0
        else:
            # The depth is the number of ancestors (nodes that can reach this node)
            depths[node] = torch.sum(P[:, node],dim=0)
    
    return depths

def fast_phyleticity(V, A, P, node_info, depths, sites=None):
    '''
    If all nodes can be reached from the top level node in the seeding clusters,
    returns monophyletic, else polyphyletic
    '''
    
    V, A = prep_V_A_inputs(V, A)
    all_seeding_clusters = seeding_clusters(V, A, node_info, sites)

    if len(all_seeding_clusters) == 0: # no seeding
        return "no seeding"
    
    # Get the seeding node that is closest to the root
    highest_node = min(all_seeding_clusters, key=lambda num: depths.get(num, float('-inf')))

    # Check if all nodes can be reached from the top level node in the seeding
    # nodes (seeding node that is closest to the root)
    for node in all_seeding_clusters:
        if not P[highest_node, node] == 1:
            return "polyphyletic"
    return "monophyletic"

def check_equal_trees(As):
    """Check if all adjacency matrices in a list are equal.
    
    Args:
        As (list): List of adjacency matrices (torch tensors) to compare

    Returns:
        bool: True if all matrices are equal, False otherwise
    """
    if len(As) <= 1:
        return True
    first_A = As[0]
    for A in As[1:]:
        if not vutil.sparse_tensors_equal(first_A, A):
            return False
    return True

def weighted_phyleticity(pkl, sites):
    """
    Calculate weighted phyleticity classification across all solutions.

    Args:
        pkl (dict): Pickle file containing tree data with labeling matrices, adjacency matrices, 
                   loss dictionaries and node information
        sites (list[str] | None, optional): List of anatomical site names to restrict phyleticity 
        calculation to. Defaults to None.

    Returns:
        str: Weighted phyleticity classification ('monophyletic' or 'polyphyletic') based on 
             solution probabilities
    """
    probabilities, Vs, As, node_infos = _get_weighted_classification_data(pkl)

    # If all trees are the same, we can use the fast phyleticity method
    equal_trees = check_equal_trees(As)
    if equal_trees:
        P = vutil.path_matrix(As[0], remove_self_loops=False).to_dense()
        depths = compute_depths(As[0], P)

    phyleticities = []
    for i in range(len(Vs)):
        if equal_trees:
            phyleticities.append(fast_phyleticity(Vs[i], As[i], P, node_infos[i], depths, sites))
        else:
            phyleticities.append(phyleticity(Vs[i], As[i], node_infos[i], sites))
    
    return weighted_classification(probabilities, phyleticities)

def weighted_genetic_clonality(pkl):
    """
    Calculate weighted genetic clonality classification across all solutions.

    Args:
        pkl (dict): Pickle file containing tree data with labeling matrices, adjacency matrices, 
                   loss dictionaries and node information
    Returns:
        str: Weighted genetic clonality classification ('monoclonal' or 'polyclonal') based on 
             solution probabilities
    """
    probabilities, Vs, As, node_infos = _get_weighted_classification_data(pkl)
    
    clonalities = []
    for V, A, node_info in zip(Vs, As, node_infos):
        clonalities.append(genetic_clonality(V, A, node_info))
    
    return weighted_classification(probabilities, clonalities)

def weighted_site_clonality(pkl):
    """
    Calculate weighted site clonality classification across all solutions.

    Args:
        pkl (dict): Pickle file containing tree data with labeling matrices, adjacency matrices, 
                   loss dictionaries and node information
    Returns:
        str: Weighted site clonality classification ('monoclonal' or 'polyclonal') based on 
             solution probabilities
    """
    probabilities, Vs, As, _ = _get_weighted_classification_data(pkl)
    
    clonalities = []
    for V, A in zip(Vs, As):
        clonalities.append(site_clonality(V, A))
    
    return weighted_classification(probabilities, clonalities)

def weighted_seeding_pattern(pkl):
    """
    Calculate weighted seeding pattern classification across all solutions.

    Args:
        pkl (dict): Pickle file containing tree data with labeling matrices, adjacency matrices, 
                   loss dictionaries and node information
    Returns:
        str: Weighted seeding pattern classification based on solution probabilities
    """
    probabilities, Vs, As, _ = _get_weighted_classification_data(pkl)
    
    patterns = []
    for V, A in zip(Vs, As):
        patterns.append(seeding_pattern(V, A))
    
    return weighted_classification(probabilities, patterns)

def write_tree(tree_edge_list, output_filename, add_germline_node=False):
    """
    Writes the full tree to file like so:
    0 1
    1 2;3
    """
    if add_germline_node:
        tree_edge_list.append(('GL', tree_edge_list[0][0]))
    with open(output_filename, 'w') as f:
        for edge in tree_edge_list:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")

def write_tree_vertex_labeling(vertex_name_to_site_map, output_filename, add_germline_node=False):
    """
    Writes the full tree's vertex labeling to file like so:
    1 P
    1_P P
    25;32_M1 M1
    """
    if add_germline_node:
        vertex_name_to_site_map['GL'] = "P"
    with open(output_filename, 'w') as f:
        for vert_label in vertex_name_to_site_map:
            f.write(f"{vert_label} {vertex_name_to_site_map[vert_label]}")
            f.write("\n")

def write_migration_graph(migration_edge_list, output_filename):
    """
    Writes the full migration graph to file like so:
    P M1
    P M2
    P M1
    M1 M2
    """
    with open(output_filename, 'w') as f:
        for edge in migration_edge_list:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")

def plot_losses(losses):
    plt.plot([x for x in range(len(losses))],losses, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def plot_temps(temps):
    plt.plot([x for x in range(len(temps))],temps, label="temp")
    plt.xlabel("epoch")
    plt.ylabel("temp")
    plt.show()

def plot_loss_components(loss_dicts, weights):
    # if else statements to handle lr schedules where we do not calculate
    # all loss components at every epoch

    mig_losses = [e[MIG_KEY] for e in loss_dicts]
    seed_losses = [e[SEEDING_KEY] for e in loss_dicts]
    neg_entropy = [e[ENTROPY_KEY] for e in loss_dicts]

    plt.plot([x for x in range(len(loss_dicts))],mig_losses, label="m")
    plt.plot([x for x in range(len(loss_dicts))],seed_losses, label="s")
    plt.plot([x for x in range(len(loss_dicts))],neg_entropy, label="neg. ent.")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.show()

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

def contains_delim(s, delims):
    for delim in delims:
        if delim in s:
            return True
    return False

def pruned_mut_label(mut_names, shorten_label, to_string):
    if not shorten_label and not to_string:
        return ([str(m) for m in mut_names])
    elif not shorten_label and to_string:
        return ";".join([str(m) for m in mut_names])
    # If mutation name contains :, ;, _ (e.g. LOC1:9:123), take everything before the first colon for display
    delims = [":", ";", "_"]
    gene_names = []
    for mut_name in mut_names:
        mut_name = str(mut_name)
        if not contains_delim(mut_name, delims):
            gene_names.append(mut_name)
        else:
            gene_names.append(re.split(r"[_;:]", mut_name)[0])
    # Try to find relevant cancer genes to label
    gene_candidates = set()
    for gene in gene_names:
        gene = gene.upper()
        if gene in CANCER_DRIVER_GENES:
            gene_candidates.add(gene)
        elif gene in ENSEMBLE_TO_GENE_MAP:
            gene_candidates.add(ENSEMBLE_TO_GENE_MAP[gene])
    final_genes = gene_names if len(gene_candidates) == 0 else gene_candidates

    k = 2 if len(final_genes) > 2 else len(final_genes)
    if to_string:
        return ";".join(list(final_genes)[:k])
    else:
        return list(final_genes)

def full_tree_node_idx_to_label(T, node_collection, shorten_label=True, to_string=False):
    """
    Build a map of node_idx to (label, is_witness, is_polytomy_resovler_node) for plotting and saving
    information to pickle file
    """

    full_node_idx_to_label_map = dict()
    for i, j in vutil.tree_iterator(T):
        node_i = node_collection.get_node(i)
        label = pruned_mut_label(node_i.label, shorten_label, to_string)
        full_node_idx_to_label_map[i] = (label, node_i.is_witness, node_i.is_polytomy_resolver_node)
        
        node_j = node_collection.get_node(j)
        label = pruned_mut_label(node_j.label, shorten_label, to_string)
        full_node_idx_to_label_map[j] = (label, node_j.is_witness, node_j.is_polytomy_resolver_node)
    return full_node_idx_to_label_map

def idx_to_color(custom_colors, idx, alpha=1.0):
    rgb = mcolors.to_rgb(custom_colors[idx])
    rgb_alpha = (rgb[0], rgb[1], rgb[2], alpha)
    return mcolors.to_hex(rgb_alpha, keep_alpha=True)

def prep_V_A_inputs(V, A):
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch.float)
    elif V.dtype != torch.float:
        V = V.float()
    if not isinstance(A, torch.Tensor):
        if isinstance(A, tuple):
            # Assuming A is a tuple of (indices, values, size) for sparse COO tensor
            indices, values, size = A
            A = torch.sparse_coo_tensor(indices, values, size)
        else:
            A = torch.tensor(A)
    return V, A

def migration_graph(V, A):
    """
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    """
    V, A = prep_V_A_inputs(V, A)
    migration_graph = (V @ A) @ V.T
    migration_graph_no_diag = torch.mul(migration_graph, 1-torch.eye(migration_graph.shape[0], migration_graph.shape[1], device=migration_graph.device))
    
    return migration_graph_no_diag

def find_abbreviation_mark(input_string):
    abbreviation_marks = [',', '-', '|']  # Add other abbreviation marks as needed

    for mark in abbreviation_marks:
        if input_string.count(mark) == 1:
            return mark
    return None

def migration_graph_dot(V, A, ordered_sites, custom_colors, show=True):
    """
    Plots migration graph G which represents the migrations/comigrations between
    all anatomical sites.

    Returns a list of edges (e.g. [('P' ,'M1'), ('P', 'M2')])
    """
    # Reformat anatomical site strings if too long for display and there
    # is an easy way to split the string (abbreviation of some sort)
    fmted_ordered_sites = []
    for site in ordered_sites:
        if len(site) > 17:
            mark = find_abbreviation_mark(site)
            if mark != None:
                fmted_ordered_sites.append(f"{mark}\n".join(site.split(mark)))
            else:
                fmted_ordered_sites.append(site)
        else:
            fmted_ordered_sites.append(site)

    mig_graph_no_diag = migration_graph(V, A)

    G = nx.MultiDiGraph()
    for node, color in zip(fmted_ordered_sites, custom_colors):
        G.add_node(node, shape="box", color=color, fillcolor='white', fontname=FONT, penwidth=3.0)

    for i, adj_row in enumerate(mig_graph_no_diag):
        for j, num_edges in enumerate(adj_row):
            if num_edges > 0:
                for _ in range(int(num_edges.item())):
                    G.add_edge(fmted_ordered_sites[i], fmted_ordered_sites[j], color=f'"{custom_colors[i]};0.5:{custom_colors[j]}"', penwidth=3)

    dot = nx.nx_pydot.to_pydot(G)
    if show:
        view_pydot(dot)

    dot_lines = dot.to_string().split("\n")
    dot_lines.insert(1, 'dpi=600;size=3.5;')
    dot_str = ("\n").join(dot_lines)

    return dot_str

def migration_history_tree_dot(V, T, gen_dist, custom_colors, node_collection=None, show=True, display_labels=True):

    # (1) Create full directed graph 
    # these labels are used for display in plotting
    display_node_idx_to_label_map = full_tree_node_idx_to_label(T, node_collection, shorten_label=True, to_string=True)            

    color_map = { i:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":FONT, "fontsize":14,
                    "fixedsize":"true", "height":0.3}

    if gen_dist != None:
        gen_dist = ((gen_dist / torch.max(gen_dist[gen_dist>0]))*1.5) + 1

    for i, j in vutil.tree_iterator(T):
        label_i, _, _ = display_node_idx_to_label_map[i]
        label_j, is_witness, _ = display_node_idx_to_label_map[j]
        if not display_labels:
            label_i, label_j = "", ""
        G.add_node(i, xlabel=label_i, fillcolor=color_map[i], 
                    color=color_map[i], style="filled", **node_options)
        G.add_node(j, xlabel="" if is_witness else label_j, fillcolor=color_map[j], fontcolor="white" if is_witness else "black",
                    color=color_map[j], style="solid" if is_witness else "filled", **node_options)
       
        style = "dashed" if is_witness else "solid"
        penwidth = 5 if is_witness else 5.5

        # We're iterating through i,j of the full tree (including leaf nodes),
        # while G only has genetic distances between internal nodes
        minlen = gen_dist[i, j].item() if (gen_dist != None and i < len(gen_dist) and j < len(gen_dist)) else 1.0

        G.add_edge(i, j,color=f'"{color_map[i]};0.5:{color_map[j]}"', 
                   penwidth=penwidth, arrowsize=0, style=style, minlen=minlen)

    # Add edge from normal to root 
    root_idx = vutil.get_root_index(T)
    root_label = display_node_idx_to_label_map[root_idx][0]
    G.add_node("normal", label="", xlabel=root_label, penwidth=3, style="invis")
    G.add_edge("normal", root_idx, label="", 
                color=f'"{color_map[root_idx]}"', 
                penwidth=4, arrowsize=0, style="solid")

    assert(nx.is_tree(G))

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G)
    dot.set_graph_defaults(layout='dot', seed=42) 
    dot = dot.to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    # dot.insert(1, 'graph[splines=false]; nodesep=0.7; rankdir=TB; ranksep=0.6; forcelabels=true; dpi=800; size=2.5;')
    dot.insert(1, 'graph[splines=false]; nodesep=0.4; rankdir=TB; ranksep=0.4; forcelabels=true; dpi=800; size=2.5; seed=42')

    dot_str = ("\n").join(dot)

    if show:
        dot = nx.nx_pydot.to_pydot(dot_str)
        view_pydot(dot)

    return dot_str

def collect_top_tree_info(V, T, node_collection, ordered_sites):
    """
    Get info about the best migration history to return to the user
    """
    # these labels are used for writing out full vertex names to file
    full_node_idx_to_label_map = full_tree_node_idx_to_label(T, node_collection, shorten_label=False, to_string=False)
    tree_edges = []
    for i, j in vutil.tree_iterator(T):
        tree_edges.append((full_node_idx_to_label_map[i][0], full_node_idx_to_label_map[j][0]))
    vertex_name_to_site_map = { ";".join(full_node_idx_to_label_map[i][0]):ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}

    mig_graph_no_diag = migration_graph(V, T)

    mig_edges = []
    for i, adj_row in enumerate(mig_graph_no_diag):
        for j, num_edges in enumerate(adj_row):
            if num_edges > 0:
                for _ in range(int(num_edges.item())):
                    mig_edges.append((ordered_sites[i], ordered_sites[j]))

    return tree_edges, full_node_idx_to_label_map, vertex_name_to_site_map, mig_edges

def construct_loss_dict(soln, full_loss):
    loss_dict = {MIG_KEY: soln.m, COMIG_KEY:soln.c, SEEDING_KEY: soln.s, ORGANOTROP_KEY: soln.o, GEN_DIST_KEY: soln.g, ENTROPY_KEY: soln.e}
    loss_dict = {**loss_dict, **{FULL_LOSS_KEY: round(torch.mean(full_loss).item(), 3)}}
    return loss_dict

def convert_lists_to_np_arrays(pickle_outputs, keys):
    """
    Makes unpickling these outputs much faster
    """
    for key in keys:
        if key in pickle_outputs:
            pickle_outputs[key] = np.array(pickle_outputs[key])

    return pickle_outputs

def figure_output_pattern(V, A, idx_to_label):
    """
    Display clonality and phyleticity
    """
    gen_clonality = genetic_clonality(V, A, idx_to_label).replace("clonal", "")
    st_clonality = site_clonality(V, A).replace("clonal", "")
    pattern = seeding_pattern(V, A)
    phyletic = phyleticity(V, A, idx_to_label)
    output_str = f"{pattern}, {phyletic}\n"
    output_str += f"genetic clonality: {gen_clonality}, site clonality: {st_clonality}\n"
    return output_str

def get_parents_sparse(tensor):
    """
    Get parents vector from sparse adjacency matrix
    """
    # Extract parents vector
    size = tensor.size()
    parents = -1 * np.ones(size[0])  # Initialize with -1 (for roots)

    # Set parents using adjacency matrix
    # Coalesce the sparse tensor first to ensure unique indices
    tensor = tensor.coalesce()
    indices = tensor.indices()
    parents[indices[1]] = indices[0]

    return parents

def get_parents(tensor):
    """Convert adjacency matrix tensor to parents vector.
    
    Args:
        tensor: Adjacency matrix tensor (sparse or dense)
        
    Returns:
        numpy array: Parents vector where index i contains the parent node of node i,
                    with -1 indicating the root node
    """
    # Convert to dense numpy array if sparse
    if tensor.is_sparse:
       return get_parents_sparse(tensor)
    
    # Convert to CPU numpy array if on GPU
    adj = tensor.detach().cpu().numpy()
    
    # Zero out diagonal
    np.fill_diagonal(adj, 0)
    
    # Find parent indices for all nodes
    parents = np.argmax(adj, axis=0)
    
    # Set parent to -1 where there is no parent (i.e., where column sum is 0)
    col_sums = adj.sum(axis=0)
    parents[col_sums == 0] = -1
    
    return parents

def dense_to_sparse(dense_tensor):
    """Convert a dense tensor to a sparse COO tensor and delete the dense tensor."""
    # Ensure the input is a 2D tensor (you can modify for N-D tensors if needed)
    if dense_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional.")
    
    # Get the indices of non-zero elements
    indices = torch.nonzero(dense_tensor, as_tuple=False).t()
    
    # Get the values of the non-zero elements
    values = dense_tensor[indices[0], indices[1]]

    # Create a sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=dense_tensor.size())
    
    # Delete the dense tensor to free memory
    del dense_tensor

    return sparse_tensor

def restructure_matrices_for_plotting(T, node_collection, G, V, U, original_root_idx):
    """
    Restructures adjacency matrices to match original cluster indices for plotting.
    
    Args:
        T: Transition matrix (sparse or dense)
        node_collection: Collection of nodes
        G: Graph matrix
        V, U: Additional matrices
        original_root_idx: Original root index (-1 if no restructuring needed)
    
    Returns:
        tuple: (T, node_collection, G, V, U)
    """
    if original_root_idx == -1:
        return T, node_collection, G, V, U
        
    needs_sparse_conversion = False
    if T.is_sparse:
        T = T.to_dense()
        needs_sparse_conversion = True
    
    node_collection = copy.deepcopy(node_collection)
    T, _, _, node_collection, G, _, V, U = vutil.restructure_matrices(
        0, original_root_idx, T, None, None, node_collection, G, None, V, U
    )
    
    if needs_sparse_conversion:
        T = dense_to_sparse(T)
        
    return T, node_collection, G, V, U



def save_best_trees(min_loss_solutions, U, O, weights, ordered_sites, print_config, primary, output_dir, run_name, original_root_idx=-1):
    """
    min_loss_solutions is in order from lowest to highest loss 

    original_root_idx: if not -1, swap the original_root_idx with 0 in all
    data that we save that involves node/cluster indices. This will then match
    the inputs from the user's again.
    """
    
    primary_idx = ordered_sites.index(primary)

    ret = None
    figure_outputs = []
    pickle_outputs = {OUT_LABElING_KEY:[], OUT_LOSSES_KEY:[],OUT_IDX_LABEL_KEY:[],
                      OUT_PARENTS_KEY:[], OUT_SITES_KEY:ordered_sites, OUT_LOSS_DICT_KEY:[],
                      OUT_PRIMARY_KEY:primary, 
                      OUT_SOFTV_KEY:[], OUT_GEN_DIST_KEY:[]}

    with torch.no_grad():
        if print_config.custom_colors == None:
            custom_colors = copy.deepcopy(DEFAULT_COLORS)
            # Reorder so that green is always the primary
            green_idx = custom_colors.index(DEFAULT_GREEN)
            custom_colors[primary_idx], custom_colors[green_idx] = custom_colors[green_idx], custom_colors[primary_idx]
        else:
            custom_colors = print_config.custom_colors

        for i, min_loss_solution in enumerate(min_loss_solutions):
            V = min_loss_solution.V
            soft_V = min_loss_solution.soft_V
            T = min_loss_solution.T
            G = min_loss_solution.G
            full_loss = min_loss_solution.loss
            node_collection = min_loss_solution.node_collection
            loss_dict = construct_loss_dict(min_loss_solution, full_loss)
            
            # Restructure adjacency matrices to match original cluster indices
            T, node_collection, G, V, U = restructure_matrices_for_plotting(
                T, node_collection, G, V, U, original_root_idx
            )
            
            edges, full_tree_idx_to_label, vertices_to_sites_map, mig_graph_edges = collect_top_tree_info(V, T, node_collection, ordered_sites)
            if print_config.visualize:
                pattern = figure_output_pattern(V, T, full_tree_idx_to_label)
                tree_dot = migration_history_tree_dot(V, T, G, custom_colors, node_collection, show=False, display_labels=print_config.display_labels)
                mig_graph_dot = migration_graph_dot(V, T, ordered_sites, custom_colors, show=False)
            else:
                pattern = ""
                tree_dot, mig_graph_dot = None, None

            figure_outputs.append((tree_dot, mig_graph_dot, loss_dict, pattern))
            pickle_outputs[OUT_LABElING_KEY].append(V.detach().cpu().numpy())
            pickle_outputs[OUT_LOSSES_KEY].append(full_loss.cpu().numpy())
            
            pickle_outputs[OUT_PARENTS_KEY].append(get_parents(T))
            pickle_outputs[OUT_SOFTV_KEY].append(soft_V.detach().cpu().numpy())
            pickle_outputs[OUT_OBSERVED_CLONES_KEY] = U.detach().cpu().numpy() if U != None else np.array([])
        
            if G != None:
                pickle_outputs[OUT_GEN_DIST_KEY].append(G.detach().cpu().numpy())                
            pickle_outputs[OUT_LOSS_DICT_KEY].append(loss_dict)
            pickle_outputs[OUT_IDX_LABEL_KEY].append(full_tree_idx_to_label)
            if i == 0: # Best tree
                ret = (edges, vertices_to_sites_map, mig_graph_edges, loss_dict)

        #pickle_outputs = convert_lists_to_np_arrays(pickle_outputs, [OUT_LABElING_KEY, OUT_LOSSES_KEY, OUT_PARENTS_KEY, OUT_SOFTV_KEY, OUT_GEN_DIST_KEY])

        save_outputs(figure_outputs, print_config, output_dir, run_name, pickle_outputs, weights)

    return ret

def formatted_loss_string(loss_dict, weights):
    s = f"Loss: {loss_dict[FULL_LOSS_KEY]}\n\n"

    s += f"Migration num.: {int(loss_dict[MIG_KEY])}\n"
    s += f"Comigration num.: {int(loss_dict[COMIG_KEY])}\n"
    s += f"Seeding site num.: {int(loss_dict[SEEDING_KEY])}\n"
    s += f"Neg. entropy: {round(float(loss_dict[ENTROPY_KEY]), 3)}\n"

    if weights.gen_dist != 0:
        s += f"Genetic dist. loss: {round(float(loss_dict[GEN_DIST_KEY]), 3)}\n"
    if weights.organotrop != 0:
        s += f"Organotrop. loss: {round(float(loss_dict[ORGANOTROP_KEY]), 3)}\n"
    return s

def save_outputs(figure_outputs, print_config, output_dir, run_name, pickle_outputs, weights):

    if print_config.visualize:
        k = print_config.k_best_trees
        sys_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        for font in sys_fonts:
            if FONT in font:
                matplotlib.font_manager.fontManager.addfont(font)
                rcParams['font.family'] = FONT

        n = len(figure_outputs)
        print(run_name)

        max_trees = 10
        if n > max_trees:
            print(f"More than {max_trees} solutions detected, only plotting top {max_trees} trees.")
            n = max_trees
        # Create a figure and subplots
        #fig, axs = plt.subplots(3, k*2, figsize=(10, 8))

        plt.suptitle(run_name)

        z = 2 # number of trees displayed per row

        nrows = math.ceil(n/z)
        h = nrows*4
        fig = plt.figure(figsize=(8,h))
        
        vspace = 1/nrows

        for i, (tree_dot, mig_graph_dot, loss_info, seeding_pattern) in enumerate(figure_outputs):
            if i >= max_trees:
                break
            tree = pgv.AGraph(string=tree_dot).draw(format="png", prog="dot", args="-Glabel=\"\"")
            tree = PILImage.open(io.BytesIO(tree))
            mig_graph = pgv.AGraph(string=mig_graph_dot).draw(format="png", prog="dot")
            mig_graph = PILImage.open(io.BytesIO(mig_graph))

            gs = gridspec.GridSpec(3, 1, height_ratios=[0.02, 0.73, 0.25])

            row = math.floor(i/2)
            pad = 0.02

            # left = 0.0 if i is odd, 0.55 if even
            # right = 0.45 if i is odd, 1.0 if even
            gs.update(left=0.0+((i%2)*0.53), right=0.47+0.55*(i%2), top=1-(row*vspace)-pad, bottom=1-((row+1)*vspace)+pad, wspace=0.05)

            # Top row: Title
            ax1 = plt.subplot(gs[0])
            ax1.text(0.5, 0.5, f'Solution {i+1}\n{seeding_pattern}', ha='center', va='center', fontsize=7)
            ax1.axis('off')  # Hide the axis

            # Second row: Plot for the tree
            ax2 = plt.subplot(gs[1])
            ax2.imshow(tree)
            ax2.axis('off')

            # Third row: Create a subgrid for the migration graph and loss information
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.05)

            # Left column for the migration graph
            ax3 = plt.subplot(gs_bottom[0])
            ax3.imshow(mig_graph)
            ax3.axis('off')

            # Right column for loss information
            ax4 = plt.subplot(gs_bottom[1])
            ax4.text(0.5, 0.5, formatted_loss_string(loss_info, weights), ha='center', va='center', fontsize=7)
            ax4.axis('off')

        fig1 = plt.gcf()
        plt.show()
        plt.close()
        if print_config.save_outputs: 
            fig1.savefig(os.path.join(output_dir, f'{run_name}.png'), dpi=600, bbox_inches='tight')

    if print_config.save_outputs:
        if not os.path.isdir(output_dir):
            raise ValueError(f"{output_dir} does not exist.")
        if print_config.verbose: print(f"Saving {run_name} to {output_dir}")
        # Save results to pickle file
        # with open(os.path.join(output_dir, f"{run_name}.pickle"), 'wb') as handle:
        with gzip.open(os.path.join(output_dir,f"{run_name}.pkl.gz"), 'wb') as gzip_file:
            pickle.dump(pickle_outputs, gzip_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save best dot to file
        tree_dot, mig_graph_dot, _, _ = figure_outputs[0]
        if tree_dot is not None:
            with open(os.path.join(output_dir, f"{run_name}.tree.dot"), 'w') as file:
                file.write(tree_dot)
        if mig_graph_dot is not None:
            with open(os.path.join(output_dir, f"{run_name}.mig_graph.dot"), 'w') as file:
                file.write(mig_graph_dot)
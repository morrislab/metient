import numpy as np
import torch
import networkx as nx
import math
from queue import Queue
from metient.util.globals import *
from collections import deque
import copy

import scipy.sparse as sp
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_columns', None)

LAST_P = None

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

######################################################
##################### CLASSES ########################
######################################################

    
# Defines a unique adjacency matrix and vertex labeling
class LabeledTree:
    def __init__(self, tree, labeling):
        if (tree.shape[0] != tree.shape[1]):
            raise ValueError("Adjacency matrix should have shape (num_nodes x num_nodes)")
        if (tree.shape[0] != labeling.shape[1]):
            raise ValueError("Vertex labeling matrix should have shape (num_sites x num_nodes)")

        self.tree = tree
        self.labeling = labeling

    def _nonzero_tuple(self, tensor):
        # Get the indices of non-zero elements and convert to a hashable form
        indices = torch.nonzero(tensor, as_tuple=False)
        return tuple(map(tuple, indices.tolist()))

    def __hash__(self):
        # Compute a hash based on the positions of non-zero entries
        return hash((self._nonzero_tuple(self.labeling), self._nonzero_tuple(self.tree)))

    def __eq__(self, other):
        # Check for equality based on the positions of non-zero entries in both tensors
        if not isinstance(other, LabeledTree):
            return False
        return (self._nonzero_tuple(self.labeling) == self._nonzero_tuple(other.labeling) and
                self._nonzero_tuple(self.tree) == self._nonzero_tuple(other.tree))

    def __str__(self):
        A = str(np.where(self.tree == 1))
        V = str(np.where(self.labeling == 1))
        return f"Tree: {A}\nVertex Labeling: {V}"

def convert_pars_metric_to_int(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return int(x.item())  
    raise ValueError(f"Got unexpected value for parsimony metric {x}")

# Convenience object to package information needed for a final solution
class VertexLabelingSolution:
    def __init__(self, loss, m, c, s, V, soft_V, T, G, idx_to_label):
        self.loss = loss
        self.m = convert_pars_metric_to_int(m)
        self.c = convert_pars_metric_to_int(c)
        self.s = convert_pars_metric_to_int(s)
        self.V = V
        self.T = T
        self.G = G
        self.idx_to_label = idx_to_label
        self.soft_V = soft_V

    # Override the comparison operator
    def __lt__(self, other):
        return self.loss < other.loss
    
    def _nonzero_tuple(self, tensor):
        # Get the indices of non-zero elements and convert to a hashable form
        indices = torch.nonzero(tensor, as_tuple=False)
        return tuple(map(tuple, indices.tolist()))

    def __hash__(self):
        # Compute a hash based on the positions of non-zero entries
        return hash((self._nonzero_tuple(self.V), self._nonzero_tuple(self.T)))

    def __eq__(self, other):
        # Check for equality based on the positions of non-zero entries in both tensors
        if not isinstance(other, VertexLabelingSolution):
            return False
        return (self._nonzero_tuple(self.V) == self._nonzero_tuple(other.V) and
                self._nonzero_tuple(self.T) == self._nonzero_tuple(other.T))



# Convenience object to package information needed for known 
# labelings/indices of vertex labeling matrix V
class FixedVertexLabeling:
    def __init__(self, known_indices, unknown_indices, known_labelings):
        self.known_indices = known_indices
        self.unknown_indices = unknown_indices
        self.known_labelings = known_labelings

######################################################
############# CALCULATING PARSIMONY METRICS ##########
######################################################

def migration_number(site_adj):
    '''
    Args:
        - site_adj: batch_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j
    Returns:
        - migration number: number of total migrations between sites (no same site to same site migrations)
    '''
    site_adj_trace = torch.diagonal(site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    m = torch.sum(site_adj, dim=(1, 2)) - site_adj_trace
    return m

def seeding_site_number(site_adj_no_diag):
    '''
    Args:
        - site_adj_no_diag: batch_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j, and no same site migrations are included
    Returns:
        - seeding site number: number of sites that have outgoing edges
    '''
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=2)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    binarized_row_sums_site_adj = torch.sigmoid(BINARY_ALPHA * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj, dim=(1))
    return s

def comigration_number(site_adj, A, VA, VT, X, update_path_matrix):
    '''
    Args:
        - site_adj: batch_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j
        - A: Adjacency matrix (directed) of the full tree (batch_size x num_nodes x num_nodes)
        - VA: V*A
        - VT: transpose of V 
        - X: VT*V (1 if node i and node j are the same color)
        - update_path_matrix: whether we need to update the path matrix or we can use a cached version
        (need to update when we're actively resolving polytomies)
    Returns:
        - comigration number: a subset of the migration edges between two anatomical sites, such that 
        the migration edges occur on distinct branches of the clone tree. It is the number of multi-edges in 
        migration graph G (V*A*V.T)
    '''

    VAT = torch.transpose(VA, 2, 1)
    W = VAT @ VA # 1 if two nodes' parents are the same color
    Y = torch.sum(torch.mul(VAT, 1-VT), axis=2) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X) # 1 if two nodes' parents are same color AND nodes are same color
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    global LAST_P # this is expensive to compute, so hash it if we don't need to update it
    if LAST_P != None and not update_path_matrix:
        P = LAST_P
    else:
        P = path_matrix(A, remove_self_loops=True)
        LAST_P = P

    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=2)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y), axis=1)
    binarized_site_adj = torch.sigmoid(BINARY_ALPHA * (2 * site_adj - 1))
    bin_site_trace = torch.diagonal(binarized_site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    c = torch.sum(binarized_site_adj, dim=(1,2)) - bin_site_trace + repeated_temporal_migrations
    return c

def genetic_distance_score(G, m, A, X):
    '''
    Args:
        - G: Matrix of genetic distances between internal nodes (shape: batch_size x num_internal_nodes x num_internal_nodes).
             Lower values indicate lower branch lengths, i.e. more genetically similar.
        - m: vector of migration numbers (length = batch_size)
        - A: Adjacency matrix (directed) of the full tree (batch_size x num_nodes x num_nodes)
        - X: VT*V (1 if node i and node j are the same color)
    Returns:
        - genetic distance score, summed over batch_size # of solutions
    '''
    g = 0
    if G != None:
        # Calculate if 2 nodes are in diff sites and there's an edge between them (i.e there is a migration edge)
        R = torch.mul(A, (1-X))
        adjusted_G = -torch.log(G+0.01)
        R = torch.mul(R, adjusted_G)
        g = torch.sum(R, dim=(1,2))/(m)
    return g

def organotropism_score(O, site_adj_no_diag, p, bs, num_sites):
    '''
    Args:
        - O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        - site_adj_no_diag: batch_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j, and no self loops
        - p: one-hot vector indicating site of the primary
        - bs: batch_size (number of samples)
        - num_sites: number of anatomical sites
    Returns:
        - organotropism score, summed over batch_size # of solutions
    '''
    o = 0
    if O != None:
        # the organotropism frequencies can only be used on the first 
        # row, which is for the migrations from primary cancer site to
        # other metastatic sites (we don't have frequencies for every 
        # site to site migration)
        prim_site_idx = torch.nonzero(p)[0][0]
        O = O.repeat(bs,1).reshape(bs, O.shape[0])
        adjusted_freqs = -torch.log(O+0.01)
        num_mig_from_prim = site_adj_no_diag[:,prim_site_idx,:]
        organ_penalty = torch.mul(num_mig_from_prim, adjusted_freqs)
        o = torch.sum(organ_penalty, dim=(1))/torch.sum(num_mig_from_prim, dim=(1))
    return o

def ancestral_labeling_metrics(V, A, G, O, p, update_path_matrix):

    single_A = A
    bs = V.shape[0]
    num_sites = V.shape[1]
    A = A if len(A.shape) == 3 else repeat_n(single_A, bs) 

    # Compute matrices used for all parsimony metrics
    VA = V @ A
    VT = torch.transpose(V, 2, 1)
    site_adj = VA @ VT
    # Remove the same site transitions from the site adjacency matrix
    site_adj_no_diag = torch.mul(site_adj, repeat_n(1-torch.eye(num_sites, num_sites), bs))
    X = VT @ V # 1 if two nodes are the same color

    # 1. Migration number
    m = migration_number(site_adj)

    # 2. Seeding site number
    s = seeding_site_number(site_adj_no_diag)

    # 3. Comigration number
    c = comigration_number(site_adj, A, VA, VT, X, update_path_matrix)

    # 4. Genetic distance
    g = genetic_distance_score(G, m, A, X)

    # 5. Organotropism
    o = organotropism_score(O, site_adj_no_diag, p, bs, num_sites)

    return m, c, s, g, o

def calc_entropy(V, soft_V):
    '''
    Args:
        - V: one-hot vertex labeling matrix
        - soft_V: underlying theta parameters of V
    Returns:
        - entropy for the categorical variable representing each node's vertex label
    '''
    eps = 1e-7 # to avoid nans when values in soft_V get very close to 0
    return torch.sum(torch.mul(soft_V, torch.log2(soft_V+eps)), dim=(1, 2)) / V.shape[2]

def get_repeating_weight_vector(bs, weight_list):
    # Calculate the number of times each weight should be repeated
    total_weights = len(weight_list)
    repeats = bs // total_weights
    remaining_elements = bs % total_weights

    # Create a list where each weight is repeated 'repeats' times
    repeated_list = [weight for weight in weight_list for _ in range(repeats)]

    # Add remaining elements to match the batch size exactly
    if remaining_elements > 0:
        additional_weights = weight_list[:remaining_elements]
        additional_repeated = [weight for weight in additional_weights]
        repeated_list += additional_repeated

    # Convert the list to a tensor
    weights_vec = torch.tensor(repeated_list)

    return weights_vec

def get_mig_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.mig)

def get_seed_site_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.seed_site)

def clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=1):

    # Combine all 5 components with their weights
    # Explore different weightings
    if isinstance(weights.mig, list) and isinstance(weights.seed_site, list):
        mig_weights_vec = get_mig_weight_vector(bs, weights)
        seeding_sites_weights_vec = get_seed_site_weight_vector(bs, weights)
        mig_loss = torch.mul(mig_weights_vec, m)
        seeding_loss = torch.mul(seeding_sites_weights_vec, s)
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
        
    else:
        mig_loss = weights.mig*m
        seeding_loss = weights.seed_site*s
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
    return labeling_loss

def clone_tree_labeling_objective(V, soft_V, A, G, O, p, weights, update_path_matrix):
    '''
    Args:
        V: Vertex labeling of the full tree (batch_size x num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        G: Matrix of genetic distances between internal nodes (shape:  num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        p: one-hot vector indicating site of the primary
        weights: Weights object

    Returns:
        Loss to score the ancestral vertex labeling of the given tree. This combines (1) migration number, (2) seeding site
        number, (3) comigration number, and optionally (4) genetic distance and (5) organotropism.
    '''
    V = add_batch_dim(V)
    soft_V = add_batch_dim(soft_V)
    m, c, s, g, o = ancestral_labeling_metrics(V, A, G, O, p, update_path_matrix)
    # Entropy
    e = calc_entropy(V, soft_V)

    labeling_loss = clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=V.shape[0])

    return labeling_loss, (m, c, s)

######################################################
######### POST U MATRIX ESTIMATION UTILITIES #########
######################################################

def get_leaf_labels_from_U(U):
    
    U = U[:,1:] # don't include column for normal cells
    internal_node_idx_to_sites = {}
    for node_idx in range(U.shape[1]):
        for site_idx in range(U.shape[0]):
            node_U = U[site_idx,node_idx]
            if node_U > U_CUTOFF:
                if node_idx not in internal_node_idx_to_sites:
                    internal_node_idx_to_sites[node_idx] = []
                internal_node_idx_to_sites[node_idx].append(site_idx)
    
    return internal_node_idx_to_sites


def full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val):
    '''
    All non-zero values of U represent extant clones (leaf nodes of the full tree).
    For each of these non-zero values, we add an edge from parent clone to extant clone.
    '''
    num_leaves = sum(len(lst) for lst in idx_to_sites_present.values())
    full_adj = torch.nn.functional.pad(input=input_T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
    leaf_idx = input_T.shape[0]
    # Also add branch lengths (genetic distances) for the edges we're adding.
    # Since all of these edges we're adding represent genetically identical clones,
    # we are going to add a very small but non-zero value.
    full_G = torch.nn.functional.pad(input=input_G, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0) if input_G is not None else None

    leaf_labels = []
    # Iterate through the internal nodes that we want to attach leaf nodes to
    for internal_node_idx in idx_to_sites_present:
        # Attach a leaf node for every site that this internal node is observed in
        for site in idx_to_sites_present[internal_node_idx]:
            full_adj[internal_node_idx, leaf_idx] = 1
            if input_G is not None:
                full_G[internal_node_idx, leaf_idx] = G_identical_clone_val
            leaf_idx += 1
            leaf_labels.append(site)
    # Anatomical site labels of the leaves
    L = torch.nn.functional.one_hot(torch.tensor(leaf_labels), num_classes=num_sites).T
    return full_adj, full_G, L
    
def full_adj_matrix_using_inputted_observed_clones(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val):
    '''
    Use inputted observed clones to fill out T and G by adding leaf nodes
    '''
    full_adj, full_G, L = full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val)
    return L, full_adj, full_G

def full_adj_matrix_using_inferred_observed_clones(U, input_T, input_G, num_sites, G_identical_clone_val):
    '''
    Use inferred observed clones to fill out T and G by adding leaf nodes
    '''
    internal_node_idx_to_sites = get_leaf_labels_from_U(U)
    full_adj, full_G, L = full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, internal_node_idx_to_sites, num_sites, G_identical_clone_val)
    
    return full_adj, full_G, L, internal_node_idx_to_sites

def remove_leaf_indices_not_observed_sites(removal_indices, U, input_T, T, G, node_idx_to_label, idx_to_observed_sites):
    '''
    Remove clone tree leaf nodes that are not detected in any sites. 
    These are not well estimated
    '''
    if len(removal_indices) == 0:
        return U, input_T, T, G, node_idx_to_label, idx_to_observed_sites
    
    for remove_idx in removal_indices:
        child_indices = get_child_indices(T, [remove_idx])
        assert len(child_indices) == 0

    # Remove indices from input_T, full T (now with observed clones), U and G
    T = np.delete(T, removal_indices, 0)
    T = np.delete(T, removal_indices, 1)

    # Remove indices from T, U and G
    input_T = np.delete(input_T, removal_indices, 0)
    input_T = np.delete(input_T, removal_indices, 1)

    U = np.delete(U, removal_indices, 1)

    if G != None: 
        G = np.delete(G, removal_indices, 0)
        G = np.delete(G, removal_indices, 1)

    # Reindex the idx to label dict
    new_node_idx_to_label, old_index_to_new_index = reindex_dict(node_idx_to_label, removal_indices)
    new_idx_to_observed_sites = {}
    for old_idx in idx_to_observed_sites:
        new_idx = old_index_to_new_index[old_idx]
        new_idx_to_observed_sites[new_idx] = idx_to_observed_sites[old_idx]

    return U, input_T, T, G, new_node_idx_to_label, new_idx_to_observed_sites

######################################################
################## RANDOM UTILITIES ##################
######################################################

def mutation_matrix_with_normal_cells(T):
    B = mutation_matrix(T)
    # Add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # Add a column of ones to indicate that every clone is a descendent of the non-cancerous root node
    B = torch.hstack([torch.ones(B.shape[0]).reshape(-1,1), B])
    return B

def adjacency_matrix_to_edge_list(adj_matrix):
    edges = []
    for i, j in tree_iterator(adj_matrix):
        edges.append((i, j))
    return edges

def reindex_dict(original_dict, indices_to_remove):
    # Create a new dictionary to hold the re-indexed entries
    new_dict = {}
    
    old_index_to_new_index = {}
    # Initialize the new index
    new_index = 0
    # Iterate through the original dictionary in sorted index order
    for old_index in sorted(original_dict.keys()):
        # Skip the indices that need to be removed
        if old_index in indices_to_remove:
            continue
        
        # Assign the new index to the current label
        new_dict[new_index] = original_dict[old_index]
        old_index_to_new_index[old_index] = new_index
        # Increment the new index
        new_index += 1
    
    return new_dict, old_index_to_new_index

def add_batch_dim(x):
    if len(x.shape) == 3:
        return x
    return x.reshape(1, x.shape[0], x.shape[1])

def to_tensor(t):
    if torch.is_tensor(t):
        return t
    return torch.stack(t)

def remove_leaf_nodes_idx_to_label_dicts(dicts):
    '''
    After running migration history inference, the leaf nodes 
    (U nodes) get added to the idx to label dicts when plotting 
    and saving to pickle files, so we don't want to do that twice
    when in calibrate mode
    '''
    new_dicts = []
    for i,dct in enumerate(dicts):
        new_dicts.append(copy.deepcopy(dct))
        for key in dct:
            if dct[key][1] == True:
                del new_dicts[i][key]
            else:
                new_dicts[i][key] = new_dicts[i][key][0]
    return new_dicts

def create_reweighted_solution_set_from_pckl(pckl, O, p, weights):
    # Make a solution set from the pickled files
    Ts, Vs, soft_Vs, Gs = pckl[OUT_ADJ_KEY], pckl[OUT_LABElING_KEY], pckl[OUT_SOFTV_KEY], pckl[OUT_GEN_DIST_KEY]
    idx_to_label_dicts = remove_leaf_nodes_idx_to_label_dicts(pckl[OUT_IDX_LABEL_KEY])
    solution_set = []
    for T, V, soft_V, G, idx_to_label in zip(Ts, Vs, soft_Vs, Gs, idx_to_label_dicts):
        loss, (m,c,s) = clone_tree_labeling_objective(torch.tensor(V), torch.tensor(soft_V), torch.tensor(T), torch.tensor(G), O, p, weights, True)
        solution_set.append(VertexLabelingSolution(loss, m, c, s, torch.tensor(V), torch.tensor(soft_V), torch.tensor(T), torch.tensor(G), idx_to_label))
    return solution_set

def calculate_batch_size(T, sites, solve_polytomies):
    '''
    Calculate the number of samples to initialize for a run based on 
    the number of tree nodes, the number of anatomical sites, and if we're
    solving polytomies
    '''
    num_nodes = T.shape[0]
    num_sites = len(sites)
    min_size = 256
    min_size += num_nodes*num_sites*4

    if solve_polytomies:
        min_size *= 2

    # cap this to a reasonably high sample size
    min_size = min(min_size, 60000)
    #print("calculate_batch_size", min_size)
    return min_size

def tree_iterator(T):
    ''' 
    Iterate an adjacency matrix, yielding i and j for all values = 1
    '''
    # Enumerating through a torch tensor is pretty computationally expensive,
    # so convert to a sparse matrix to efficiently access non-zero values
    T = T if isinstance(T, np.ndarray) else T.detach().numpy()
    T = sp.coo_matrix(T)
    for i, j in zip(T.row, T.col):
        yield i,j

def bfs_iterator(tree, start_node):
    '''
    Iterate an adjacency matrix in breadth first search order
    '''
    queue = deque([start_node])  # Start the queue with the given start node index
    visited = set(queue)         # Mark the start node as visited

    # BFS loop that yields nodes in the order they are visited
    while queue:
        current = queue.popleft()
        yield current  # Yield the current node

        # Check each adjacency in the row for the current node
        for neighbor, is_connected in enumerate(tree[current]):
            if is_connected and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def has_leaf_node(adjacency_matrix, start_node, num_internal_nodes):
    num_nodes = adjacency_matrix.size(0)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    queue = Queue()

    queue.put(start_node)
    visited[start_node] = True

    while not queue.empty():
        current_node = int(queue.get())
        # Get neighbors of the current node
        neighbors = torch.nonzero(adjacency_matrix[current_node]).squeeze(1)

        # Enqueue unvisited neighbors
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if not visited[neighbor]:
                queue.put(neighbor)
                visited[neighbor] = True
                # it's a leaf node if its not in the clone tree idx label dict
                if neighbor > num_internal_nodes:
                    return True
    return False

def get_root_index(T):
    '''
    Returns the root idx (node with no inbound edges) from adjacency matrix T
    '''

    candidates = set([x for x in range(len(T))])
    #print("candidates", candidates)
    for _, j in tree_iterator(T):
        if j not in candidates:
            print("T", adjacency_matrix_to_edge_list(T))
            print("j", j)
        candidates.remove(j)
    msg = "More than one" if len(candidates) > 1 else "No"
    assert (len(candidates) == 1), f"{msg} root node detected"

    return list(candidates)[0]

def reverse_bfs_order(A):
    '''
    Returns nodes in reverse bfs order of adjacency matrix A
    '''
    root_idx = get_root_index(A)
    nodes = []
    for x in bfs_iterator(A, root_idx):
        nodes.append(x)
    nodes.reverse()
    return nodes

def get_leaves(A):
    '''
    Returns leaves of adjacency matrix A
    '''
    return [int(x) for x in torch.where(A.sum(dim=1)==0)[0]]

def get_descendants(A, i):
    '''
    Returns all descendant nodes of node i in  adjacency matrix A
    '''
    path = path_matrix(A,remove_self_loops=True)
    descendants = [int(x) for x in torch.where(path[i]==1)[0]]
    return descendants

def swap_keys(d, key1, key2):
    if key1 in d and key2 in d:
        # Both keys exist, swap their values
        d[key1], d[key2] = d[key2], d[key1]
    elif key1 in d and key2 not in d:
        # Only key1 exists, move its value to key2
        d[key2] = d.pop(key1)
    elif key2 in d and key1 not in d:
        # Only key2 exists, move its value to key1
        d[key1] = d.pop(key2)
    # If neither key exists, do nothing
    return d

def restructure_matrices_root_index_zero(adj_matrix, ref_matrix, var_matrix, node_idx_to_label, gen_dist_matrix, idx_to_observed_sites):
    '''
    Restructure the inputs so that the node at index 0 becomes the root node.
    '''
    og_root_idx = get_root_index(adj_matrix)
    return restructure_matrices(og_root_idx, 0, adj_matrix, ref_matrix, var_matrix, 
                                node_idx_to_label, gen_dist_matrix, idx_to_observed_sites, None, None)

def restructure_matrices(source_root_idx, target_root_idx, adj_matrix, ref_matrix, var_matrix, 
                         node_idx_to_label, gen_dist_matrix, idx_to_observed_sites, V, U):
    '''
    Restructure the inputs so that the order of nodes has source_root_idx and target_root_idx swapped

    Returns:
        - Restructured adjacency matrix, reference matrix, variant matrix, node_idt_to_label dcitionary,
        and genetic distance matricx
    '''
    if source_root_idx == target_root_idx:
        # Nothing to restructure here!
        return adj_matrix, ref_matrix, var_matrix, node_idx_to_label, gen_dist_matrix, idx_to_observed_sites, V, U
    
    new_order = [x for x in range(len(adj_matrix))]
    new_order[source_root_idx] = target_root_idx
    new_order[target_root_idx] = source_root_idx
    # Swap rows and columns to move the first row and column to the desired position
    swapped_adjacency_matrix = adj_matrix[new_order, :][:, new_order]

    # Swap columns 
    swapped_ref_matrix = ref_matrix[:, new_order] if ref_matrix != None else None
    swapped_var_matrix = var_matrix[:, new_order] if var_matrix != None else None
    swapped_V = V[:, new_order] if V != None else None
    
    if U == None:
        swapped_U = None
    else:
        normal_cells = U[:,0].view(-1, 1)
        U_order = [x+1 for x in new_order if x < U.shape[1]-1]
        swapped_U = U[:, U_order]
        swapped_U = torch.cat((normal_cells, swapped_U), dim=1)
        
    if gen_dist_matrix == None:
        swapped_gen_dist_matrix = None
    else:
        swapped_gen_dist_matrix = gen_dist_matrix[new_order, :][:, new_order]
    
    if idx_to_observed_sites == None:
        idx_to_observed_sites = None
    else:
        idx_to_observed_sites = swap_keys(idx_to_observed_sites, source_root_idx, target_root_idx)

    node_idx_to_label = swap_keys(node_idx_to_label, source_root_idx, target_root_idx)
    
    return swapped_adjacency_matrix, swapped_ref_matrix, swapped_var_matrix, node_idx_to_label, swapped_gen_dist_matrix, idx_to_observed_sites, swapped_V, swapped_U

def nodes_w_leaf_nodes(adj_matrices, num_internal_nodes):
    '''
    Args:
        - adj_matrices: 3d matrix, where each inner matrix is a 2d adjacency matric
        - num_internal_nodes: number of nodes in the clone tree which are not leaf nodes
          indicating clone presences

    Returns:
        A list of lists, where list i is for adjacency matrix i, and the inner list is a list
        of booleans indicated whether node j has a leaf clone presence node or not
    '''
    # Convert the 3D adjacency matrix to a PyTorch tensor
    adj_tensor = torch.tensor(adj_matrices, dtype=torch.float32)
    
    # Get dimensions
    _, n, _ = adj_tensor.size()

    # Step 2: Create a mask for columns with indices >= num_internal_nodes
    col_indices = torch.arange(n)
    col_mask = col_indices >= num_internal_nodes

    # Step 3: Apply the column mask to the 3D adjacency matrix
    # Broadcast the col_mask to the shape of the last dimension of adj_tensor
    filtered_cols = adj_tensor[:, :, col_mask]

    # Step 4: Check for each row in each 2D matrix if there's at least one '1' in the filtered columns
    mask = filtered_cols.any(dim=2)

    return mask

def print_U(U, B, node_idx_to_label, ordered_sites, ref, var):
    cols = ["GL"]+[";".join([str(i)]+node_idx_to_label[i][:2]) for i in range(len(node_idx_to_label))]
    U_df = pd.DataFrame(U.detach().numpy(), index=ordered_sites, columns=cols)

    print("U\n", U_df)
    F_df = pd.DataFrame((var/(ref+var)).numpy(), index=ordered_sites, columns=cols[1:])
    print("F\n", F_df)
    Fhat_df = pd.DataFrame((U @ B).detach().numpy()[:,1:], index=ordered_sites, columns=cols[1:])
    print("F hat\n", Fhat_df)

def top_k_integers_by_count(lst, k, min_num_sites, cutoff):
    # find unique sites and their counts
    unique_sites, counts = np.unique(lst, return_counts=True)
    # filter for sites that occur at least min_num_sites times
    filtered_values = unique_sites[counts >= min_num_sites]
    # recount after filtering
    unique_sites, counts = np.unique(filtered_values, return_counts=True)
    # sort based on counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    # get the top k integers and their counts
    if len(unique_sites) > k and cutoff:
        return list(unique_sites[sorted_indices[:k]])
        
    return list(unique_sites)

def traverse_until_Uleaf_or_mult_children(node, input_T, internal_node_idx_to_sites):
    """
    Traverse down the tree on a linear branch until a leaf node is found, and then
    return that node
    
    Special case where a node isn't detected in any sites and is on a long linear branch,
    so there are no children or grandchildren observed to help bias towards
    """
    if node in internal_node_idx_to_sites:
        return node
    
    children = input_T[node].nonzero()[:,0].tolist()
    
    if len(children) == 1:
        return traverse_until_Uleaf_or_mult_children(children[0], input_T, internal_node_idx_to_sites)
    
    return None  # no node has a leaf or node has more than one child
   
def get_resolver_sites_of_node_and_children(input_T, node_idx_to_sites, num_children, node_idx, include_children, min_num_sites, cutoff):
    '''
    Looking at the sites that node_idx and node_idx's children are in,
    figure out the sites that the polytomy resolver nodes should be initialized with
    (any sites that are detected >= thres times)
    '''
   
    clone_tree_children = input_T[node_idx].nonzero()[:,0].tolist()
    # calculated based on clone tree children and leaf nodes estimated form U
    num_possible_resolvers = math.floor(num_children/2)
    # get the sites that the node and node's children are in
    if include_children:
        node_and_children = clone_tree_children+ [int(node_idx)]
    else:
        node_and_children = [int(node_idx)]
    sites = []
    for key in node_and_children:
        if key in node_idx_to_sites:
            sites.extend(node_idx_to_sites[key])
    # print("node_idx", node_idx, "node_and_children", node_and_children, "sites", sites)
    # Special case where the node of interest (node_idx) isn't detected in any sites
    # and is on a linear branch 
    if not include_children and len(sites) == 0 and len(clone_tree_children) == 1:
        leaf_on_linear_branch = traverse_until_Uleaf_or_mult_children(node_idx, input_T, node_idx_to_sites)
        # print("leaf_on_linear_branch", leaf_on_linear_branch)
        if leaf_on_linear_branch != None:
            sites = node_idx_to_sites[leaf_on_linear_branch]
    
    return top_k_integers_by_count(sites, num_possible_resolvers, min_num_sites, cutoff)
    
def get_k_or_more_children_nodes(input_T, T, internal_node_idx_to_sites, k, include_children, min_num_sites, cutoff=True):
    '''
    returns the indices and proposed labeling for nodes
    that are under nodes with k or more children
    e.g. 
    input_T =[[0,1,1,],
             [0,0,0],
             [0,0,0],]
    T = [[0,1,1,1,0,0,0],
         [0,0,0,0,1,0,1],
         [0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],]
    internal_node_idx_to_sites = {0:[1], 1:[0,1], 2:[1}
    k = 3
    include_children = True
    min_num_sites = 1
    returns ([0], [[1]]) (node at index 0 has 3 children, and the resolver node under it should be
    place in site 1, since node 0 and its child node 1 are both detected in site 1)

    if include_children is False, we only look at node 0's U values and not its children's U values
    '''
    row_sums = torch.sum(T, axis=1)
    node_indices = list(np.where(row_sums >= k)[0])
    filtered_node_indices = []
    all_resolver_sites = []
    for node_idx in node_indices:
        # get the sites that node_idx and node_idx's children are estimated in U
        resolver_sites = get_resolver_sites_of_node_and_children(input_T, internal_node_idx_to_sites, row_sums[node_idx], node_idx, include_children, min_num_sites, cutoff)
        # print("resolver_sites", resolver_sites)
        # a resolver node wouldn't help for this polytomy
        if len(resolver_sites) == 0:
            continue
        filtered_node_indices.append(node_idx)
        all_resolver_sites.append(resolver_sites)
    return filtered_node_indices, all_resolver_sites

def find_first_branching_point(adj_matrix):
    out_degrees = adj_matrix.sum(dim=1)  # Calculate out-degrees of each node
    for node, out_degree in enumerate(out_degrees):
        if out_degree > 1:
            return node
    return None  # Return None if no branching point is found

import torch
import networkx as nx
from itertools import product

# Helper function to build a NetworkX tree from an adjacency matrix
def build_tree(adj_matrix):
    G = nx.DiGraph()
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

# Helper function to find the first common branching point in two graphs
def find_first_branching_point(adj_matrix):
    out_degrees = adj_matrix.sum(dim=1)  # Calculate out-degrees of each node
    for node, out_degree in enumerate(out_degrees):
        if out_degree > 1:
            return node
    return None  # Return None if no branching point is found

# Helper function to extract subtrees rooted at the given node
def extract_subtrees(G, root):
    subtrees = []
    for child in G.successors(root):
        subtree = nx.dfs_tree(G, source=child)
        subtrees.append(subtree)
    return subtrees

# Helper function to combine subtrees and labels from multiple solutions
def combine_subtrees(subtrees_list, labels_list, common_root):
    combinations = []
    num_subtrees = len(subtrees_list[0])
    
    # Generate all combinations of subtrees from each solution
    for indices in product(range(num_subtrees), repeat=len(subtrees_list)):
        combined_adj_matrix = torch.zeros_like(subtrees_list[0][0].adj_matrix)
        combined_labels = torch.zeros_like(labels_list[0])
        
        for i, idx in enumerate(indices):
            subtree = subtrees_list[i][idx]
            labels = labels_list[i]
            # Combine adjacency matrices
            for edge in subtree.edges:
                combined_adj_matrix[edge[0], edge[1]] = 1
            # Combine labels
            for node in subtree.nodes:
                combined_labels[:, node] = labels[:, node]
        
        combinations.append((combined_adj_matrix, combined_labels))
    return combinations

def get_child_indices(T, indices):
    '''
    returns the indices of direct children of the nodes at indices
    '''
    all_child_indices = []

    for parent_idx in indices:
        children = np.where(T[parent_idx,:] > 0)[0]
        for child_idx in children:
            if child_idx not in all_child_indices:
                all_child_indices.append(child_idx)

    return all_child_indices

def find_parents_children(T, node):
    num_nodes = len(T)
    
    # Find parents: Look in the node's column
    parents = [i for i in range(num_nodes) if T[i][node] != 0]
    
    # Find children: Look in the node's row
    children = [j for j in range(num_nodes) if T[node][j] != 0]
    
    return parents, children

def repeat_n(x, n):
    '''
    Repeats tensor x 'n' times along the first axis, returning a tensor
    w/ dim (n, x.shape[0], x.shape[1])
    '''
    if n == 0:
        return x
    return x.repeat(n,1).reshape(n, x.shape[0], x.shape[1])

def path_matrix(T, remove_self_loops=False):
    '''
    T is a numpy ndarray or tensor adjacency matrix (where Tij = 1 if there is a path from i to j)
    '''
    bs = 1 if len(T.shape) == 2 else T.shape[0]
    # Path matrix that tells us if path exists from node i to node j
    I = torch.eye(T.shape[1]).repeat(bs, 1, 1)  # Repeat identity matrix along batch dimension
    B = torch.logical_or(T, I).int()  # Convert to int for more efficient matrix multiplication
    # Initialize path matrix with direct connections
    P = B.clone()
    
    # Floyd-Warshall algorithm
    for k in range(T.shape[1]):
        # Compute shortest paths including node k
        B = torch.logical_or(B, B[:, :, k].unsqueeze(2) & B[:, k, :].unsqueeze(1))
        # Update path matrix
        P |= B
        
    if remove_self_loops:
        P = torch.logical_xor(P, I.int())
    return P.squeeze(0) if len(T.shape) == 2 else P

def mutation_matrix(A):
    '''
    A is an numpy ndarray or tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return path_matrix(A.T, remove_self_loops=False)

def get_adj_matrix_from_edge_list(edge_list):
    T = []
    nodes = set([node for edge in edge_list for node in edge])
    T = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for edge in edge_list:
        T[ord(edge[0]) - 65][ord(edge[1]) - 65] = 1
    return torch.tensor(T, dtype = torch.float32)

def is_tree(adj_matrix):
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges)
    return (not nx.is_empty(g) and nx.is_tree(g))

def pareto_front(solutions, all_pars_metrics):
    """ 
    Args:
        - solutions: list of VertexLabelingSolutions of length n
        - all_pars_metrics: list of tuples with parsimony mentrics
        for (migration #, comigration #, seeding site #) of length n

    Returns:
        - a list of Pareto optimal parsimony metrics and VertexLabelingSolutions
    """
    
    # Start with an empty Pareto front
    pareto_front = []

    # Loop through each solution in the list
    for candidate_metric, soln in zip(all_pars_metrics, solutions):
        # Assume candidate is not dominated; check against all others
        if not any(all(other[0][i] <= candidate_metric[i] for i in range(len(candidate_metric))) and
                   any(other[0][i] < candidate_metric[i] for i in range(len(candidate_metric))) 
                   for other in pareto_front):
            # If no one in the current Pareto front dominates the candidate, add it
            pareto_front.append((candidate_metric, soln))
            # Remove any from the pareto_front that is dominated by the new candidate
            pareto_front = [front for front in pareto_front if not all(candidate_metric[i] <= front[0][i] for i in range(len(candidate_metric))) or not any(candidate_metric[i] < front[0][i] for i in range(len(candidate_metric)))]

    pareto_metrics = [front[0] for front in pareto_front]
    pareto_solutions = [front[1] for front in pareto_front]
    return pareto_metrics, pareto_solutions
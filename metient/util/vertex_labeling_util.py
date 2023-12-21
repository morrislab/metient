import numpy as np
import torch
import networkx as nx
import math
from queue import Queue

from metient.util.globals import *

import scipy.sparse as sp
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class LabeledTree:
    # TODO: remove tree from here
    def __init__(self, tree, labeling):
        if (tree.shape[0] != tree.shape[1]):
            raise ValueError("Adjacency matrix should have shape (num_nodes x num_nodes)")
        if (tree.shape[0] != labeling.shape[1]):
            raise ValueError("Vertex labeling matrix should have shape (num_sites x num_nodes)")

        self.tree = tree
        self.labeling = labeling

    def __eq__(self, other):
        return ( isinstance(other, LabeledTree) and 
                (torch.equal(torch.nonzero(self.labeling), torch.nonzero(other.labeling))) and
                (torch.equal(torch.nonzero(self.tree), torch.nonzero(other.tree))))

    def __hash__(self):
        return hash(self.labeling.numpy().tobytes())*hash(self.tree.numpy().tobytes())

    def __str__(self):
        A = str(np.where(self.tree == 1))
        V = str(np.where(self.labeling == 1))
        return f"Tree: {A}\nVertex Labeling: {V}"


def tree_iterator(T):
    '''
    iterate an adjacency matrix, returning i and j for all values = 1
    '''
    # enumerating through a torch tensor is pretty computationally expensive,
    # so convert to a sparse matrix to efficiently access non-zero values
    T = T if isinstance(T, np.ndarray) else T.detach().numpy()
    T = sp.coo_matrix(T)
    for i, j, val in zip(T.row, T.col, T.data):
        yield i,j


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
    returns the root idx (node with no inbound edges) from adjacency matrix T
    '''

    candidates = set([x for x in range(len(T))])
    for i, j in tree_iterator(T):
        candidates.remove(j)
    msg = "More than one" if len(candidates) > 1 else "No"
    assert (len(candidates) == 1), f"{msg} root node detected"

    return list(candidates)[0]

def restructure_matrices(adj_matrix,ref_matrix, var_matrix, node_idx_to_label, gen_dist_matrix):
    """
    Restructure the adjacency matrix, ref matrix and var matrix so that the node at index 0 becomes the root node.

    Parameters:
    - adj_matrix: 2D tensor representing the adjacency matrix.

    Returns:
    - Restructured adjacency matrix.
    """
    root_idx = get_root_index(adj_matrix)
    new_order = [x for x in range(len(adj_matrix)) if x != root_idx]
    new_order.insert(0, root_idx)

    # Swap rows and columns to move the first row and column to the desired position
    swapped_adjacency_matrix = adj_matrix[new_order, :][:, new_order]
    swapped_ref_matrix = ref_matrix[:, new_order]
    swapped_var_matrix = var_matrix[:, new_order]
    if gen_dist_matrix == None:
        swapped_gen_dist_matrix = None
    else:
        swapped_gen_dist_matrix = gen_dist_matrix[new_order, :][:, new_order]

    original_root_label = node_idx_to_label[root_idx]
    original_node0_label = node_idx_to_label[0]
    node_idx_to_label[0] = original_root_label
    node_idx_to_label[root_idx] = original_node0_label

    return swapped_adjacency_matrix, swapped_ref_matrix, swapped_var_matrix, node_idx_to_label, swapped_gen_dist_matrix


def top_k_integers_by_count(lst, k, thres, cutoff):
    # find unique sites and their counts
    unique_sites, counts = np.unique(lst, return_counts=True)
    # filter for sites that occur at least thres times
    filtered_values = unique_sites[counts >= thres]
    # recount after filtering
    unique_sites, counts = np.unique(filtered_values, return_counts=True)
    # sort based on counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    # get the top k integers and their counts
    if len(unique_sites) > k and cutoff:
        return list(unique_sites[sorted_indices[:k]])
        
    return list(unique_sites)

def get_resolver_sites_of_node_and_children(input_T, T, U, num_children, node_idx, thres, cutoff):
    '''
    Looking at the sites that node_idx and node_idx's children are in,
    figure out the sites that the resolver nodes should be initialized with
    (any sites that are detected >= 2 times)
    '''
    clone_tree_children = input_T[node_idx].nonzero()[:,0].tolist()
    # calculated based on clone tree children and leaf 
    # nodes estimated form U
    num_possible_resolvers = math.floor(num_children/2)
    U = U[:,1:] # don't include column for normal cells
    # get the sites that the node and node's children are in
    node_and_children = clone_tree_children+ [int(node_idx)]
    sites = (U[:,node_and_children] > U_CUTOFF).nonzero()[:,0]
    return top_k_integers_by_count(sites, num_possible_resolvers, thres, cutoff)
    
def get_k_or_more_children_nodes(input_T, T, U, k, thres, cutoff=True):
    '''
    returns the indices and proposed sites for resolver nodes
    that are placed under existing nodes with k or more children
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
    U = [[0,0,1,0,],
         [0,1,1,1,],]
    k = 3
    returns ([0], [[1]]) (node at index 0 has 3 children, and the resolver node under it should be
    place in site 1, since node 0 and its child node 1 are both detected in site 1)
    '''
    row_sums = torch.sum(T, axis=1)
    node_indices = list(np.where(row_sums >= k)[0])
    filtered_node_indices = []
    all_resolver_sites = []
    num_children = []
    for node_idx in node_indices:
        # get the sites that node_idx and node_idx's children are estimated in U
        resolver_sites = get_resolver_sites_of_node_and_children(input_T, T, U, row_sums[node_idx], node_idx, thres, cutoff)
        # a resolver node wouldn't help for this polytomy
        if len(resolver_sites) == 0:
            continue
        filtered_node_indices.append(node_idx)
        all_resolver_sites.append(resolver_sites)
    return filtered_node_indices, all_resolver_sites

# def get_k_or_more_children_nodes(T, k):
#     '''
#     returns the indices of nodes with k or more children
#     '''
#     row_sums = torch.sum(T, axis=1)
#     return list(np.where(row_sums >= k)[0])

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

def repeat_n(x, n):
    '''
    Repeats tensor x 'n' times along the first axis, returning a tensor
    w/ dim (n, x.shape[0], x.shape[1])
    '''
    if n == 0:
        return x
    return x.repeat(n,1).reshape(n, x.shape[0], x.shape[1])

def get_path_matrix(T, remove_self_loops=False):
    bs = 0 if len(T.shape) == 2 else T.shape[0]
    # Path matrix that tells us if path exists from node i to node j
    I = repeat_n(torch.eye(T.shape[1]), bs)
    # M is T with self loops.
    # Convert to bool to get more efficient matrix multiplicaton
    B = torch.logical_or(T,I).int()
    # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
    # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
    k = np.ceil(np.log2(T.shape[1]))
    for i in range(int(k)):
        B = torch.matmul(B, B)
    if remove_self_loops:
        B = torch.logical_xor(B,I)
    P = torch.sigmoid(BINARY_ALPHA * (2*B - 1))
    return P

# def adj_matrix_hash(T):
#     sparse_T = sp.coo_matrix(T.detach().numpy())
#     return hash((str(sparse_T.row), str(sparse_T.col)))

# def get_single_adj_matrix_path_matrix(T, remove_self_loops):
#     # Path matrix that tells us if path exists from node i to node j
#     I = torch.eye(T.shape[1])
#     # M is T with self loops.
#     # Convert to bool to get more efficient matrix multiplicaton
#     B = torch.logical_or(T,I).int()
#     # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
#     # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
#     k = np.ceil(np.log2(T.shape[1]))
    
#     for i in range(int(k)):
#         B = torch.matmul(B, B)
#     if remove_self_loops:
#         B = torch.logical_xor(B,I)
#     P = torch.sigmoid(BINARY_ALPHA * (2*B - 1))
#     return P
    
# def get_path_matrix(T, remove_self_loops=False):
#     # no need to do any caching stuff when it's just the singular adjacency matrix
#     if len(T.shape) == 2:
#         T = T.reshape(1, T.shape[0], T.shape[1])
#         print(T.shape)
    
#     num_times_found = 0
#     global PATH_MATRICES_CACHE
#     final_P = torch.zeros(T.shape)
#     for bs_idx in range(T.shape[0]):
#         hsh = adj_matrix_hash(T[bs_idx])
#         if hsh in PATH_MATRICES_CACHE:
#             final_P[bs_idx] = PATH_MATRICES_CACHE[hsh]
#             num_times_found += 1
#         else:
#             P = get_single_adj_matrix_path_matrix(T[bs_idx], remove_self_loops)
#             final_P[bs_idx] = P
#             PATH_MATRICES_CACHE[hsh] = P
#     return final_P

def get_path_matrix_tensor(A):
    '''
    A is a numpy adjacency matrix
    '''
    return torch.tensor(get_path_matrix(A, remove_self_loops=True), dtype = torch.float32)

def get_mutation_matrix_tensor(A):
    '''
    A is an numpy ndarray or tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return torch.tensor(get_path_matrix(A.T, remove_self_loops=False), dtype = torch.float32)

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
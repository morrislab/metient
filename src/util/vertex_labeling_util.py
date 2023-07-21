import numpy as np
import torch
import networkx as nx

from src.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class LabeledTree:
    def __init__(self, tree, labeling, U, branch_lengths):
        if (tree.shape[0] != tree.shape[1]):
            raise ValueError("Adjacency matrix should have shape (num_nodes x num_nodes)")
        if (tree.shape[0] != labeling.shape[1]):
            raise ValueError("Vertex labeling matrix should have shape (num_sites x num_nodes)")

        self.tree = tree
        self.labeling = labeling
        self.U = U
        self.branch_lengths = branch_lengths

    def __eq__(self, other):
        return ( isinstance(other, LabeledTree) and 
                (self.tree.numpy().tobytes() == other.tree.numpy().tobytes()) and
                (self.labeling.numpy().tobytes() == other.labeling.numpy().tobytes()))

    def __hash__(self):
        return hash((self.tree.numpy().tobytes(), self.labeling.numpy().tobytes()))

def get_path_matrix(T, remove_self_loops=False):
    # Path matrix that tells us if path exists from node i to node j
    I = np.identity(T.shape[0])
    # M is T with self loops.
    # Convert to bool to get more efficient matrix multiplicaton
    B = np.logical_or(T,I).astype(bool)
    # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
    # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
    k = np.ceil(np.log2(len(T)))
    for i in range(int(k)):
        B = np.dot(B, B)
    if remove_self_loops:
        B = np.logical_xor(B,I)
    P = B.astype(int)
    return P

def adj_matrix_hash(A):
    return hash((str(np.where(A == 1)[0]), str(np.where(A == 1)[1])))

# Adapted from: https://forum.kavli.tudelft.nl/t/caching-of-python-functions-with-array-input/59/6
# Caching path matrices because it's expensive to compute.
# TODO: Consider using dask or something similar in the future
def np_array_cache(function):
    cache = {}

    @wraps(function)
    def wrapped(array):
        hsh = adj_matrix_hash(array)

        if hsh not in cache:
            cache[hsh] = function(array)
        # Not using built-in hash because it's prone to collisions.
        return cache[hsh]

    return wrapped

# TODO: figure out why this isn't speeding things up :/
#@np_array_cache
def get_path_matrix_tensor(A):
    '''
    A is a numpy adjacency matrix
    '''
    return torch.tensor(get_path_matrix(A, remove_self_loops=True), dtype = torch.float32)


def get_mutation_matrix_tensor(A):
    '''
    A is a tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return torch.tensor(get_path_matrix(A.cpu().numpy().T, remove_self_loops=False), dtype = torch.float32)

def get_adj_matrix_from_edge_list(edge_list):
    T = []
    G = nx.DiGraph()
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
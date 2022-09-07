import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import random
import torch
from torch.autograd import Variable
import sys
import matplotlib.patches as mpatches
import pydot
from IPython.display import Image, display
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source

import string

from functools import wraps
import hashlib

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)

def plot_migration_graph(V, full_tree, ordered_sites, custom_colors, primary, show=True):
    '''
    Plots migration graph G which represents the migrations/comigrations between
    all anatomical sites.

    Returns a list of edges (e.g. [('P' ,'M1'), ('P', 'M2')])
    '''
    colors = custom_colors
    if colors == None:
        colors = plt.get_cmap("Set3").colors
    assert(len(ordered_sites) <= len(colors))

    migration_graph = (V @ full_tree) @ V.T
    migration_graph_no_diag = torch.mul(migration_graph, 1-torch.eye(migration_graph.shape[0], migration_graph.shape[1]))

    G = nx.MultiDiGraph()
    for node, color in zip(ordered_sites, colors):
        G.add_node(node, shape="box", color=color, fillcolor='white', fontname="Helvetica", penwidth=3.0)

    edges = []
    for i, adj_row in enumerate(migration_graph_no_diag):
        for j, num_edges in enumerate(adj_row):
            if num_edges > 0:
                for _ in range(int(num_edges.item())):
                    G.add_edge(ordered_sites[i], ordered_sites[j], color=f'"{colors[i]};0.5:{colors[j]}"', penwidth=3)
                    edges.append((ordered_sites[i], ordered_sites[j]))

    dot = nx.nx_pydot.to_pydot(G)
    if show:
        view_pydot(dot)

    return edges

def plot_tree(V, T, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):
    pastel_colors = plt.get_cmap("Set3").colors
    assert(len(ordered_sites) < len(pastel_colors))

    # custom_node_idx_to_label only gives the internal node labels
    full_node_idx_to_label_map = dict()
    for i, adj_row in enumerate(T):
        for j, val in enumerate(adj_row):
            if val == 1:
                if custom_node_idx_to_label != None:
                    if i in custom_node_idx_to_label:
                        full_node_idx_to_label_map[i] = custom_node_idx_to_label[i]
                    if j in custom_node_idx_to_label:
                        full_node_idx_to_label_map[j] = custom_node_idx_to_label[j]
                    elif j not in custom_node_idx_to_label:
                        site_idx = np.where(V[:,j] == 1)[0][0]
                        full_node_idx_to_label_map[j] = f"{custom_node_idx_to_label[i]}_{ordered_sites[site_idx]}"
                else:
                    full_node_idx_to_label_map[i] = chr(i+65)
                    full_node_idx_to_label_map[j] = chr(j+65)

    def idx_to_color(idx):
        if custom_colors != None:
            return custom_colors[idx]
        return pastel_colors[idx]

    patches = []
    for i, site in enumerate(ordered_sites):
        patch = mpatches.Patch(color=idx_to_color(i), label=site)
        patches.append(patch)

    color_map = { full_node_idx_to_label_map[i]:idx_to_color(np.where(V[:,i] == 1)[0][0]) for i in range(V.shape[1])}
    G = nx.DiGraph()
    edges = []
    for i, adj_row in enumerate(T):
        for j, val in enumerate(adj_row):
            if val == 1:
                label_i = full_node_idx_to_label_map[i]
                label_j = full_node_idx_to_label_map[j]
                edges.append((label_i, label_j))
                G.add_node(label_i, color=color_map[label_i], penwidth=3)
                G.add_node(label_j, color=color_map[label_j], penwidth=3)
                G.add_edge(label_i, label_j, color=f'"{color_map[label_i]};0.5:{color_map[label_j]}"', penwidth=3)

    assert(nx.is_tree(G))

    nodes = [full_node_idx_to_label_map[i] for i in range(len(T))]

    dot = to_pydot(G).to_string()
    src = Source(dot) # dot is string containing DOT notation of graph
    if show:
        display(src)

    vertex_name_to_site_map = { full_node_idx_to_label_map[i]:ordered_sites[np.where(V[:,i] == 1)[0][0]] for i in range(V.shape[1])}
    return edges, vertex_name_to_site_map

def write_tree(tree_edge_list, output_filename):
    '''
    Writes the full tree to file like so:
    GL 0
    0 1
    1 2;3
    '''
    tree_edge_list.append(('GL', tree_edge_list[0][0]))
    with open(output_filename, 'w') as f:
        for edge in tree_edge_list:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")

def write_tree_vertex_labeling(vertex_name_to_site_map, output_filename):
    '''
    Writes the full tree's vertex labeling to file like so:
    GL P
    1 P
    1_P P
    25;32_M1 M1
    '''
    vertex_name_to_site_map['GL'] = "P"
    with open(output_filename, 'w') as f:
        for vert_label in vertex_name_to_site_map:
            f.write(f"{vert_label} {vertex_name_to_site_map[vert_label]}")
            f.write("\n")

def write_migration_graph(migration_edge_list, output_filename):
    '''
    Writes the full migration graph to file like so:
    P M1
    P M2
    P M1
    M1 M2
    '''
    with open(output_filename, 'w') as f:
        for edge in migration_edge_list:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")

def plot_losses(losses):
    plt.plot([x for x in range(len(losses))],losses, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

# TODO: write unit tests for these
# TODO: figure out why this isn't speeding things up :/
#@np_array_cache
# def get_path_matrix_tensor(A, remove_self_loops=False):
#     '''
#     A is a tensor adjacency matrix
#     '''
#     # Path matrix that tells us if path exists from node i to node j
#     I = torch.eye(A.shape[0])
#     print("A\n", A)
#     print("I\n", I)
#     # M is T with self loops.
#     # Convert to bool to get more efficient matrix multiplicaton
#     B = torch.logical_or(A,I)
#     # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
#     # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
#     k = np.ceil(np.log2(len(A)))
#     for i in range(int(k)):
#         B = torch.matmul(B, B)
#     if remove_self_loops:
#         B = torch.logical_xor(B,I)
#     #P = B.astype(int)
#     print("B\n", B)
#     return B

# TODO: pytorch doesn't have boolean matrix multiplication support
# which is why I convert to numpy here, but this is inefficient
def get_path_matrix_tensor(A, remove_self_loops=False):
    '''
    A is a tensor adjacency matrix
    '''
    A = A.numpy()

    # Path matrix that tells us if path exists from node i to node j
    I = np.identity(A.shape[0])
    # A with self loops
    M = np.logical_or(A,I)
    # Path matrix that tells us if path exists from node i to node j
    P = np.linalg.matrix_power(M, len(A) - 1)
    if remove_self_loops:
        P = np.logical_xor(P,I)
    return torch.tensor(P)

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

def get_mutation_matrix_tensor(A):
    '''
    A is a tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return get_path_matrix_tensor(A.T, remove_self_loops=False)

def get_adj_matrix_from_edge_list(edge_list):
    T = []
    G = nx.DiGraph()
    nodes = set([node for edge in edge_list for node in edge])
    T = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for edge in edge_list:
        T[ord(edge[0]) - 65][ord(edge[1]) - 65] = 1
    return torch.tensor(T, dtype = torch.float32)

# Taken from pairtree
def convert_parents_to_adjmatrix(parents):
    K = len(parents) + 1
    adjm = np.eye(K)
    adjm[parents,np.arange(1, K)] = 1
    return adjm

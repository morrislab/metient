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

import metient.util.vertex_labeling_util as vutil 
from metient.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

FONT = "Arial"

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def is_cyclic(G):
    '''
    returns True if graph contains cycles
    '''
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
    '''
    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    '''
    if torch.all(G == 0):
        return "n/a"
    return "polyclonal" if ((G > 1).any()) else  "monoclonal"

def site_clonality(V, A):
    '''
    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    '''
    V, A = prep_V_A_inputs(V, A)
    G = migration_graph(V, A)
    return site_clonality_with_G(G)

def genetic_clonality(V, A):
    '''
    Returns monoclonal if every site is seeded by the *same* clone,
    else returns polyclonal.
    '''
    V, A = prep_V_A_inputs(V, A)
    all_seeding_clusters = seeding_clusters(V, A)
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
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    returns: verbal description of the seeding pattern, one of:
    {primary single-source, single-source, multi-source, reseeding}
    '''
    G = migration_graph(V, A)
    return seeding_pattern_with_G(G)
    

def migration_edges(V, A, sites=None):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    sites: optional, set of indices in range [0,num_sites) that you restrict migration edges to
    Returns:
        Returns a matrix where Yij = 1 if there is a migration edge from node i to node j
    '''
    X = V.T @ V 
    Y = torch.mul(A, (1-X))
    # Remove migration edges to sites that we're not interested in 
    if sites != None:
        for i,j in vutil.tree_iterator(Y):
            k = (V[:,j] == 1).nonzero()[0][0].item()
            if k not in sites:
                Y[i,j] = 0
    return Y

def seeding_clusters(V, A, sites=None):
    '''
    returns: list of nodes whose child is a different color
    '''
    V, A = prep_V_A_inputs(V, A)
    Y = migration_edges(V,A, sites)
    seeding_clusters = torch.nonzero(Y.any(dim=1)).squeeze()
    # Check if it's a scalar (0D tensor)
    if seeding_clusters.dim() == 0:
        # Convert to a 1D tensor (vector)
        seeding_clusters = seeding_clusters.unsqueeze(0)
    seeding_clusters = [int(x) for x in seeding_clusters]
    return seeding_clusters

def mrca(adj_matrix, nodes_to_check):
    '''
    Gets the most recent common ancestor of nodes in nodes_to_check
    '''
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
                 
def phyleticity(V, A, sites=None):
    '''
    If all nodes can be reached from the top level node in the seeding clusters,
    returns monophyletic, else polyphyletic
    '''
    def dfs(node, target):
        visited[node] = True
        if node == target:
            return True
        for neighbor, connected in enumerate(A[node]):
            if connected and not visited[neighbor] and dfs(neighbor, target):
                return True
        return False
    
    V, A = prep_V_A_inputs(V, A)
    clonality = genetic_clonality(V,A)
    all_seeding_clusters = seeding_clusters(V, A, sites)
    if "monoclonal" in clonality:
        return "monophyletic"

    num_nodes = len(A)
    visited = [False] * num_nodes
    highest_node = mrca(A, all_seeding_clusters)
    
    # Check if all nodes can be reached from the top level node in the seeding
    # nodes (seeding node that is closest to the root)
    for node in all_seeding_clusters:
        visited = [False] * num_nodes
        if not dfs(highest_node, node):
            return "polyphyletic"
    return "monophyletic"

def tracerx_phyleticity(V, A):
    '''
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
    '''
    V, A = prep_V_A_inputs(V, A)
    clonality = genetic_clonality(V,A)
    all_seeding_clusters = seeding_clusters(V, A)

    is_hamiltonian = has_hamiltonian_path_with_set(A, all_seeding_clusters)
    if "monoclonal" in clonality:
        return "monophyletic"
    phyleticity = "monophyletic" if is_hamiltonian else "polyphyletic"
    return phyleticity
    
def tracerx_seeding_pattern(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Monoclonal if only one clone seeds met(s), else polyclonal
    Monophyletic if there is a Hamiltonian path connecting all seeding clusters
    returns: one of {no seeding, monoclonal monophyletic, polyclonal polyphyletic, polyclonal monophyletic}
    '''
    V, A = prep_V_A_inputs(V, A)
    G = migration_graph(V, A)
    non_zero = torch.where(G > 0)
    source_sites = non_zero[0]
    if len(torch.unique(source_sites)) == 0:
        return "no seeding"

    # 1) determine if monoclonal (only one clone seeds met(s)), else polyclonal
    clonality = genetic_clonality(V,A)
    
    # 2) determine if monophyletic or polyphyletic
    phyleticity = tracerx_phyleticity(V, A)

    return clonality + phyleticity

def write_tree(tree_edge_list, output_filename, add_germline_node=False):
    '''
    Writes the full tree to file like so:
    0 1
    1 2;3
    '''
    if add_germline_node:
        tree_edge_list.append(('GL', tree_edge_list[0][0]))
    with open(output_filename, 'w') as f:
        for edge in tree_edge_list:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")

def write_tree_vertex_labeling(vertex_name_to_site_map, output_filename, add_germline_node=False):
    '''
    Writes the full tree's vertex labeling to file like so:
    1 P
    1_P P
    25;32_M1 M1
    '''
    if add_germline_node:
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

def full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites, shorten_label=True, to_string=False):
    '''
    custom_node_idx_to_label only gives the internal node labels, so build a map of
    node_idx to (label, is_leaf) 
    e.g. ("0;9", False), ("0;9_P", True),  or ("5_liver", True) 
    '''
    full_node_idx_to_label_map = dict()
    for i, j in vutil.tree_iterator(T):
        if i in custom_node_idx_to_label:
            full_node_idx_to_label_map[i] = (pruned_mut_label(custom_node_idx_to_label[i], shorten_label, to_string), False)
        if j in custom_node_idx_to_label:
            full_node_idx_to_label_map[j] = (pruned_mut_label(custom_node_idx_to_label[j], shorten_label, to_string), False)
        elif j not in custom_node_idx_to_label: # observed clone leaf node
            site_idx = (V[:,j] == 1).nonzero()[0][0].item()
            labels = copy.deepcopy(custom_node_idx_to_label[i])
            labels.append(ordered_sites[site_idx])
            full_node_idx_to_label_map[j] = (pruned_mut_label(labels, shorten_label, to_string), True)
    return full_node_idx_to_label_map

def idx_to_color(custom_colors, idx, alpha=1.0):
    rgb = mcolors.to_rgb(custom_colors[idx])
    rgb_alpha = (rgb[0], rgb[1], rgb[2], alpha)
    return mcolors.to_hex(rgb_alpha, keep_alpha=True)

def prep_V_A_inputs(V, A):
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V)
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    return V, A

def migration_graph(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    '''
    V, A = prep_V_A_inputs(V, A)
    migration_graph = (V @ A) @ V.T
    migration_graph_no_diag = torch.mul(migration_graph, 1-torch.eye(migration_graph.shape[0], migration_graph.shape[1]))
    
    return migration_graph_no_diag

def find_abbreviation_mark(input_string):
    abbreviation_marks = [',', '-', '|']  # Add other abbreviation marks as needed

    for mark in abbreviation_marks:
        if input_string.count(mark) == 1:
            return mark
    return None

def plot_migration_graph(V, A, ordered_sites, custom_colors, show=True):
    '''
    Plots migration graph G which represents the migrations/comigrations between
    all anatomical sites.

    Returns a list of edges (e.g. [('P' ,'M1'), ('P', 'M2')])
    '''
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

    edges = []
    for i, adj_row in enumerate(mig_graph_no_diag):
        for j, num_edges in enumerate(adj_row):
            if num_edges > 0:
                for _ in range(int(num_edges.item())):
                    G.add_edge(fmted_ordered_sites[i], fmted_ordered_sites[j], color=f'"{custom_colors[i]};0.5:{custom_colors[j]}"', penwidth=3)
                    edges.append((fmted_ordered_sites[i], fmted_ordered_sites[j]))

    dot = nx.nx_pydot.to_pydot(G)
    if show:
        view_pydot(dot)

    dot_lines = dot.to_string().split("\n")
    dot_lines.insert(1, 'dpi=600;size=3.5;')
    dot_str = ("\n").join(dot_lines)

    return dot_str, edges

def plot_tree(V, T, gen_dist, ordered_sites, custom_colors, custom_node_idx_to_label=None, show=True):

    # (1) Create full directed graph 
    # these labels are used for display in plotting
    display_node_idx_to_label_map = full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites,
                                                                    shorten_label=True, to_string=True)
    # these labels are used for writing out full vertex names to file
    full_node_idx_to_label_map = full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites,
                                                                 shorten_label=False, to_string=False)
    color_map = { i:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":FONT, "fontsize":"12pt",
                    "fixedsize":"true", "height":0.25}

    if gen_dist != None:
        gen_dist = ((gen_dist / torch.max(gen_dist[gen_dist>0]))*1.5) + 1

    edges = []
    for i, j in vutil.tree_iterator(T):
        label_i, _ = display_node_idx_to_label_map[i]
        label_j, is_leaf = display_node_idx_to_label_map[j]
        
        G.add_node(i, xlabel=label_i, fillcolor=color_map[i], 
                    color=color_map[i], style="filled", **node_options)
        G.add_node(j, xlabel="" if is_leaf else label_j, fillcolor=color_map[j], 
                    color=color_map[j], style="solid" if is_leaf else "filled", **node_options)

        style = "dashed" if is_leaf else "solid"
        penwidth = 5 if is_leaf else 5.5

        # We're iterating through i,j of the full tree (including leaf nodes),
        # while G only has genetic distances between internal nodes
        minlen = gen_dist[i, j].item() if (gen_dist != None and i < len(gen_dist) and j < len(gen_dist)) else 1.0

        G.add_edge(i, j,color=f'"{color_map[i]};0.5:{color_map[j]}"', 
                   penwidth=penwidth, arrowsize=0, style=style, minlen=minlen)

        edges.append((full_node_idx_to_label_map[i][0], full_node_idx_to_label_map[j][0]))

    # Add edge from normal to root 
    # print("T", vutil.adjacency_matrix_to_edge_list(T))
    root_idx = vutil.get_root_index(T)
    root_label = display_node_idx_to_label_map[root_idx][0]
    G.add_node("normal", label="", xlabel=root_label, penwidth=3, style="invis")
    G.add_edge("normal", root_idx, label="", 
                color=f'"{color_map[root_idx]}"', 
                penwidth=4, arrowsize=0, style="solid")

    assert(nx.is_tree(G))

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    dot.insert(1, 'graph[splines=false]; nodesep=0.7; rankdir=TB; ranksep=0.6; forcelabels=true; dpi=600; size=2.5;')
    dot_str = ("\n").join(dot)

    if show:
        dot = nx.nx_pydot.to_pydot(dot_str)
        view_pydot(dot)

    vertex_name_to_site_map = { ";".join(full_node_idx_to_label_map[i][0]):ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return dot_str, edges, vertex_name_to_site_map, full_node_idx_to_label_map

def construct_loss_dict(V, soft_V, T, G, O, p, full_loss):
    V = V.reshape(1, V.shape[0], V.shape[1]) # add batch dimension
    soft_V = soft_V.reshape(1, soft_V.shape[0], soft_V.shape[1]) # add batch dimension
    m, c, s, g, o = vutil.ancestral_labeling_metrics(V, T, G, O, p, True)
    e = vutil.calc_entropy(V, soft_V) # this entropy is saved only based on second round of optimization
    loss_dict = {MIG_KEY: m, COMIG_KEY:c, SEEDING_KEY: s, ORGANOTROP_KEY: o, GEN_DIST_KEY: g, ENTROPY_KEY: e}
    loss_dict = {**loss_dict, **{FULL_LOSS_KEY: round(torch.mean(full_loss).item(), 3)}}
    return loss_dict

def convert_lists_to_np_arrays(pickle_outputs, keys):
    '''
    makes unpickling these outputs much faster
    '''
    for key in keys:
        if key in pickle_outputs:
            pickle_outputs[key] = np.array(pickle_outputs[key])

    return pickle_outputs

def figure_output_pattern(V, A):
    '''
    
    '''
    gen_clonality = genetic_clonality(V, A).replace("clonal", "")
    st_clonality = site_clonality(V, A).replace("clonal", "")
    pattern = seeding_pattern(V, A)
    phyletic = phyleticity(V, A)
    output_str = f"{pattern}, {phyletic}\n"
    output_str += f"genetic clonality: {gen_clonality}, site clonality: {st_clonality}\n"
    return output_str


def save_best_trees(min_loss_solutions, U, O, weights, ordered_sites,
                    print_config, custom_colors, primary, output_dir, 
                    run_name, original_root_idx=-1):
    '''
    min_loss_solutions is in order from lowest to highest loss 

    original_root_idx: if not -1, swap the original_root_idx with 0 in all
    data that we save that involves node/cluster indices. This will then match
    the inputs from the user's again.
    '''
    
    primary_idx = ordered_sites.index(primary)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites)).T

    ret = None
    figure_outputs = []
    pickle_outputs = {OUT_LABElING_KEY:[], OUT_LOSSES_KEY:[],OUT_IDX_LABEL_KEY:[],
                      OUT_ADJ_KEY:[], OUT_SITES_KEY:ordered_sites, OUT_LOSS_DICT_KEY:[],
                      OUT_PRIMARY_KEY:primary, 
                      OUT_SOFTV_KEY:[], OUT_GEN_DIST_KEY:[]}

    with torch.no_grad():
        if custom_colors == None:
            custom_colors = DEFAULT_COLORS
            # Reorder so that green is always the primary
            green_idx = custom_colors.index(DEFAULT_GREEN)
            custom_colors[primary_idx], custom_colors[green_idx] = custom_colors[green_idx], custom_colors[primary_idx]

        for i, min_loss_solution in enumerate(min_loss_solutions):
            V = min_loss_solution.V
            soft_V = min_loss_solution.soft_V
            T = min_loss_solution.T
            G = min_loss_solution.G
            full_loss = min_loss_solution.loss
            node_idx_to_label = min_loss_solution.idx_to_label
            loss_dict = construct_loss_dict(V, soft_V, T, G, O, p, full_loss)
            
            # Restructure adjacency matrices so that node indices match the cluster indices
            # that the user originally input (we restructure them s.t. root index is 0 during
            # inference to make indexing logic much simpler)
            if original_root_idx != -1:
                T, _, _, node_idx_to_label, G, _, V, U = vutil.restructure_matrices(0, original_root_idx, T, None, None, node_idx_to_label, G, None, V, U)
            tree_dot, edges, vertices_to_sites_map, full_tree_idx_to_label = plot_tree(V, T, G, ordered_sites, custom_colors, node_idx_to_label, show=False)
            mig_graph_dot, mig_graph_edges = plot_migration_graph(V, T, ordered_sites, custom_colors, show=False)

            pattern = figure_output_pattern(V, T)
            figure_outputs.append((tree_dot, mig_graph_dot, loss_dict, pattern))
            pickle_outputs[OUT_LABElING_KEY].append(V.detach().numpy())
            pickle_outputs[OUT_LOSSES_KEY].append(full_loss.numpy())
            pickle_outputs[OUT_ADJ_KEY].append(T.detach().numpy())
            pickle_outputs[OUT_SOFTV_KEY].append(soft_V.numpy())
            pickle_outputs[OUT_OBSERVED_CLONES_KEY] = U.numpy() if U != None else np.array([])

            if G != None:
                pickle_outputs[OUT_GEN_DIST_KEY].append(G.numpy())                
            pickle_outputs[OUT_LOSS_DICT_KEY].append(loss_dict)
            pickle_outputs[OUT_IDX_LABEL_KEY].append(full_tree_idx_to_label)
            if i == 0: # Best tree
                ret = (edges, vertices_to_sites_map, mig_graph_edges, loss_dict)

        #pickle_outputs = convert_lists_to_np_arrays(pickle_outputs, [OUT_LABElING_KEY, OUT_LOSSES_KEY, OUT_ADJ_KEY, OUT_SOFTV_KEY, OUT_GEN_DIST_KEY])

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
        if n < k:
            was = "was" if n==1 else "were"
            print(f"{k} unique trees were not found ({n} {was} found). Retry with a higher sample size if you want to get more trees.")
            k = n

        max_trees = 20
        if n > max_trees:
            print("More than 20 solutions detected, only plotting top 20 trees.")
            k = max_trees
        # Create a figure and subplots
        #fig, axs = plt.subplots(3, k*2, figsize=(10, 8))

        plt.suptitle(run_name)

        z = 2 # number of trees displayed per row

        nrows = math.ceil(k/z)
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

            gs1 = gridspec.GridSpec(3, 3)

            row = math.floor(i/2)
            pad = 0.02 if k < 20 else 0.001

            # left = 0.0 if i is odd, 0.55 if even
            # right = 0.45 if i is odd, 1.0 if even
            gs1.update(left=0.0+((i%2)*0.53), right=0.47+0.55*(i%2), top=1-(row*vspace)-pad, bottom=1-((row+1)*vspace)+pad, wspace=0.05)
            ax1 = plt.subplot(gs1[:-1, :])
            ax2 = plt.subplot(gs1[-1, :-1])
            ax3 = plt.subplot(gs1[-1, -1])

            # Render and display each graph
            ax1.imshow(tree)
            ax1.axis('off')
            ax1.set_title(f'Solution {i+1}\n{seeding_pattern}', fontsize=7, loc="left", va="top", x=-0.1, y=1.0)

            ax2.imshow(mig_graph)
            ax2.axis('off')

            ax3.text(0.5, 0.5, formatted_loss_string(loss_info, weights), ha='center', va='center', fontsize=7)
            ax3.axis('off')

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
        with open(os.path.join(output_dir, f"{run_name}.tree.dot"), 'w') as file:
            file.write(tree_dot)
        with open(os.path.join(output_dir, f"{run_name}.mig_graph.dot"), 'w') as file:
            file.write(mig_graph_dot)
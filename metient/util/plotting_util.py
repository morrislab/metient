import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import pydot
import torch
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source
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
import string

# TODO: this cyclical import is not great
import metient.lib.vertex_labeling as vert_label
from metient.util.vertex_labeling_util import LabeledTree, get_root_index, tree_iterator
from metient.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

import seaborn as sns

COLORS = ["#6aa84fff","#c27ba0ff", "#e69138ff", "#be5742e1", "#2496c8ff", "#674ea7ff"] + sns.color_palette("Paired").as_hex()


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
            if G[node, neighbor] == 1:
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

def get_seeding_pattern(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    returns: verbal description of the seeding pattern, one of:
    {monoclonal, polyclonal} {single-source, multi-source, reseeding}
    '''
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V)
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)

    G = get_migration_graph(V, A)
    pattern = ""
    # 1) determine if monoclonal (no multi-eges) or polyclonal (multi-edges)
    pattern = "polyclonal " if ((G > 1).any()) else  "monoclonal "

    # 2) determine if single-source seeding (all incoming edges to a site in G 
    # originate from the same site) OR multi-source seeding (at least one site is 
    # seeded from multiple other sites) OR (R) reseeding (at least one site seeds 
    # its originating site)
    non_zero = torch.where(G > 0)
    source_sites = non_zero[0]
    dest_sites = non_zero[1]
    binarized_G = (G != 0).to(torch.int)

    col_sums = torch.sum(binarized_G, axis=0)
    # single-source means that each site is only seeded by ONE other site 
    # (not that seeding site is 1)
    is_single_source = torch.all(col_sums <= 1).item()

    unique_source_sites = torch.unique(source_sites)

    if len(unique_source_sites) == 0:
        return "no seeding"
    elif is_cyclic(G):
        pattern += "reseeding"
    elif len(unique_source_sites) == 1 or is_single_source:
        pattern += "single-source seeding"
    else:
        pattern += "multi-source seeding"
    return pattern

def get_verbose_seeding_pattern(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    returns: one of: {monoclonal, polyclonal} {primary single-source, single-source, multi-source, reseeding}
    '''
    pattern = get_seeding_pattern(V, A)
    
    G = get_migration_graph(V, A)
    non_zero = torch.where(G > 0)    
    unique_source_sites = torch.unique(non_zero[0])

    if len(unique_source_sites) == 1:
        items = pattern.split(" ")
        items.insert(1, "primary")
        return (" ").join(items)
    return pattern

def get_migration_edges(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns:
        Adjacency matrix where Aij = 1 if there is a migration edge between nodes i and j
    '''
    X = V.T @ V 
    Y = torch.mul(A, (1-X))
    return Y

def get_shared_clusters(V, A, ordered_sites, primary_site, full_node_idx_to_label):
    '''
    returns: list of list of lists with dim: (len(ordered_sites), len(ordered_sites), len(shared_clusters))
    where the innermost list contains the clusters that are shared between site i and site j
    '''
    Y = get_migration_edges(V,A)
    
    shared_clusters = [[[] for x in range(len(ordered_sites))] for y in range(len(ordered_sites))]
    for i,j in tree_iterator(Y):
        site_i = (V[:,i] == 1).nonzero()[0][0].item()
        site_j = (V[:,j] == 1).nonzero()[0][0].item()
        assert(site_i != site_j)
        # if j is a subclonal presence leaf node, add i as the shared cluster 
        # (b/c i is the mutation cluster that j represents)
        if full_node_idx_to_label[j][1] == True:
            shared_clusters[site_i][site_j].append(i)
        else:
            shared_clusters[site_i][site_j].append(j)
    return shared_clusters

def find_highest_level_node(adj_matrix, nodes_to_check):
    start_node = get_root_index(adj_matrix)
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
    
def is_monophyletic(adj_matrix, nodes_to_check):
    def dfs(node, target):
        visited[node] = True
        if node == target:
            return True
        for neighbor, connected in enumerate(adj_matrix[node]):
            if connected and not visited[neighbor] and dfs(neighbor, target):
                return True
        return False

    # Initialize variables
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    highest_node = find_highest_level_node(adj_matrix, nodes_to_check)
    if highest_node == get_root_index(adj_matrix):
        return False
    # Check if all nodes can be reached from the top level node in the seeding
    # nodes (seeding node that is closest to the root)
    for node in nodes_to_check:
        visited = [False] * num_nodes
        if not dfs(highest_node, node):
            return False
    return True
    
def get_tracerx_seeding_pattern(V, A, ordered_sites, primary_site, full_node_idx_to_label):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    ordered_sites: list of the anatomical site names (e.g. ["breast", "lung_met"]) 
    with length =  num_anatomical_sites) and the order matches the order of cols in V
    primary_site: name of the primary site (must be an element of ordered_sites)

    TRACERx has a different definition of monoclonal vs. polyclonal:
    "If only a single metastatic sample was considered for a case, the case-level 
    dissemination pattern matched the metastasis level dissemination pattern. 
    If multiple metastases were sampled and the dissemination pattern of any 
    individual metastatic sample was defined as polyclonal, the case-level 
    dissemination pattern was also defined as polyclonal. Conversely,if all metastatic 
    samples follow a monoclonal dissemination pattern, all shared clusters between 
    the primary tumour and each metastasis were extracted. If all shared clusters 
    overlapped across all metastatic samples, the case-level dissemination pattern 
    was classified as monoclonal, whereas,  if any metastatic sample shared 
    additional clusters with the primary tumour, the overall dissemination pattern 
    was defined as polyclonal."

    and they define monophyletic vs. polyphyletics as:
    "the origin of the seeding clusters was determined as monophyletic if all 
    clusters appear along a single branch, and polyphyletic if clusters were
    spread across multiple branches of the phylogenetic tree. Thus, if a 
    metastasis was defined as monoclonal, the origin was necessarily monophyletic. 
    For polyclonal metastases, the clusters were mapped to branches of the 
    evolutionary tree. If multiple branches were found, the origin was determined 
    to be polyphyletic, whereas, if only a single branch gave rise to all shared 
    clusters, the origin was defined as monophyletic."
    (from https://www.nature.com/articles/s41586-023-05729-x#Sec7)

    returns: verbal description of the seeding pattern
    '''
    
    Y = get_migration_edges(V,A)
    G = get_migration_graph(V, A)
    non_zero = torch.where(G > 0)
    source_sites = non_zero[0]
    if len(torch.unique(source_sites)) == 0:
        return "no seeding"

    pattern = ""
    # 1) determine if monoclonal (no multi-eges) or polyclonal (multi-edges)
    if len(ordered_sites) == 2:
        pattern = "polyclonal " if ((G > 1).any()) else  "monoclonal "
    elif ((G > 1).any()):
        pattern = "polyclonal "
    else:
        shared_clusters = get_shared_clusters(V, A, ordered_sites, primary_site, full_node_idx_to_label)
        prim_to_met_clusters = shared_clusters[ordered_sites.index(primary_site)]
        all_seeding_clusters = set([cluster for seeding_clusters in prim_to_met_clusters for cluster in seeding_clusters])
        monoclonal = True
        for cluster_set in prim_to_met_clusters:
            # if clusters that seed the primary to each met are not identical,
            # then this is a polyclonal pattern
            if len(cluster_set) != 0 and (set(cluster_set) != all_seeding_clusters):
                monoclonal = False
                break
        pattern = "monoclonal " if monoclonal else "polyclonal "

    # 2) determine if monophyletic or polyphyletic
    if pattern == "monoclonal ":
        pattern += "monophyletic"
        return pattern
    
    seeding_clusters = set()
    for i,j in tree_iterator(Y):
        # if j is a subclonal presence leaf node, add i as the shared cluster 
        # (b/c i is the mutation cluster that j represents)
        if full_node_idx_to_label[j][1] == True:
            seeding_clusters.add(i)
        else:
            seeding_clusters.add(j)
        
    phylo = "monophyletic" if is_monophyletic(A,list(seeding_clusters)) else "polyphyletic"
    
    return pattern + phylo


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

def relabel_cluster(label, shorten, pad):
    if not shorten:
        return label

    out = ""
    # e.g. 1_M2 -> 1_M2
    if len(label) <=4 :
        out = label
    # e.g. 1;3;6;19_M2 -> 1_M2
    elif ";" in label and "_" in label:
        out = label[:label.find(";")] + label[label.find("_"):]
    # e.g. 100_M2 -> 100_M2
    elif "_" in label:
        out = label
    # e.g. 2;14;15 -> 2;14
    else:
        out = ";".join(label.split(";")[:2])
    if pad:
        return out.center(5)
    else:
        return out

def truncated_cluster_name(cluster_name):
    '''
    Displays a max of two mutation names associated with the cluster (e.g. 9;15;19;23;26 -> 9;15)
    Does nothing if the cluster name is not in above format
    '''
    assert(isinstance(cluster_name, str))
    split_name = cluster_name.split(";")
    truncated_name = ";".join(split_name) if len(split_name) <= 2 else ";".join(split_name[:2])
    return truncated_name

def get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites, shorten_label=True, pad=False):
    '''
    custom_node_idx_to_label only gives the internal node labels, so build a map of
    node_idx to (label, is_leaf) 
    e.g. ("0;9", False), ("0;9_P", True),  or ("5_liver", True) 
    '''
    full_node_idx_to_label_map = dict()
    for i, j in tree_iterator(T):
        if i in custom_node_idx_to_label:
            full_node_idx_to_label_map[i] = (relabel_cluster(custom_node_idx_to_label[i], shorten_label, pad), False)
        if j in custom_node_idx_to_label:
            full_node_idx_to_label_map[j] = (relabel_cluster(custom_node_idx_to_label[j], shorten_label, pad), False)
        elif j not in custom_node_idx_to_label:
            site_idx = (V[:,j] == 1).nonzero()[0][0].item()
            full_node_idx_to_label_map[j] = (relabel_cluster(f"{custom_node_idx_to_label[i]}_{ordered_sites[site_idx]}", shorten_label, pad), True)
    return full_node_idx_to_label_map

def idx_to_color(custom_colors, idx, alpha=1.0):
    if custom_colors != None:
        rgb = mcolors.to_rgb(custom_colors[idx])
        rgb_alpha = (rgb[0], rgb[1], rgb[2], alpha)
        return mcolors.to_hex(rgb_alpha, keep_alpha=True)

    # TODO repeat colors in this case
    assert(idx < len(COLORS))
    return COLORS[idx]

def get_migration_graph(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    '''
    migration_graph = (V @ A) @ V.T
    migration_graph_no_diag = torch.mul(migration_graph, 1-torch.eye(migration_graph.shape[0], migration_graph.shape[1]))
    
    return migration_graph_no_diag

def plot_migration_graph(V, A, ordered_sites, custom_colors, primary, show=True):
    '''
    Plots migration graph G which represents the migrations/comigrations between
    all anatomical sites.

    Returns a list of edges (e.g. [('P' ,'M1'), ('P', 'M2')])
    '''
    colors = custom_colors
    if colors == None:
        colors = COLORS
    assert(len(ordered_sites) <= len(colors))

    mig_graph_no_diag = get_migration_graph(V, A)

    G = nx.MultiDiGraph()
    for node, color in zip(ordered_sites, colors):
        G.add_node(node, shape="box", color=color, fillcolor='white', fontname="Lato", penwidth=3.0)

    edges = []
    for i, adj_row in enumerate(mig_graph_no_diag):
        for j, num_edges in enumerate(adj_row):
            if num_edges > 0:
                for _ in range(int(num_edges.item())):
                    G.add_edge(ordered_sites[i], ordered_sites[j], color=f'"{colors[i]};0.5:{colors[j]}"', penwidth=3)
                    edges.append((ordered_sites[i], ordered_sites[j]))

    dot = nx.nx_pydot.to_pydot(G)
    if show:
        view_pydot(dot)

    dot_lines = dot.to_string().split("\n")
    dot_lines.insert(1, 'dpi=600;size=3.5;')
    dot_str = ("\n").join(dot_lines)

    return dot_str, edges

def print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors, ordered_sites, print_config):
    '''
    Returns an averaged tree over all (TODO: all or top k?) converged trees
    by weighing each tree edge or vertex label by the softmax of the 
    likelihood of that tree 
    '''
    _, min_loss_indices = torch.topk(losses_tensor, len(losses_tensor), largest=False, sorted=True)
    # averaged tree edges are weighted by the average of the softmax of the negative log likelihoods
    # TODO*: is this the right way to compute weights?
    def softmaxed_losses(losses_tensor):
        if not torch.is_tensor(losses_tensor):
            losses_tensor = torch.tensor(losses_tensor)
        return torch.softmax(-1.0*(torch.log2(losses_tensor)/ torch.log2(torch.tensor(1.1))), dim=0)

    weights = torch.softmax(-1.0*(torch.log2(losses_tensor)/ torch.log2(torch.tensor(1.1))), dim=0)
    #print("losses tensor\n", losses_tensor, weights)

    weighted_edges = dict() # { edge_0 : [loss_0, loss_1] }
    weighted_node_colors = dict() # { node_0 : { anatomical_site_0 : [loss_0, loss_3]}}
    for sln_idx in min_loss_indices:
        loss = losses_tensor[sln_idx]
        weight = weights[sln_idx]

        full_tree_node_idx_to_label = get_full_tree_node_idx_to_label(V[sln_idx], full_trees[sln_idx], node_idx_to_label, ordered_sites)

        for i, j in tree_iterator(full_trees[sln_idx]):
            edge = full_tree_node_idx_to_label[i][0], full_tree_node_idx_to_label[j][0]
            if edge not in weighted_edges:
                weighted_edges[edge] = []
            weighted_edges[edge].append(weight.item())

        for node_idx in full_tree_node_idx_to_label:
            site_idx = (V[sln_idx][:,node_idx] == 1).nonzero()[0][0].item()
            node_label, _ = full_tree_node_idx_to_label[node_idx]
            if node_label not in weighted_node_colors:
                weighted_node_colors[node_label] = dict()
            if site_idx not in weighted_node_colors[node_label]:
                weighted_node_colors[node_label][site_idx] = []
            weighted_node_colors[node_label][site_idx].append(loss.item())
    
    avg_node_colors = dict()
    for node_label in weighted_node_colors:
        avg_node_colors[node_label] = dict()
        avg_losses = []
        ordered_labels = weighted_node_colors[node_label]
        for site_idx in ordered_labels:
            vals = weighted_node_colors[node_label][site_idx]
            avg_losses.append(sum(vals)/len(vals))

        #softmaxed = np.exp(softmaxed)/sum(np.exp(softmaxed))
        softmaxed = softmaxed_losses(avg_losses)
        for site_idx, soft in zip(ordered_labels, softmaxed):
            avg_node_colors[node_label][site_idx] = soft
    #print("avg_node_colors\n", avg_node_colors)

    avg_edges = dict()
    for edge in weighted_edges:
        avg_edges[edge] = sum(weighted_edges[edge])/len(weighted_edges[edge])

    #print("avg_edges\n", avg_edges)

    plot_averaged_tree(avg_edges, avg_node_colors, ordered_sites, custom_colors, node_idx_to_label, show=print_config.visualize)

# TODO: make custom_node_idx_to_label a required argument

def plot_averaged_tree(avg_edges, avg_node_colors, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):

    penwidth = 2.0
    alpha = 1.0

    max_edge_weight = max(list(avg_edges.values()))

    def rescaled_edge_weight(edge_weight):
        return (penwidth/max_edge_weight)*edge_weight
    
    
    G = nx.DiGraph()

    for label_i, label_j in avg_edges.keys():
        
        node_i_color = ""
        for site_idx in avg_node_colors[label_i]:
            node_i_color += f'"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_i][site_idx]}:"'
        node_j_color = ""
        for site_idx in avg_node_colors[label_j]:
            node_j_color += f'"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_j][site_idx]}:"'
        is_leaf = False

        G.add_node(label_i, xlabel=label_i, label="", shape="circle", fillcolor=node_i_color, 
                    color="none", penwidth=3, style="wedged",
                    fixedsize="true", height=0.35, fontname="Lato", 
                    fontsize="10pt")
        G.add_node(label_j, xlabel="" if is_leaf else label_j, label="", shape="circle", 
                    fillcolor=node_j_color, color="none", 
                    penwidth=3, style="solid" if is_leaf else "wedged",
                    fixedsize="true", height=0.35, fontname="Lato", 
                    fontsize="10pt")

        # G.add_node(label_i, shape="circle", style="wedged", fillcolor=node_i_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        # G.add_node(label_j, shape="circle", style="wedged", fillcolor=node_j_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        #print(label_i, label_j, avg_edges[(label_i, label_j)], rescaled_edge_weight(avg_edges[(label_i, label_j)]))
        # G.add_edge(label_i, label_j, color="#black", penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0.75, spline="ortho")
        style = "dashed" if is_leaf else "solid"
        penwidth = 2 if is_leaf else 2.5
        xlabel = "" if is_leaf else label_j
        G.add_edge(label_i, label_j,
                    color=f'"grey"', 
                    penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0, fontname="Lato", 
                    fontsize="10pt", style=style)

    #assert(nx.is_tree(G))
    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string()
    # hack since there doesn't seem to be API to modify graph attributes...
    dot_lines = dot.split("\n")
    dot_lines.insert(1, 'graph[splines=false]; nodesep=0.7; ranksep=0.6; forcelabels=true;')
    dot = ("\n").join(dot_lines)
    src = Source(dot) # dot is string containing DOT notation of graph
    if show:
        display(src)

def generate_legend_dot(ordered_sites, custom_colors, node_options):
    legend = nx.DiGraph()
    # this whole reversed business is to get the primary at the top of the legend...
    for i, site in enumerate(reversed(ordered_sites)):
        color = idx_to_color(custom_colors, len(ordered_sites)-1-i)
        legend.add_node(i, shape="plaintext", style="solid", label=f"{site}\r", 
                        width=0.3, height=0.2, fixedsize="true",
                        fontname="Lato", fontsize="10pt")
        legend.add_node(f"{i}_circle", fillcolor=color, color=color, 
                        style="filled", height=0.2, **node_options)

    legend_dot = to_pydot(legend).to_string()
    legend_dot = legend_dot.replace("strict digraph", "subgraph cluster_legend")
    legend_dot = legend_dot.split("\n")
    legend_dot.insert(1, 'rankdir="LR";{rank=source;'+" ".join(str(i) for i in range(len(ordered_sites))) +"}")
    legend_dot = ("\n").join(legend_dot)
    return legend_dot


def plot_tree(V, T, gen_dist, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):

    # (1) Create full directed graph 
    
    # these labels are used for display in plotting
    display_node_idx_to_label_map = get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites,
                                                                    shorten_label=True, pad=False)
    # these labels are used for writing out full vertex names to file
    full_node_idx_to_label_map = get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites,
                                                                 shorten_label=False, pad=False)

    color_map = { i:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":"Lato", "fontsize":"12pt",
                    "fixedsize":"true", "height":0.25}

    # TODO: come up w better scaling mechanism for genetic distance
    if gen_dist != None:
        gen_dist = ((gen_dist / torch.max(gen_dist[gen_dist>0]))*1.5) + 1
        #gen_dist = torch.clamp(gen_dist, max=2) # for visualization purposes

    edges = []
    for i, j in tree_iterator(T):
        label_i, _ = display_node_idx_to_label_map[i]
        label_j, is_leaf = display_node_idx_to_label_map[j]
        
        G.add_node(i, xlabel=label_i, fillcolor=color_map[i], 
                    color=color_map[i], style="filled", **node_options)
        G.add_node(j, xlabel="" if is_leaf else label_j, fillcolor=color_map[j], 
                    color=color_map[j], style="solid" if is_leaf else "filled", **node_options)

        style = "dashed" if is_leaf else "solid"
        penwidth = 5 if is_leaf else 5.5
        xlabel = "" if is_leaf else label_j

        # We're iterating through i,j of the full tree (including leaf nodes),
        # while G only has genetic distances between internal nodes
        minlen = gen_dist[i, j].item() if (gen_dist != None and i < len(gen_dist) and j < len(gen_dist)) else 1.0
        # TODO: make branch lengths a print config option
        #minlen = 1.0
        #print(i, j, minlen)
        G.add_edge(i, j,color=f'"{color_map[i]};0.5:{color_map[j]}"', 
                   penwidth=penwidth, arrowsize=0, style=style, minlen=minlen)

        edges.append((full_node_idx_to_label_map[i][0], full_node_idx_to_label_map[j][0]))

    # Add edge from normal to root 
    root_idx = get_root_index(T)
    root_label = display_node_idx_to_label_map[root_idx][0]
    G.add_node("normal", label="", xlabel=root_label, penwidth=3, style="invis")
    G.add_edge("normal", root_idx, label="", 
                color=f'"{color_map[root_idx]}"', 
                penwidth=4, arrowsize=0, style="solid")

    assert(nx.is_tree(G))

    # (2) Create legend
    #legend_dot = generate_legend_dot(ordered_sites, custom_colors, node_options)

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    dot.insert(1, 'graph[splines=false]; nodesep=0.7; rankdir=TB; ranksep=0.6; forcelabels=true; dpi=600; size=2.5;')
    #dot.insert(2, legend_dot)
    dot_str = ("\n").join(dot)

    if show:
        dot = nx.nx_pydot.to_pydot(dot_str)
        view_pydot(dot)


    vertex_name_to_site_map = { full_node_idx_to_label_map[i][0]:ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return dot_str, edges, vertex_name_to_site_map

def get_loss_dict(V, soft_V, T, G, O, p, full_loss):
    V = V.reshape(1, V.shape[0], V.shape[1]) # add batch dimension
    soft_V = soft_V.reshape(1, soft_V.shape[0], soft_V.shape[1]) # add batch dimension
    m, c, s, g, o = vert_label.get_ancestral_labeling_metrics(V, T, G, O, p)
    e = vert_label.get_entropy(V, soft_V)
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

def print_best_trees(min_loss_solutions, U, ref_matrix, var_matrix, O, weights, ordered_sites,
                     print_config, custom_colors, primary, output_dir, run_name):
    '''
    min_loss_solutions is in order from lowest to highest loss 
    '''
    
    primary_idx = ordered_sites.index(primary)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites)).T

    ret = None
    figure_outputs = []
    pickle_outputs = {OUT_LABElING_KEY:[], OUT_LOSSES_KEY:[],OUT_IDX_LABEL_KEY:[],
                      OUT_ADJ_KEY:[], OUT_SITES_KEY:ordered_sites, OUT_LOSS_DICT_KEY:[],
                      OUT_PRIMARY_KEY:primary, OUT_SUB_PRES_KEY:U.numpy(), OUT_WEIGHTS_KEY:[],
                      OUT_SOFTV_KEY:[], OUT_GEN_DIST_KEY:[]}

    with torch.no_grad():
        for i, min_loss_solution in enumerate(min_loss_solutions):
            V = min_loss_solution.V
            soft_V = min_loss_solution.soft_V
            T = min_loss_solution.T
            G = min_loss_solution.G
            full_loss = min_loss_solution.loss
            node_idx_to_label = min_loss_solution.node_idx_to_label
            loss_dict = get_loss_dict(V, soft_V, T, G, O, p, full_loss)

            tree_dot, edges, vertices_to_sites_map = plot_tree(V, T, G, ordered_sites, custom_colors, node_idx_to_label, show=False)
            mig_graph_dot, mig_graph_edges = plot_migration_graph(V, T, ordered_sites, custom_colors, primary, show=False)

            seeding_pattern = get_seeding_pattern(V, T)
            figure_outputs.append((tree_dot, mig_graph_dot, loss_dict, seeding_pattern))
            pickle_outputs[OUT_LABElING_KEY].append(V.detach().numpy())
            pickle_outputs[OUT_LOSSES_KEY].append(full_loss.numpy())
            pickle_outputs[OUT_ADJ_KEY].append(T.detach().numpy())
            pickle_outputs[OUT_SOFTV_KEY].append(soft_V.numpy())
            if G != None:
                pickle_outputs[OUT_GEN_DIST_KEY].append(G.numpy())                
            pickle_outputs[OUT_LOSS_DICT_KEY].append(loss_dict)
            pickle_outputs[OUT_WEIGHTS_KEY].append((min_loss_solution.mig_weight, min_loss_solution.comig_weight, min_loss_solution.seed_weight))

            full_tree_idx_to_label = get_full_tree_node_idx_to_label(V, T, node_idx_to_label, ordered_sites,
                                                                     shorten_label=False, pad=False)
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
            if "Lato" in font:
                matplotlib.font_manager.fontManager.addfont(font)
                rcParams['font.family'] = 'Lato'

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
            ax1.set_title(f'Tree {i+1}\n{seeding_pattern}', fontsize=10, loc="left", va="top", x=-0.1, y=1.0)
            #ax1.text(0.1, 0.65, seeding_pattern, fontname=fontname)

            ax2.imshow(mig_graph)
            ax2.axis('off')

            ax3.text(0.5, 0.5, formatted_loss_string(loss_info, weights), ha='center', va='center', fontsize=8)
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




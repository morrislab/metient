import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import torch
from IPython.display import Image, display
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source
from PIL import Image as PILImage
import io
import matplotlib.gridspec as gridspec
import math
import matplotlib.font_manager
from matplotlib import rcParams
import os

import string

import pygraphviz as pgv

import hashlib

# TODO: this cyclical import is not great
import src.lib.vertex_labeling as vert_label
from src.util.vertex_labeling_util import LabeledTree
from src.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

import seaborn as sns

COLORS = ["#6aa84fff","#c27ba0ff", "#e69138ff", "#be5742e1", "#2496c8ff", "#674ea7ff"] + sns.color_palette("Paired").as_hex()


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class PrintConfig:
    def __init__(self, visualize=True, verbose=False, viz_intermeds=False, k_best_trees=1, save_imgs=False):
        '''
        visualize: bool, whether to visualize loss, best tree, and migration graph
        verbose: bool, whether to print debug info
        viz_intermeds: bool, whether to visualize intermediate solutions to best tree
        k_best_trees: int, number of best tree solutions to visualize (if 1, only show best tree)
        '''
        self.visualize = visualize
        self.verbose = verbose 
        self.viz_intermeds = viz_intermeds
        self.k_best_trees = k_best_trees
        self.save_imgs = save_imgs

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

def is_cyclic(G):
    '''
    returns True if graph contains cycles
    '''
    
    def _helper(v, visited, rec_stack):
 
        # Mark current node as visited and add to stack
        visited[v] = True
        rec_stack[v] = True
 
        # Recur for all neighbours if any neighbour is visited and in
        # stack then graph is cyclic
        neighbors = [i for i, value in enumerate(G[v]) if value > 0]
        for neighbour in neighbors:
            if visited[neighbour] == False:
                if _helper(neighbour, visited, rec_stack) == True:
                    return True
            elif rec_stack[neighbour] == True:
                return True
 
        # The node needs to be popped from recursion stack before 
        # function ends
        rec_stack[v] = False
        return False

    G = G.tolist() # this just makes things easier...
    n = len(G)
    visited = [False] * (n + 1)
    rec_stack = [False] * (n + 1)
    for node in range(n):
        if visited[node] == False:
            if _helper(node, visited, rec_stack) == True:
                    return True
        return False


def get_seeding_pattern_from_migration_graph(G):
    '''
    G: directed adjacency matrix containing the number of 
    migrations between sites (num_sites x num_sites)

    returns: verbal description of the seeding pattern
    '''

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

    unique_source_sites = torch.unique(source_sites)

    if len(unique_source_sites) == 1:
        pattern += "single-source seeding"
    elif is_cyclic(G):
        pattern += "reseeding"
    else:
        pattern += "multi-source seeding"
    return pattern


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
    mig_losses = [weights.mig*e["mig"] for e in loss_dicts]
    comig_losses = [weights.comig*e["comig"] for e in loss_dicts]
    seed_losses = [weights.seed_site*e["seeding"] for e in loss_dicts]
    data_fit_losses = [weights.data_fit*e["nll"] for e in loss_dicts]
    reg_losses = [weights.reg*e["reg"] for e in loss_dicts]
    total_losses = [e["loss"] for e in loss_dicts]

    plt.plot([x for x in range(len(loss_dicts))],mig_losses, label="m")
    plt.plot([x for x in range(len(loss_dicts))],comig_losses, label="c")
    plt.plot([x for x in range(len(loss_dicts))],seed_losses, label="s")
    plt.plot([x for x in range(len(loss_dicts))],data_fit_losses, label="nll")
    plt.plot([x for x in range(len(loss_dicts))],reg_losses, label="reg")
    plt.plot([x for x in range(len(loss_dicts))],total_losses, label="total_loss")

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

def tree_iterator(T):
    '''
    iterate an adjacency matrix, returning i and j for all values = 1
    '''
    for i, adj_row in enumerate(T):
        for j, val in enumerate(adj_row):
            if val == 1:
                yield i, j

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

    # TODO pass in printconfig save option 
    dot_lines = dot.to_string().split("\n")
    dot_lines.insert(1, 'dpi=600;size=3.5;')
    dot = ("\n").join(dot_lines)
    
    #dot = pydot.graph_from_dot_data(dot)[0]
    dot = Source(dot)
    #dot.write_png('mig_graph.png')

    return dot, edges

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
            node_i_color += f"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_i][site_idx]}:"
        node_j_color = ""
        for site_idx in avg_node_colors[label_j]:
            node_j_color += f"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_j][site_idx]}:"
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

    assert(nx.is_tree(G))
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


    color_map = { display_node_idx_to_label_map[i][0]:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":"Lato", "fontsize":"11pt",
                    "fixedsize":"true", "height":0.25}

    # TODO: come up w better scaling mechanism for genetic distance
    if gen_dist != None:
        gen_dist = gen_dist / torch.min(gen_dist[gen_dist>0])
        gen_dist = torch.clamp(gen_dist, max=2) # for visualization purposes
    edges = []
    for i, j in tree_iterator(T):
        label_i, _ = display_node_idx_to_label_map[i]
        label_j, is_leaf = display_node_idx_to_label_map[j]
        
        G.add_node(label_i, xlabel=label_i, fillcolor=color_map[label_i], 
                    color=color_map[label_i], style="filled", **node_options)
        G.add_node(label_j, xlabel="" if is_leaf else label_j, fillcolor=color_map[label_j], 
                    color=color_map[label_j], style="solid" if is_leaf else "filled", **node_options)

        style = "dashed" if is_leaf else "solid"
        penwidth = 4 if is_leaf else 4.5
        xlabel = "" if is_leaf else label_j

        # We're iterating through i,j of the full tree (including leaf nodes),
        # while G only has genetic distances between internal nodes
        minlen = gen_dist[i, j].item() if (gen_dist != None and i < len(gen_dist) and j < len(gen_dist)) else 1.0
        #print(i, j, minlen)
        G.add_edge(label_i, label_j,
                    color=f'"{color_map[label_i]};0.5:{color_map[label_j]}"', 
                    penwidth=penwidth, arrowsize=0, style=style, minlen=minlen)

        edges.append((full_node_idx_to_label_map[i][0], full_node_idx_to_label_map[j][0]))

    # Add edge from normal to root 
    root_idx = get_root_index(T)
    root_label = display_node_idx_to_label_map[root_idx][0]
    G.add_node("normal", label="", xlabel=root_label, penwidth=3, style="invis")
    G.add_edge("normal", root_label, label="", 
                color=f'"{color_map[root_label]}"', 
                penwidth=4, arrowsize=0, style="solid")

    assert(nx.is_tree(G))

    # (2) Create legend
    #legend_dot = generate_legend_dot(ordered_sites, custom_colors, node_options)

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    dot.insert(1, 'graph[splines=false]; nodesep=0.7; ranksep=0.6; forcelabels=true; dpi=600; size=2.5;')
    #dot.insert(2, legend_dot)
    dot = ("\n").join(dot)
    #dot = pydot.graph_from_dot_data(dot)[0]
    
    if show:
        dot = pydot.graph_from_dot_data(dot)[0]
        view_pydot(dot)

    dot = Source(dot)

    #dot.write_png('labeled_tree.png')

    vertex_name_to_site_map = { full_node_idx_to_label_map[i][0]:ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return dot, edges, vertex_name_to_site_map

# TODO: make this a diff option to display
def plot_tree_deprecated(V, T, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):

    full_node_idx_to_label_map = get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites)

    patches = []
    for i, site in enumerate(ordered_sites):
        patch = mpatches.Patch(color=idx_to_color(custom_colors, i), label=site)
        patches.append(patch)

    color_map = { full_node_idx_to_label_map[i]:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    edges = []
    for i, j in tree_iterator(T):
        label_i = full_node_idx_to_label_map[i]
        label_j = full_node_idx_to_label_map[j]
        edges.append((label_i, label_j))
        G.add_node(label_i, shape="circle", color=color_map[label_i], penwidth=3, fixedsize="true", fontname = "Lato", fontsize="10pt")
        G.add_node(label_j, shape="circle", color=color_map[label_j], penwidth=3, fixedsize="true", fontname = "Lato", fontsize="10pt")
        G.add_edge(label_i, label_j, color=f'"{color_map[label_i]};0.5:{color_map[label_j]}"', penwidth=3, arrowsize=0.75)

    #assert(nx.is_tree(G))

    nodes = [full_node_idx_to_label_map[i] for i in range(len(T))]

    dot = to_pydot(G).to_string()
    src = Source(dot) # dot is string containing DOT notation of graph
    if show:
        display(src)

    vertex_name_to_site_map = { full_node_idx_to_label_map[i]:ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return edges, vertex_name_to_site_map


def print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, 
                    node_idx_to_label, ordered_sites, max_iter, show):
    
    # Debugging information
    U_clipped = labeled_tree.U.cpu().detach().numpy()
    U_clipped[np.where(U_clipped<U_CUTOFF)] = 0
    logger.debug(f"\nU > {U_CUTOFF}\n")
    col_labels = ["norm"] + [truncated_cluster_name(node_idx_to_label[k]) if k in node_idx_to_label else "0" for k in range(U_clipped.shape[1] - 1)]
    df = pd.DataFrame(U_clipped, columns=col_labels, index=ordered_sites)
    logger.debug(df)
    logger.debug("\nF_hat")
    F_hat_df = pd.DataFrame((labeled_tree.U @ B).cpu().detach().numpy(), index=ordered_sites)
    logger.debug(F_hat_df)

    # Loss information
    loss, loss_dict = vert_label.objective(labeled_tree.labeling, labeled_tree.tree, ref_matrix, var_matrix, 
                                           labeled_tree.U, B, labeled_tree.branch_lengths, O, weights, -1, 
                                           max_iter, "constant")
    if show:
        print(formatted_loss_string(loss_dict))

    return loss_dict

def print_best_trees(losses_tensor, V, U, full_trees, full_branch_lengths, ref_matrix, var_matrix, B, O, G,
                     weights, node_idx_to_label, ordered_sites, print_config, intermediate_data, custom_colors, 
                     primary, max_iter, output_dir, run_name):

    def _visualize_intermediate_trees(best_tree_idx):
        '''
        Visualizes the best tree at intermediate iterations. This shows how the vertex labeling
        gets changed over iterations as the loss converges.
        '''
        for itr, data in enumerate(intermediate_data):
            losses_tensor, full_trees, V, U, full_branch_lengths, soft_X = intermediate_data[itr][0], intermediate_data[itr][1], intermediate_data[itr][2], intermediate_data[itr][3], intermediate_data[itr][4], intermediate_data[itr][5]
            print("="*30 + " INTERMEDIATE TREE " + "="*30+"\n")
            print(f"Iteration: {itr*20}, Intermediate best tree idx {best_tree_idx}")
            # skip root index (which is root, and we know the vert. label)
            cols = [node_idx_to_label[i] for i in range(0,len(node_idx_to_label)) if i != get_root_index(full_trees[best_tree_idx])]
            softx_df = pd.DataFrame(soft_X[best_tree_idx].cpu().detach().numpy(), columns=cols, index=ordered_sites)
            logger.info("softmax_X\n")
            logger.info(softx_df)

            tree = LabeledTree(full_trees[best_tree_idx], V[best_tree_idx], U[best_tree_idx], full_branch_lengths[best_tree_idx])

            print_tree_info(tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, max_iter, print_config)
            plot_tree(tree.labeling, tree.tree, G, ordered_sites, custom_colors, node_idx_to_label, show=print_config.visualize)
            plot_migration_graph(tree.labeling, tree.tree, ordered_sites, custom_colors, primary)

    # Get the top k unique trees
    _, min_loss_indices = torch.topk(losses_tensor, len(losses_tensor), largest=False, sorted=True)
    k_trees_and_losses = []
    tree_set = set()
    # Iterate from best to worst tree, and only get trees with unique U or V matrices
    for i in min_loss_indices:
        labeled_tree = LabeledTree(full_trees[i], V[i], U[i], full_branch_lengths[i])
        if labeled_tree not in tree_set:
            tree_set.add(labeled_tree)
            if len(k_trees_and_losses) < print_config.k_best_trees:
                k_trees_and_losses.append((labeled_tree, losses_tensor[i]))

        if i == 0 and print_config.viz_intermeds:
            best_tree_idx = min_loss_indices[0]
            _visualize_intermediate_trees(best_tree_idx)

    ret = None
    figure_outputs = []
    npz_outputs = {'ancestral_labelings':[], 'subclonal_presence_matrices':[], 
                   'full_adjacency_matrices':[], 'ordered_anatomical_sites':ordered_sites,
                   'node_idx_to_label':node_idx_to_label}
    for i, tup in enumerate(k_trees_and_losses):
        tree = tup[0]

        loss_info = print_tree_info(tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, max_iter, show=False)
        tree_dot, edges, vertices_to_sites_map = plot_tree(tree.labeling, tree.tree, G, ordered_sites, custom_colors, node_idx_to_label, show=False)
        mig_graph_dot, mig_graph_edges = plot_migration_graph(tree.labeling, tree.tree, ordered_sites, custom_colors, primary, show=False)

        seeding_pattern = get_seeding_pattern_from_migration_graph(get_migration_graph(tree.labeling, tree.tree))
        figure_outputs.append((tree_dot, mig_graph_dot, loss_info, seeding_pattern))
        npz_outputs['ancestral_labelings'].append(tree.labeling)
        npz_outputs['subclonal_presence_matrices'].append(tree.U)
        npz_outputs['full_adjacency_matrices'].append(tree.tree)

        if i == 0: # Best tree
            ret = (edges, vertices_to_sites_map, mig_graph_edges, loss_info)

    save_outputs(figure_outputs, print_config, output_dir, run_name, npz_outputs)

    return ret

def formatted_loss_string(loss_dict):
    s = f"Loss: {loss_dict['loss']}\n\n"

    s += f"Migration num.: {int(loss_dict['mig'])}\n"
    s += f"Comigration num.: {int(loss_dict['comig'])}\n"
    s += f"Seeding site num.: {int(loss_dict['seeding'])}\n"

    s += f"Data fit nll: {loss_dict['nll']}\n"

    if (loss_dict['gen'] != 0):
        s += f"Genetic distance loss: {loss_dict['gen']}\n"
    if (loss_dict['gen'] != 0):
        s += f"Organotropism loss: {loss_dict['gen']}\n"
    return s


def save_outputs(figure_outputs, print_config, output_dir, run_name, npz_outputs):

    k = print_config.k_best_trees
    sys_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in sys_fonts:
        if "Lato" in font:
            matplotlib.font_manager.fontManager.addfont(font)
            rcParams['font.family'] = 'Lato'

    n = len(figure_outputs)

    if k < n:
        print("{k} unique best trees were not found ({n} were found). Retry with higher batch size if you want to try and get more trees.")

    # Create a figure and subplots
    #fig, axs = plt.subplots(3, k*2, figsize=(10, 8))

    plt.suptitle(run_name)

    z = 2 # number of trees displayed per row

    nrows = math.ceil(k/z)
    h = nrows*5
    fig = plt.figure(figsize=(10,h))
    
    vspace = 1/nrows

    for i, (tree_dot, mig_graph_dot, loss_info, seeding_pattern) in enumerate(figure_outputs):
        # Create graphviz objects from the DOT code
        tree = PILImage.open(tree_dot.render(os.path.join(output_dir,f"T_{run_name}"),format="png", view=False))
        mig_graph = PILImage.open(mig_graph_dot.render(os.path.join(output_dir, f"G_{run_name}"),format="png", view=False))

        gs1 = gridspec.GridSpec(3, 3)

        row = math.floor(i/2)
        pad=0.02

        # left = 0.0 if i is odd, 0.55 if even
        # right = 0.45 if i is odd, 1.0 if even
        gs1.update(left=0.0+((i%2)*0.53), right=0.47+0.55*(i%2), top=1-(row*vspace)-pad, bottom=1-((row+1)*vspace)+pad, wspace=0.05)
        ax1 = plt.subplot(gs1[:-1, :])
        ax2 = plt.subplot(gs1[-1, :-1])
        ax3 = plt.subplot(gs1[-1, -1])

        # Render and display each graph
        ax1.imshow(tree)
        ax1.axis('off')
        ax1.set_title(f'Tree {i+1}\n{seeding_pattern}', fontsize=12, ha="left", va="top", x=0, y=1.0)
        #ax1.text(0.1, 0.65, seeding_pattern, fontname=fontname)

        ax2.imshow(mig_graph)
        ax2.axis('off')

        ax3.text(0.5, 0.5, formatted_loss_string(loss_info), ha='center', va='center', fontsize=10)
        ax3.axis('off')

    # Display and save the plot
    plt.tight_layout()
    fig1 = plt.gcf()
    fig1.savefig(os.path.join(output_dir, f'{run_name}.png'), dpi=300)
    print(f"Saving {run_name} to {output_dir}")

    if print_config.visualize:
        plt.show()

    # Cleanup temp files
    os.remove(os.path.join(output_dir, f"T_{run_name}"))
    os.remove(os.path.join(output_dir, f"T_{run_name}.png"))
    os.remove(os.path.join(output_dir, f"G_{run_name}"))
    os.remove(os.path.join(output_dir, f"G_{run_name}.png"))

    # Save results to npz file
    np.savez(os.path.join(output_dir, f"{run_name}.results.npz"), 
             ancestral_labelings=npz_outputs['ancestral_labelings'],
             subclonal_presence_matrices=npz_outputs['subclonal_presence_matrices'],
             full_adjacency_matrices=npz_outputs['full_adjacency_matrices'],
             ordered_anatomical_sites=npz_outputs['ordered_anatomical_sites'],
             node_idx_to_label=npz_outputs['node_idx_to_label'])




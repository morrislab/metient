import numpy as np
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import random
import torch
from torch.autograd import Variable
import sys
import pydot
from IPython.display import Image, display
from networkx.drawing.nx_pydot import to_pydot
from graphviz import Source

import string

import pygraphviz as pgv

from functools import wraps
import hashlib

# TODO: this cyclical import is not great
import src.lib.vertex_labeling as vert_label
from src.util.globals import *

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class PrintConfig:
    def __init__(self, visualize=True, verbose=False, viz_intermeds=False, k_best_trees=1):
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

class LabeledTree:
    def __init__(self, tree, labeling, U, branch_lengths):
        self.tree = tree
        self.labeling = labeling
        self.U = U
        self.branch_lengths = branch_lengths

    # TODO: write unit tests for this
    def __eq__(self, other):
        return ( isinstance(other, LabeledTree) and
               str(np.where(self.tree == 1)[0]) == str(np.where(other.tree == 1)[0]) and
               str(np.where(self.tree == 1)[1]) == str(np.where(other.tree == 1)[1]) and
               str(np.where(self.labeling == 1)[0]) == str(np.where(other.labeling == 1)[0]) and
               str(np.where(self.labeling == 1)[1]) == str(np.where(other.labeling == 1)[0])
               )

    def __hash__(self):
        hsh = hash((str(np.where(self.tree == 1)[0]), str(np.where(self.tree == 1)[1]),
                    str(np.where(self.labeling == 1)[0]), str(np.where(self.labeling == 1)[1])))
        return hsh

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
        G.add_node(node, shape="box", color=color, fillcolor='white', fontname="Corbel", penwidth=3.0)

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

def relabel_cluster(label, shorten):
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
    return out.center(5)

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

def get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites, shorten_label=True):
    '''
    custom_node_idx_to_label only gives the internal node labels, so build a map of
    node_idx to (label, is_leaf) 
    e.g. ("0;9", False), ("0;9_P", True),  or ("5_liver", True) 
    '''
    full_node_idx_to_label_map = dict()
    for i, j in tree_iterator(T):
        if i in custom_node_idx_to_label:
            full_node_idx_to_label_map[i] = (relabel_cluster(custom_node_idx_to_label[i], shorten_label), False)
        if j in custom_node_idx_to_label:
            full_node_idx_to_label_map[j] = (relabel_cluster(custom_node_idx_to_label[j], shorten_label), False)
        elif j not in custom_node_idx_to_label:
            site_idx = (V[:,j] == 1).nonzero()[0][0].item()
            full_node_idx_to_label_map[j] = (relabel_cluster(f"{custom_node_idx_to_label[i]}_{ordered_sites[site_idx]}", shorten_label), True)
    return full_node_idx_to_label_map

def print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors, ordered_sites):
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

    plot_averaged_tree(avg_edges, avg_node_colors, ordered_sites, custom_colors, node_idx_to_label)

def idx_to_color(custom_colors, idx, alpha=1.0):
    if custom_colors != None:
        rgb = colors.to_rgb(custom_colors[idx])
        rgb_alpha = (rgb[0], rgb[1], rgb[2], alpha)
        return colors.to_hex(rgb_alpha, keep_alpha=True)

    pastel_colors = plt.get_cmap("Set3").colors
    assert(idx < len(pastel_colors))
    return pastel_colors[idx]

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
        print(label_i, node_i_color)
        print(label_j, node_j_color)

        G.add_node(label_i, xlabel=label_i, label="", shape="circle", fillcolor=node_i_color, 
                    color="none", penwidth=3, style="wedged",
                    fixedsize="true", height=0.35, fontname="verdana", 
                    fontsize="10pt")
        G.add_node(label_j, xlabel="" if is_leaf else label_j, label="", shape="circle", 
                    fillcolor=node_j_color, color="none", 
                    penwidth=3, style="solid" if is_leaf else "wedged",
                    fixedsize="true", height=0.35, fontname="verdana", 
                    fontsize="10pt")

        # G.add_node(label_i, shape="circle", style="wedged", fillcolor=node_i_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        # G.add_node(label_j, shape="circle", style="wedged", fillcolor=node_j_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        print(label_i, label_j, avg_edges[(label_i, label_j)], rescaled_edge_weight(avg_edges[(label_i, label_j)]))
        # G.add_edge(label_i, label_j, color="#black", penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0.75, spline="ortho")
        style = "dashed" if is_leaf else "solid"
        penwidth = 2 if is_leaf else 2.5
        xlabel = "" if is_leaf else label_j
        G.add_edge(label_i, label_j,
                    color=f'"grey"', 
                    penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0, fontname="verdana", 
                    fontsize="10pt", style=style)

    assert(nx.is_tree(G))
    dot = to_pydot(G).to_string()
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
                        fontname="verdana", fontsize="10pt")
        legend.add_node(f"{i}_circle", fillcolor=color, color=color, 
                        style="filled", height=0.2, **node_options)

    legend_dot = to_pydot(legend).to_string()
    legend_dot = legend_dot.replace("strict digraph", "subgraph cluster_legend")
    legend_dot = legend_dot.split("\n")
    legend_dot.insert(1, 'rankdir="LR";{rank=source;'+" ".join(str(i) for i in range(len(ordered_sites))) +"}")
    legend_dot = ("\n").join(legend_dot)
    return legend_dot


def plot_tree(V, T, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):

    # (1) Create full directed graph 
    full_node_idx_to_label_map = get_full_tree_node_idx_to_label(V, T, custom_node_idx_to_label, ordered_sites)

    color_map = { full_node_idx_to_label_map[i][0]:idx_to_color(custom_colors, (V[:,i] == 1).nonzero()[0][0].item()) for i in range(V.shape[1])}
    G = nx.DiGraph()
    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":"verdana", "fontsize":"10pt",
                    "fixedsize":"true"}
    edges = []
    for i, j in tree_iterator(T):
        label_i, _ = full_node_idx_to_label_map[i]
        label_j, is_leaf = full_node_idx_to_label_map[j]
        edges.append((label_i, label_j))
        G.add_node(label_i, xlabel=label_i, fillcolor=color_map[label_i], 
                    color=color_map[label_i], style="filled", height=0.25, **node_options)
        G.add_node(label_j, xlabel="" if is_leaf else label_j, fillcolor=color_map[label_j], 
                    color=color_map[label_j], style="solid" if is_leaf else "filled", 
                    height=0.25, **node_options)

        style = "dashed" if is_leaf else "solid"
        penwidth = 4 if is_leaf else 4.5
        xlabel = "" if is_leaf else label_j
        G.add_edge(label_i, label_j,
                    color=f'"{color_map[label_i]};0.5:{color_map[label_j]}"', 
                    penwidth=penwidth, arrowsize=0, fontname="verdana", 
                    fontsize="10pt", style=style)

    # Add edge from normal to root 
    root_idx = get_root_index(T)
    root_label = full_node_idx_to_label_map[root_idx][0]
    G.add_node("normal", label="", xlabel=root_label, penwidth=3, style="invis")
    G.add_edge("normal", root_label, label="", 
                color=f'"{color_map[root_label]}"', 
                penwidth=4, arrowsize=0, fontname="verdana", 
                fontsize="10pt", style="solid")

    assert(nx.is_tree(G))

    # (2) Create legend
    #legend_dot = generate_legend_dot(ordered_sites, custom_colors, node_options)

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    dot.insert(1, 'graph[splines=false]; nodesep=0.7; ranksep=0.6; forcelabels=true;')
    #dot.insert(2, legend_dot)
    dot = ("\n").join(dot)

    if show:
        src = Source(dot) # dot is string containing DOT notation of graph
        display(src)

    vertex_name_to_site_map = { full_node_idx_to_label_map[i]:ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return edges, vertex_name_to_site_map

# TODO: remove
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
        G.add_node(label_i, shape="circle", color=color_map[label_i], penwidth=3, fixedsize="true", fontname = "verdana", fontsize="10pt")
        G.add_node(label_j, shape="circle", color=color_map[label_j], penwidth=3, fixedsize="true", fontname = "verdana", fontsize="10pt")
        G.add_edge(label_i, label_j, color=f'"{color_map[label_i]};0.5:{color_map[label_j]}"', penwidth=3, arrowsize=0.75)

    #assert(nx.is_tree(G))

    nodes = [full_node_idx_to_label_map[i] for i in range(len(T))]

    dot = to_pydot(G).to_string()
    src = Source(dot) # dot is string containing DOT notation of graph
    if show:
        display(src)

    vertex_name_to_site_map = { full_node_idx_to_label_map[i]:ordered_sites[(V[:,i] == 1).nonzero()[0][0].item()] for i in range(V.shape[1])}
    return edges, vertex_name_to_site_map

def get_root_index(T):
    '''
    returns the root idx (node with no inbound edges) from adjacency matrix T
    '''

    candidates = set([x for x in range(len(T))])
    for i, j in tree_iterator(T):
        candidates.remove(j)
    msg = "More than one" if len(candidates) > 1 else "No"
    assert(len(candidates) == 1, f"{msg} root node detected")

    return list(candidates)[0]

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

def print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, 
                    node_idx_to_label, ordered_sites, max_iter, print_config):
    loss, loss_components = vert_label.objective(labeled_tree.labeling, labeled_tree.tree, ref_matrix, var_matrix, labeled_tree.U, B, labeled_tree.branch_lengths, O, weights, -1, max_iter, print_config)
    U_clipped = labeled_tree.U.cpu().detach().numpy()
    U_clipped[np.where(U_clipped<U_CUTOFF)] = 0
    logger.debug(f"\nU > {U_CUTOFF}\n")
    col_labels = ["norm"] + [truncated_cluster_name(node_idx_to_label[k]) if k in node_idx_to_label else "0" for k in range(U_clipped.shape[1] - 1)]
    df = pd.DataFrame(U_clipped, columns=col_labels, index=ordered_sites)
    logger.debug(df)
    logger.debug("\nF_hat")
    F_hat_df = pd.DataFrame((labeled_tree.U @ B).cpu().detach().numpy(), index=ordered_sites)
    logger.debug(F_hat_df)
    return loss_components

def print_best_trees(losses_tensor, V, U, full_trees, full_branch_lengths, ref_matrix, var_matrix, B, O, weights,                 node_idx_to_label, ordered_sites, print_config, intermediate_data, custom_colors, 
                     primary, max_iter):

    _, min_loss_indices = torch.topk(losses_tensor, print_config.k_best_trees, largest=False, sorted=True)
    print("print_config.k_best_trees", print_config.k_best_trees)
    min_loss_labeled_trees_and_losses = []
    for i in min_loss_indices:
        labeled_tree = LabeledTree(full_trees[i], V[i], U[i], full_branch_lengths[i])
        min_loss_labeled_trees_and_losses.append((labeled_tree, losses_tensor[i]))

    for i, tup in enumerate(min_loss_labeled_trees_and_losses):
        labeled_tree = tup[0]
        if i == 0:
            
            if print_config.viz_intermeds:

                best_tree_idx = min_loss_indices[0]
                for itr, data in enumerate(intermediate_data):
                    losses_tensor, full_trees, V, U, full_branch_lengths, soft_X = intermediate_data[itr][0], intermediate_data[itr][1], intermediate_data[itr][2], intermediate_data[itr][3], intermediate_data[itr][4], intermediate_data[itr][5]
                    print("="*30 + " INTERMEDIATE TREE " + "="*30+"\n")
                    print(f"Iteration: {itr*20}, Intermediate best tree idx {best_tree_idx}")
                    softx_df = pd.DataFrame(soft_X[best_tree_idx].cpu().detach().numpy().T, columns=ordered_sites, index=[node_idx_to_label[i] for i in range(1,len(node_idx_to_label))]) # skip first index (which is root, and we know the vert. label)
                    print("soft_X\n", softx_df)

                    labeled_tree = LabeledTree(full_trees[best_tree_idx], V[best_tree_idx], U[best_tree_idx], full_branch_lengths[best_tree_idx])
                    print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, print_config, max_iter)
                    plot_tree(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, node_idx_to_label)
                    plot_migration_graph(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, primary)


            if print_config.visualize: 
                print("*"*30 + " BEST TREE " + "*"*30+"\n")

            best_tree = labeled_tree
            best_tree_loss_info = print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, max_iter, print_config)
            best_tree_edges, best_tree_vertex_name_to_site_map = plot_tree(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, node_idx_to_label, show=print_config.visualize)
            best_mig_graph_edges = plot_migration_graph(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, primary, show=print_config.visualize)

            #print("-"*100 + "\n")

        elif print_config.k_best_trees > 1:
            print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, print_config)
            plot_tree(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, node_idx_to_label)
            plot_migration_graph(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, primary)
            print("-"*100 + "\n")

    return best_tree_edges, best_tree_vertex_name_to_site_map, best_mig_graph_edges, best_tree_loss_info


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

# Taken from pairtree
def convert_parents_to_adjmatrix(parents):
    K = len(parents) + 1
    adjm = np.eye(K)
    adjm[parents,np.arange(1, K)] = 1
    return adjm

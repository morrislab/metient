import torch
import pickle
import os
import gzip
import numpy as np
import seaborn as sns
from collections import Counter
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

import matplotlib.pyplot as plt

import io
import torch

from matplotlib import rcParams
import math
import matplotlib
import matplotlib.gridspec as gridspec

import metient as met 
from metient.util.globals import *
from metient.util import plotting_util as putil


TISSUE_ORDER = ['LL',"RE","RW","M1","M2","Liv"]
TISSUE_COLORS = ["#6aa84f","#c27ba0","#bf4040", "#6fa8dc", "#e69138", "#9e9e9e"]
TISSUE_LABEL_TO_IDX = {label: idx for idx, label in enumerate(TISSUE_ORDER)}
FONT = "Arial"


def extract_info(metient_result_dir, clone):
    """
    Extract and process solution information from a Metient result directory for a specific clone.
    
    Args:
        metient_result_dir (str): Path to the directory containing Metient results
        clone (str): Clone identifier
        
    Returns:
        tuple containing:
            - unnormed_tissue_trans (torch.Tensor): Unnormalized tissue transition matrix
            - weighted_normed_tissue_trans (torch.Tensor): Normalized and weighted tissue transition matrix
            - pars_metrics (list): List of tuples containing parsimony metrics (m,c,s) for each solution
            - mig_graphs (list): List of migration graphs as torch tensors
    """
    
    with gzip.open(os.path.join(metient_result_dir,f"{clone}_LL.pkl.gz") ,"rb") as f:
        pckl = pickle.load(f)
        
    num_solutions = len(pckl['clone_tree_labeling_matrices'])
    print(f"{num_solutions} solutions")
    ordered_sites = pckl['ordered_anatomical_sites']

    mig_graphs = []
    pars_metrics = []
    unnormed_tissue_trans = torch.zeros((len(TISSUE_ORDER),len(TISSUE_ORDER)))

    for tree_idx in range(num_solutions):
        V = pckl['clone_tree_labeling_matrices'][tree_idx]
        A = met.adjacency_matrix_from_parents(pckl['full_adjacency_matrices'][tree_idx])
        G = met.migration_graph(V, A).cpu().numpy()
        
        m, c, s = extract_pars_metrics(pckl, tree_idx)
        pars_metrics.append((m,c,s))

        mig_graph = torch.zeros((len(TISSUE_ORDER),len(TISSUE_ORDER)))
        for i, row_label in enumerate(ordered_sites):
            for j, col_label in enumerate(ordered_sites):
                # Get the indices in the universal matrix
                row_idx = TISSUE_LABEL_TO_IDX[row_label]
                col_idx = TISSUE_LABEL_TO_IDX[col_label]
                # Set the value in the universal matrix
                unnormed_tissue_trans[row_idx, col_idx] += int(G[i, j])
                mig_graph[row_idx, col_idx] = int(G[i, j])

        mig_graphs.append(mig_graph.cpu())

    unnormed_tissue_trans.fill_diagonal_(0)
    
    # Scaled, weighted average
    weighted_scaled_tissue_trans = weighted_scaled_migration_graph(
        putil.losses_to_probabilities(pckl['losses']), 
        mig_graphs
    )

    return unnormed_tissue_trans, weighted_scaled_tissue_trans, pars_metrics, mig_graphs

def scale_matrix(matrix, eps=1e-10):
    """Scale matrix values to prevent overflow while preserving proportions"""
    max_val = matrix.max()
    if max_val > eps:
        return matrix / max_val
    return matrix

def extract_scaled_fitch_tissue_trans(fitch_tissue_trans_data, clone):
    tissue_trans = torch.zeros((len(TISSUE_ORDER),len(TISSUE_ORDER)))
    if clone not in fitch_tissue_trans_data:
        print(f"No fitch tissue trans for clone {clone}")
        return tissue_trans 

    fitch_labels = fitch_tissue_trans_data[clone][1]
    fitch_tissue_trans = scale_matrix(torch.tensor(fitch_tissue_trans_data[clone][0]))
    fitch_tissue_trans.fill_diagonal_(0)

    tissue_trans = torch.zeros((len(TISSUE_ORDER),len(TISSUE_ORDER)))

    # Fill the universal matrix with values from the current matrix
    for i, row_label in enumerate(fitch_labels):
        for j, col_label in enumerate(fitch_labels):
            # Get the indices in the universal matrix
            row_idx = TISSUE_LABEL_TO_IDX[row_label]
            col_idx = TISSUE_LABEL_TO_IDX[col_label]
            tissue_trans[row_idx, col_idx] = fitch_tissue_trans[i, j]
    return tissue_trans

def extract_pars_metrics(pckl, tree_idx):
    loss_dict = pckl['loss_dict'][tree_idx]
    m = int(loss_dict['migration_number'])
    c = int(loss_dict['comigration_number'])
    s = int(loss_dict['seeding_site_number'])
    return m, c, s

def universally_ordered_mig_graph(G, ordered_sites):
    '''
    Take a migration graph G and create a new migration graph 
    where rows and columns are TISSUE_ORDER
    '''
    mig_graph = torch.zeros((len(TISSUE_ORDER),len(TISSUE_ORDER)))
    for i, row_label in enumerate(ordered_sites):
        for j, col_label in enumerate(ordered_sites):
            # Get the indices in the universal matrix
            row_idx = TISSUE_LABEL_TO_IDX[row_label]
            col_idx = TISSUE_LABEL_TO_IDX[col_label]
            # Set the value in the universal matrix
            mig_graph[row_idx, col_idx] = int(G[i, j])
    return mig_graph

def weighted_scaled_migration_graph(weights, mig_graphs):

    # mig_graphs is a list of 2D tensors (migration graphs)
    # and weights is a list of scalars (weights for each migration graph)
    # Initialize an empty tensor for the weighted sum
    weighted_summed_tissue_trans = torch.zeros_like(mig_graphs[0])  # Same shape as the first matrix
    
    # Perform weighted summation
    for w, m in zip(weights, mig_graphs):
        weighted_summed_tissue_trans += w * m
    
    weighted_summed_tissue_trans.fill_diagonal_(0)
    
    return scale_matrix(weighted_summed_tissue_trans)
    
def plot_overall_tissue_transiton(tissue_trans):
    # Overall tissue transition matrix
    fig = plt.figure(figsize = (2,2))
    g = sns.heatmap(tissue_trans, cmap="Reds", square=True, vmin = 0, vmax=1.0, cbar=False)
    g.set_xticks([x+0.5 for x in range(len(TISSUE_ORDER))],TISSUE_ORDER)
    g.set_yticks([x+0.5 for x in range(len(TISSUE_ORDER))],TISSUE_ORDER)
    plt.subplots_adjust(hspace=0.7, wspace=0.1)
    plt.tight_layout()
    plt.show()
    plt.close()


def all_migration_graphs(sorted_matrices, sorted_matrix_counts):
    matrices_per_row = 4
    num_matrices = len(sorted_matrices)
    num_rows = (num_matrices + matrices_per_row - 1) // matrices_per_row  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, matrices_per_row, figsize=(8, 2 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    # Set the overall color map and range

    vmin, vmax = 0, int(np.max([matrix.max() for matrix in sorted_matrices]))  # Shared color scale
    cmap = ListedColormap(sns.color_palette("YlGnBu", n_colors=vmax))
    ticks = np.arange(vmin, vmax + 1, (vmax-vmin)/10)  # Discrete ticks

    # Create heatmaps for unique matrices with counts
    for i, (matrix, count) in enumerate(zip(sorted_matrices, sorted_matrix_counts)):
        matrix = matrix.copy()
        np.fill_diagonal(matrix, np.nan)
        sns.heatmap(matrix, ax=axes[i], cmap="YlGnBu", square=True, 
                    vmin=vmin, vmax=vmax, cbar=False)
        axes[i].set_title(f'n = {count}')
        axes[i].set_xlabel('Destination tissue')
        axes[i].set_ylabel('Origin tissue')
        axes[i].set_xticks([x+0.5 for x in range(len(TISSUE_ORDER))],TISSUE_ORDER)
        axes[i].set_yticks([x+0.5 for x in range(len(TISSUE_ORDER))],TISSUE_ORDER)

    # Hide any unused axes
    for j in range(i + 1, num_rows * matrices_per_row):
        axes[j].axis('off')

    cbar_ax = fig.add_axes([0.3, 0.93, 0.4, 0.02])  # Adjust position of the color bar
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), 
                cax=cbar_ax, orientation='horizontal', ticks=ticks)
    cbar_ax.set_title('Migration count')

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to fit color bar

    plt.show()
    plt.close()


def create_dot_str(node_to_label, nx_digraph):

    # (1) Create full directed graph 
    root_node = None
    color_map = {}
    node_to_idx = {}
    max_idx = float("-inf")
    for i,node in enumerate(nx_digraph.nodes):
       
        label = node_to_label[node]
        color_map[node] = TISSUE_COLORS[TISSUE_ORDER.index(label)]
        if nx_digraph.in_degree(node) == 0:
            root_node = node
        node_to_idx[node] = i
        if i > max_idx: max_idx = i

    G = nx.DiGraph()

    node_options = {"label":"", "shape": "circle", "penwidth":3, 
                    "fontname":"Arial", "fontsize":12,
                    "fixedsize":"true", "height":0.25}

    new_idx = max_idx + 1

    node_to_index = {node: idx for idx, node in enumerate(nx_digraph.nodes())}
    # Sort edges by the node indices (not the labels)
    sorted_edges = sorted(nx_digraph.edges(), key=lambda x: (node_to_index[x[0]], node_to_index[x[1]]))

    for i, j in sorted_edges:
        label_i = i.split("_")[0]
        label_j = j
        j_is_real_cell = True if j.endswith("-1") else False
        
        G.add_node(node_to_idx[i], xlabel="", fillcolor=color_map[i], 
                    color=color_map[i], style="filled", **node_options)
        G.add_node(node_to_idx[j], xlabel="", fillcolor=color_map[j], 
                    color=color_map[j], style="filled", **node_options)
        
        minlen =  1.0
        if j_is_real_cell:
            G.add_node(new_idx, xlabel="",fontcolor="white", fillcolor=color_map[j], 
                    color=color_map[j], style="solid", **node_options)
            G.add_edge(node_to_idx[j], new_idx,color=color_map[j],
                   penwidth=5, arrowsize=0, style="dashed", minlen=minlen)
            new_idx += 1


        G.add_edge(node_to_idx[i], node_to_idx[j],color=f'"{color_map[i]};0.5:{color_map[j]}"', 
                   penwidth=5.5, arrowsize=0, style="solid", minlen=minlen)

    G.add_node("normal", label="", xlabel="root", penwidth=3, style="invis")
    G.add_edge("normal", node_to_idx[root_node], label="", 
                color=f'"{color_map[root_node]}"', 
                penwidth=4, arrowsize=0, style="solid")

    assert(nx.is_tree(G))

    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string().split("\n")
    # hack since there doesn't seem to be API to modify graph attributes...
    dot.insert(1, 'graph[splines=false]; nodesep=0.2; rankdir=TB; ranksep=0.5; forcelabels=true; dpi=1000; size=2.5; seed=42; layout=dot')
    dot_str = ("\n").join(dot)
    return dot_str

def plot_nx_tree(node_to_label, nx_digraph, output_name):
    '''
    '''    
    from metient.util import plotting_util as plutil
    import pygraphviz as pgv
    from PIL import Image as PILImage

    A = torch.tensor(nx.adjacency_matrix(nx_digraph).todense(), dtype=torch.float32)
    V = torch.zeros((len(TISSUE_ORDER), A.shape[0]))
    node_to_node_idx = {node: idx for idx, node in enumerate(nx_digraph.nodes())}

    for node, tissue in node_to_label.items():
        tissue_index = TISSUE_ORDER.index(tissue)
        V[tissue_index, node_to_node_idx[node]] = 1

    tree_dot = create_dot_str(node_to_label, nx_digraph)
    mig_graph_dot = plutil.migration_graph_dot(V, A, TISSUE_ORDER, TISSUE_COLORS, show=False)
    mig_graph_no_diag = met.migration_graph(V, A)
    print(f"Total migrations: {int(torch.sum(mig_graph_no_diag).item())}")
    print("mig_graph_no_diag\n",mig_graph_no_diag)

    k = 1
    sys_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    for font in sys_fonts:
        if FONT in font:
            matplotlib.font_manager.fontManager.addfont(font)
            rcParams['font.family'] = FONT

    n = 1
    z = 2 # number of trees displayed per row

    nrows = math.ceil(n/z)
    h = nrows*4
    fig = plt.figure(figsize=(10,h),dpi=800)
    
    vspace = 1/nrows

    tree = pgv.AGraph(string=tree_dot).draw(format="png", prog="dot", args="-Glabel=\"\"")
    tree = PILImage.open(io.BytesIO(tree))
    mig_graph = pgv.AGraph(string=mig_graph_dot).draw(format="png", prog="dot")
    mig_graph = PILImage.open(io.BytesIO(mig_graph))

    gs = gridspec.GridSpec(3, 1, height_ratios=[0.02, 0.5, 0.5])

    row,i = 0,0
    pad = 0.02

    # left = 0.0 if i is odd, 0.55 if even
    # right = 0.45 if i is odd, 1.0 if even
    gs.update(left=0.0+((i%2)*0.53), right=0.47+0.55*(i%2), top=1-(row*vspace)-pad, bottom=1-((row+1)*vspace)+pad, wspace=0.05)

    # Top row: Title
    ax1 = plt.subplot(gs[0])
    # ax1.text(0.5, 0.5, f'Solution {i+1}\n{seeding_pattern}', ha='center', va='center', fontsize=7)
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
    # ax4.text(0.5, 0.5, formatted_loss_string(loss_info, weights), ha='center', va='center', fontsize=7)
    ax4.axis('off')

    fig1 = plt.gcf()
    plt.show()
    plt.close()
    output_dir = "/data/morrisq/divyak/projects/metient/notebooks/lineage_tracing/outputs"
    fig1.savefig(os.path.join(output_dir, f'{output_name}.png'), dpi=1200, bbox_inches='tight')
        
    return 
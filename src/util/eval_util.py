import pickle
import glob
import heapq
import torch
import os
import re
import pandas as pd
import json

from src.util.globals import *
from src.util import plotting_util as plot_util

######### Methods taken directly from MACHINA repo #########

def load_machina_results(machina_results_dir):
    # Taken directly from https://github.com/raphael-group/machina/blob/master/result/sims/simulations_1.ipynb
    files_new_m5=['results_MACHINA_m5.txt',]
    files_new_m8=['results_MACHINA_m8.txt', ]
    res_m5 = pd.concat([pd.read_csv(os.path.join(machina_results_dir, filename)) for filename in files_new_m5]).reindex()
    res_m8 = pd.concat([pd.read_csv(os.path.join(machina_results_dir, filename)) for filename in files_new_m8]).reindex()
    res_m5 = res_m5[(res_m5['enforced']=='R') | (res_m5['enforced'].isnull())]
    res_m8 = res_m8[(res_m8['enforced']=='R') | (res_m8['enforced'].isnull())]
    res_m5 = res_m5.replace({'pattern': {'S': 'pS', 'M' : 'pM', 'R' : 'pR'}})
    res_m8 = res_m8.replace({'pattern': {'S': 'pS', 'M' : 'pM', 'R' : 'pR'}})
    res_m8_MACHINA = res_m8[res_m8['method'] == 'MACHINA'].replace({'inferred': {'pPS': 'pS', 'mPS' : 'mS'}})
    res_m5_MACHINA = res_m5[res_m5['method'] == 'MACHINA'].replace({'inferred': {'pPS': 'pS', 'mPS' : 'mS'}})

    return res_m5_MACHINA, res_m8_MACHINA

# Taken from MACHINA 
def get_mutations(edge_list, u):
    # find parent
    s = re.split("_|;|^", u)
    mutations = set()
    for i in s:
        if i.isdigit():
            mutations.add(int(i))
    #print mutations

    for edge in edge_list:
        uu = edge[0]
        vv = edge[1]
        if vv == u:
            return mutations | get_mutations(edge_list, uu)

    return mutations

# Taken from MACHINA 
def parse_clone_tree(filename_T, filename_l):
    edges = []
    with open(filename_T) as f:
        for line in f:
            s = line.rstrip("\n").split(" ")
            edges += [(s[0], s[1])]

    labeling = {}
    with open(filename_l) as f:
        for line in f:
            s = line.rstrip("\n").split(" ")
            labeling[s[0]] = s[1]

    # find migration edges
    migration_edges = []
    for (u, v) in edges:
        if labeling[u] != labeling[v]:
            migration_edges += [(u,v)]

    return edges, migration_edges,labeling

# Taken from MACHINA 
def identify_seeding_clones(edge_list, migration_edge_list):
    res = set()
    for (u,v) in migration_edge_list:
        muts_u = get_mutations(edge_list, u)
        muts_v = get_mutations(edge_list, v)
        res.add(frozenset(muts_u))

    return res

# Taken from MACHINA 
def parse_migration_graph(filename_G):
    edges = []
    with open(filename_G) as f:
        for line in f:
            s = line.rstrip("\n").split(" ")
            edges += [(s[0], s[1])]
    
    return edges

# Taken from MACHINA 
def multi_graph_to_set(edge_list):
    count = {}
    res = set()
    for edge in edge_list:
        if edge not in count:
            count[edge] = 1
        else:
            count[edge] += 1
        res.add((edge[0], edge[1], count[edge]))
    return res

########### Methods for evaluating Metient ouputs #############
def metient_parse_clone_tree(results_dict, met_tree_num):
    V = results_dict[OUT_LABElING_KEY][met_tree_num]
    A = results_dict[OUT_ADJ_KEY]
    sites = results_dict[OUT_SITES_KEY]
    G = plot_util.get_migration_graph(V, A)
    idx_to_lbl = results_dict[OUT_IDX_LABEL_KEY][met_tree_num]
    
    edges = [("GL", '0')]
    for i, j in plot_util.tree_iterator(A):
        edges.append((idx_to_lbl[i][0], idx_to_lbl[j][0]))
    
    migration_edges = []
    for i, j in plot_util.tree_iterator(A):
        if torch.argmax(V[:,i]) != torch.argmax(V[:,j]):
            migration_edges.append((idx_to_lbl[i][0], idx_to_lbl[j][0]))
    return edges, migration_edges

def metient_parse_mig_graph(results_dict, met_tree_num):
    V = results_dict[OUT_LABElING_KEY][met_tree_num]
    A = results_dict[OUT_ADJ_KEY]
    sites = results_dict[OUT_SITES_KEY]
    G = plot_util.get_migration_graph(V, A)
    migration_edges = []
    for i, row in enumerate(G):
        for j, val in enumerate(row):
            if val != 0:
                for _ in range(int(val)):
                    migration_edges.append((sites[i], sites[j]))
    return migration_edges

class HeapTree:
    def __init__(self, loss, results_dict, met_tree_num):
        self.loss = loss
        self.results_dict = results_dict
        self.met_tree_num = met_tree_num

   # override the comparison operator
    def __lt__(self, other):
        return self.loss < other.loss
    
def get_metient_min_loss_trees(site_mig_type_dir, seed, k, loss_thres=1.0):
    # Get all clone trees for the seed
    tree_pickles = glob.glob(os.path.join(site_mig_type_dir, f"*_seed{seed}.pickle"))
    
    # Keep track of the best trees and losses
    min_heap = [] 
    for tree_pickle in tree_pickles:
        clone_tree_num = os.path.basename(tree_pickle).replace("tree", "").replace(f"_seed{seed}.pickle", "")
        f = open(tree_pickle,'rb')
        results_dict = pickle.load(f)
        for met_tree_num, loss in enumerate(results_dict['losses']):
            heapq.heappush(min_heap, HeapTree(loss, results_dict, met_tree_num))
    out = []
    min_loss = min_heap[0].loss

    while len(out) < k and len(min_heap) > 0:
        item = heapq.heappop(min_heap)
        # only add tree if it's ~= to the min loss
        if abs(min_loss-item.loss) <= loss_thres:
            out.append((item.loss, item.results_dict, item.met_tree_num))
        
    return out

######## Methods for comparing Metient outputs to ground truth ############

def evaluate_seeding_clones(sim_clone_tree_fn, sim_vert_labeling_fn, met_results_dict, met_tree_num):
    edges_simulated, mig_edges_simulated, _ = parse_clone_tree(sim_clone_tree_fn, sim_vert_labeling_fn)
    seeding_clones_simulated = identify_seeding_clones(edges_simulated, mig_edges_simulated)
    edges_inferred, mig_edges_inferred = metient_parse_clone_tree(met_results_dict, met_tree_num)
    seeding_clones_inferred = identify_seeding_clones(edges_inferred, mig_edges_inferred)
    contains_resolved_polytomy = False
    for edge in edges_simulated:
        if is_resolved_polytomy_cluster(edge[0]) or is_resolved_polytomy_cluster(edge[1]):
            contains_resolved_polytomy = True
    recall = float(len(seeding_clones_inferred & seeding_clones_simulated)) / float(len(seeding_clones_simulated))
    precision = float(len(seeding_clones_inferred & seeding_clones_simulated)) / float(len(seeding_clones_inferred))
    if recall == 0 or precision == 0:
        F = 0
    else:
        F = 2.0 / ((1.0 / recall) + (1.0 / precision))

    return recall, precision, F, contains_resolved_polytomy

def evaluate_migration_graph(sim_mig_graph_fn, met_results_dict, met_tree_num):
    edge_set_G_simulated = set(parse_migration_graph(sim_mig_graph_fn))
    edge_set_G_inferred = set(metient_parse_mig_graph(met_results_dict, met_tree_num))
    recall_G = float(len(edge_set_G_inferred & edge_set_G_simulated)) / float(len(edge_set_G_simulated))
    precision_G = float(len(edge_set_G_inferred & edge_set_G_simulated)) / float(len(edge_set_G_inferred))

    if recall_G != 0 and precision_G != 0:
        F_G = 2.0 / ((1.0 / recall_G) + (1.0 / precision_G))
    else:
        F_G = 0

    return recall_G, precision_G, F_G

def evaluate_migration_multigraph(sim_mig_graph_fn, met_results_dict, met_tree_num):
    edge_multiset_G_simulated = multi_graph_to_set(parse_migration_graph(sim_mig_graph_fn))
    edge_multiset_G_inferred = multi_graph_to_set(metient_parse_mig_graph(met_results_dict, met_tree_num))
    recall_G2 = float(len(edge_multiset_G_inferred & edge_multiset_G_simulated)) / float(len(edge_multiset_G_simulated))
    precision_G2 = float(len(edge_multiset_G_inferred & edge_multiset_G_simulated)) / float(len(edge_multiset_G_inferred))
    if recall_G2 != 0 and precision_G2 != 0:
        F_G2 = 2.0 / ((1.0 / recall_G2) + (1.0 / precision_G2))
    else:
        F_G2 = 0

    return recall_G2, precision_G2, F_G2

def is_resolved_polytomy_cluster(cluster_label):
    '''
    In MACHINA simulated data, cluster labels with non-numeric components (e.g. M2_1
    instead of 1;3;4) represent polytomies
    '''
    is_polytomy = False
    for mut in cluster_label.split(";"):
        if not mut.isnumeric() and (mut.startswith('M') or mut.startswith('P')):
            is_polytomy = True
    return is_polytomy
    
    
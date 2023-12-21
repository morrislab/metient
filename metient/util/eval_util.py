import pickle
import glob
import heapq
import torch
import os
import re
import pandas as pd
import json
import joblib
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from metient.util.globals import *
from metient.util import plotting_util as plot_util

def plot_cross_ent_chart(data_dict, output_dir):
    fig = plt.figure(figsize=(2.5, 3),dpi=200)
    
    keys = list(data_dict.keys())
    sorted_keys = sorted(keys, key=lambda x: x[0], reverse=True)
    string_keys = [str((round(k[0], 2),round(k[1], 2),round(k[2], 2))) for k in sorted_keys]
    values = [-1*data_dict[k].item() for k in sorted_keys]
    snsfig = sns.barplot(x=string_keys, y=values, palette=sns.color_palette("muted"),
                         order=string_keys)

    snsfig.spines['top'].set_visible(False)
    snsfig.spines['right'].set_visible(False)

    plt.xlabel("Theta: (migration weight, seeding site weight, \nmigration delta weight)",fontsize=10)
    plt.ylabel("-E(theta)",fontsize=15)
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=12)
    plt.ylim(min(values)-5, max(values)+5)
    fig.savefig(os.path.join(output_dir, f"cross_entropy_distribution.png"), dpi=600, bbox_inches='tight', pad_inches=0.5) 
    plt.close()

def stable_softmax(x):
    # Subtract the maximum value for numerical stability
    x_max, _ = torch.max(x, dim=0, keepdim=True)
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=0)

def cross_ent(loss_dicts, thetas, tau):
    '''
    Computes the cross entropy between the target distribution (genetic distance)
    and the predicted distribution (parsimony) using the input thetas (which we are
    trying to optimize)
    '''
    theta_X = torch.zeros((len(loss_dicts)))
    gen_dist_scores = torch.zeros((len(loss_dicts)))
    organotrop_scores = torch.zeros((len(loss_dicts)))
    for i, loss_dict in enumerate(loss_dicts):
        m = loss_dict[MIG_KEY]
        c = loss_dict[COMIG_KEY]
        s = loss_dict[SEEDING_KEY]
        g = loss_dict[GEN_DIST_KEY]
        o = loss_dict[ORGANOTROP_KEY]
        theta_X[i] = -1.0*(thetas[0]*m + thetas[1]*c + thetas[2]*s)
        gen_dist_scores[i] = -tau*g
        organotrop_scores[i] = -tau*o
    #print()
    #print("theta x", theta_X, "\n gen dist", gen_dist_scores, "\n organotrop", organotrop_scores)
    # this is in the case where there is no seeding detected (happens rarely in some datasets)
    if torch.sum(theta_X) == 0 or torch.isnan(gen_dist_scores).any():
        return 0.0
    theta_X = stable_softmax(theta_X)

    cross_ent_sum = 0.0
    if not torch.sum(gen_dist_scores == 0):
        gen_dist_scores = stable_softmax(gen_dist_scores)
        cross_ent_sum += -1*torch.sum(torch.mul(gen_dist_scores, torch.log2(theta_X+0.1)))
    if not torch.sum(organotrop_scores == 0):
        organotrop_scores = stable_softmax(organotrop_scores)
        cross_ent_sum += -1*torch.sum(torch.mul(organotrop_scores, torch.log2(theta_X+0.1)))

    #cross_ent = torch.mul(gen_dist_scores, torch.log2(theta_X+0.1))
    #print("thetaX", theta_X, "gen_dist_scores", gen_dist_scores, "organotrop_scores", organotrop_scores)

    #print("cross_ent_sum", cross_ent_sum)
    return cross_ent_sum

def get_pickle_filenames(pickle_files_dirs, suffix=None):
    pickle_filenames = []
    match = f"{suffix}.pkl.gz" if suffix != None else ".pkl.gz" 
    for pickle_files_dir in pickle_files_dirs:
        for file in os.listdir(pickle_files_dir):
            if match in file:
                pickle_filenames.append(os.path.join(pickle_files_dir, file))
    return pickle_filenames
    
def get_max_cross_ent_thetas(pickle_file_dirs=None, pickle_file_list=None, tau=3.0, suffix=None, use_min_tau=False):
    '''
    pickle_file_dirs: list of directories to search for Metient results
    with .pkl.gz files

    or, just pass the paths to the actual pickle files

    Returns the parsimony weights which give the best cross entropy 
    between the target distribution (genetic distance) and predicted 
    distribution (parsimony) across all patients in pickle_files_dir
    '''
    if pickle_file_list == None :
        pickle_file_list = get_pickle_filenames(pickle_file_dirs, suffix)
    num_patients = 0
    
    min_tau = float("inf")
    #print("Restricting to patients wih > 2 sites")
    all_loss_dicts = []
    for pkl_file in pickle_file_list:
        with gzip.open(pkl_file,'rb') as f:
            pckl = pickle.load(f)
        
        sites = pckl[OUT_SITES_KEY]
        # if len(sites) > 2:
        loss_dicts = pckl[OUT_LOSS_DICT_KEY]
        all_loss_dicts.append(loss_dicts)
        if use_min_tau:
            gen_dist_scores = torch.zeros((len(loss_dicts)))
            for i, loss_dict in enumerate(loss_dicts):
                gen_dist_scores[i] = loss_dict[GEN_DIST_KEY]
            unique_gs = torch.unique(gen_dist_scores)
            if len(unique_gs) >= 2:
                print("gen dist diff", (unique_gs[1]-unique_gs[0]))
                min_tau = min(min_tau, (unique_gs[1]-unique_gs[0]))
    
    if use_min_tau:
        tau = min_tau
        print("min_tau", tau)
        
    print(f"Calibrating to {len(all_loss_dicts)} patients")
    
    raw_thetas = torch.tensor([1.0,1.0,1.0], requires_grad=True)
    optimizer = optim.SGD([raw_thetas], lr=0.01)
    max_iter = 200

    for step in range(max_iter):
        optimizer.zero_grad()
        thetas = stable_softmax(raw_thetas)
        #thetas = torch.exp(raw_thetas)
        total_cross_ent = 0.0
        for patient_loss_dicts in all_loss_dicts:
            patient_cross_ent = cross_ent(patient_loss_dicts, thetas, tau)
            total_cross_ent += patient_cross_ent
            # print("total_cross_ent", total_cross_ent)

        total_cross_ent.backward()
        optimizer.step()

        if step % 30 == 0:
            pass
            #print(f"Step {step}: theta = {thetas}, Objective = {total_cross_ent.item()}")

    print(f"Optimized thetas: {thetas}")

    return [float(thetas[0]), float(thetas[1]), float(thetas[2])]

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
    V = torch.tensor(results_dict[OUT_LABElING_KEY][met_tree_num])
    A = torch.tensor(results_dict[OUT_ADJ_KEY][met_tree_num])
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
    V = torch.tensor(results_dict[OUT_LABElING_KEY][met_tree_num])
    A = torch.tensor(results_dict[OUT_ADJ_KEY][met_tree_num])
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
    def __init__(self, loss, results_dict, met_tree_num, tree_num):
        self.loss = loss
        self.results_dict = results_dict
        self.met_tree_num = met_tree_num
        self.tree_num = tree_num

   # override the comparison operator
    def __lt__(self, other):
        return self.loss < other.loss
    
def get_metient_min_loss_trees(site_mig_type_dir, seed, k, loss_thres=1.0, suffix=""):
    # Get all clone trees for the seed
    tree_pickles = glob.glob(os.path.join(site_mig_type_dir, f"*_seed{seed}{suffix}.pkl.gz"))
    
    # Keep track of the best trees and losses
    min_heap = [] 
    for tree_pickle in tree_pickles:
        clone_tree_num = os.path.basename(tree_pickle).replace("tree", "").replace(f"_seed{seed}{suffix}.pkl.gz", "")
        with gzip.open(tree_pickle,'rb') as f:
            results_dict = joblib.load(f)
        for met_tree_num, loss in enumerate(results_dict['losses']):
            heapq.heappush(min_heap, HeapTree(loss, results_dict, met_tree_num, clone_tree_num))
    out = []
    min_loss = min_heap[0].loss
    while len(out) < k and len(min_heap) > 0:
        item = heapq.heappop(min_heap)
        # only add tree if it's ~= to the min loss
        if abs(min_loss-item.loss) <= loss_thres:
            out.append((item.loss, item.results_dict, item.met_tree_num))
    print("# min loss trees:", len(out))
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
    
    
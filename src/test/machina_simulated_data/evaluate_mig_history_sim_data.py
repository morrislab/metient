#!/usr/bin/python
import sys
import re
import os
import fnmatch
import numpy as np
import pprint


# plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="whitegrid")
sns.set(font_scale=1.5)

sns.set_style("whitegrid")
sns.set_style("ticks")
sns.despine()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pc_map = {'mS':0, 'pS': 1, 'mM': 2, 'pM': 3, 'mR': 4, 'pR': 5}

from statannot import add_stat_annotation

# TODO: This is in util/machina_data_extraction_util.py but there's an error being thrown 
# anytime I import that file here:
# OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# Abort trap: 6
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

    return edges, migration_edges

def identify_seeding_clones(edge_list, migration_edge_list):
    res = set()
    for (u,v) in migration_edge_list:
        muts_u = get_mutations(edge_list, u)
        muts_v = get_mutations(edge_list, v)
        res.add(frozenset(muts_u))

    return res

def parse_migration_graph(filename_G):
    edges = []
    with open(filename_G) as f:
        for line in f:
            s = line.rstrip("\n").split(" ")
            edges += [(s[0], s[1])]

    return edges

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

def evaluate_seeding_clones(sim_clone_tree_fn, sim_vert_labeling_fn, pred_clone_tree_fn, pred_vert_labeling_fn):
    edges_simulated, mig_edges_simulated = parse_clone_tree(sim_clone_tree_fn, sim_vert_labeling_fn)
    seeding_clones_simulated = identify_seeding_clones(edges_simulated, mig_edges_simulated)
    edges_inferred, mig_edges_inferred = parse_clone_tree(pred_clone_tree_fn, pred_vert_labeling_fn)
    seeding_clones_inferred = identify_seeding_clones(edges_inferred, mig_edges_inferred)
    #print("edges_simulated", edges_simulated)
    contains_resolved_polytomy = False
    for edge in edges_simulated:
        if is_resolved_polytomy_cluster(edge[0]) or is_resolved_polytomy_cluster(edge[1]):
            contains_resolved_polytomy = True
    # print("mig_edges_simulated", mig_edges_simulated)
    # print("seeding_clones_simulated", seeding_clones_simulated)
    # print("edges_inferred", edges_inferred)
    # print("mig_edges_inferred", mig_edges_inferred)
    # print("seeding_clones_inferred", seeding_clones_inferred)
    recall = float(len(seeding_clones_inferred & seeding_clones_simulated)) / float(len(seeding_clones_simulated))
    precision = float(len(seeding_clones_inferred & seeding_clones_simulated)) / float(len(seeding_clones_inferred))
    if recall == 0 or precision == 0:
        F = 0
    else:
        F = 2.0 / ((1.0 / recall) + (1.0 / precision))

    return recall, precision, F, contains_resolved_polytomy

def evaluate_migration_graph(sim_mig_graph_fn, predicted_mig_graph_fn):
    edge_set_G_simulated = set(parse_migration_graph(sim_mig_graph_fn))
    edge_set_G_inferred = set(parse_migration_graph(predicted_mig_graph_fn))

    recall_G = float(len(edge_set_G_inferred & edge_set_G_simulated)) / float(len(edge_set_G_simulated))
    precision_G = float(len(edge_set_G_inferred & edge_set_G_simulated)) / float(len(edge_set_G_inferred))

    if recall_G != 0 and precision_G != 0:
        F_G = 2.0 / ((1.0 / recall_G) + (1.0 / precision_G))
    else:
        F_G = 0

    return recall_G, precision_G, F_G

def evaluate_migration_multigraph(sim_mig_graph_fn, predicted_mig_graph_fn):
    edge_multiset_G_simulated = multi_graph_to_set(parse_migration_graph(sim_mig_graph_fn))
    edge_multiset_G_inferred = multi_graph_to_set(parse_migration_graph(predicted_mig_graph_fn))
    recall_G2 = float(len(edge_multiset_G_inferred & edge_multiset_G_simulated)) / float(len(edge_multiset_G_simulated))
    precision_G2 = float(len(edge_multiset_G_inferred & edge_multiset_G_simulated)) / float(len(edge_multiset_G_inferred))

    if recall_G2 != 0 and precision_G2 != 0:
        F_G2 = 2.0 / ((1.0 / recall_G2) + (1.0 / precision_G2))
    else:
        F_G2 = 0

    return recall_G2, precision_G2, F_G2

def save_boxplot(df, y, num_sites, fig_name):
    seeding_pattern_order = ["mS", "pS", "pM", "pR"]
    box_pairs = []

    for seeding_pattern in seeding_pattern_order:
        box_pairs.append(((seeding_pattern, "Gradient-based"),(seeding_pattern, "MACHINA")))
    flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='grey', alpha=0.5)
    ax = sns.boxplot(x="seeding pattern", y=y, hue="method", data=df, order=seeding_pattern_order, 
                     palette=sns.color_palette("pastel"), flierprops=flierprops, linewidth=2)
    add_stat_annotation(ax, data=df, x="seeding pattern", y=y, hue="method",
                        box_pairs=box_pairs,
                        test='t-test_welch', text_format='star', loc='inside', verbose=0, order=seeding_pattern_order, fontsize=18, comparisons_correction=None)
    ax.set(ylim=(-0.1, 1.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Seeding Pattern', fontsize=16, fontweight='bold')
    ax.set_ylabel(y.capitalize(), fontsize=16, fontweight='bold')
    ax.set_title(f"{num_sites} Anatomical Sites" , fontsize=16, fontweight='bold', y=1.1)
    plt.legend(frameon=False, loc="lower right")
    ax.get_figure().savefig("../output_plots/"+fig_name)
    plt.clf()

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

def extract_minimum_loss_trees(loss_output_txt_fn, sites, mig_types):
    '''
    Get all the trees per unique site+mig_type+seed combo with minimum loss.
    Returns dict in the following format:
    {'m5':
        {'M':
            {'<seed_num>': [list of all trees with same min loss],
            }
         'mS': ...
        }
    }
    '''

    output = { site : { mig_type: dict() for mig_type in mig_types } for site in sites }
    loss_values = []
    with open(loss_output_txt_fn, 'r') as f:
        for line in f:
            if "Predicting vertex labeling" in line:
                if len(loss_values) > 0:
                    x = np.array(loss_values)
                    min_trees = np.where(x == x.min())[0]
                    output[site_num][mig_type][seed] = list(min_trees)
                items = line.strip().split() # in format "Predicting vertex labeling for m8 M seed 241.""
                site_num = items[4]
                mig_type = items[5]
                seed = items[7][:-1] # remove period
                loss_values = []
            elif "Loss" in line:
                items = line.strip().split() # in format "Loss: 117.411"
                loss = float(items[1])
                loss_values.append(loss)

    # for the last tree in the results
    x = np.array(loss_values)
    min_trees = np.where(x == x.min())[0]
    output[site_num][mig_type][seed] = list(min_trees)

    pprint.pprint(output)
    return(output)


def get_min_loss_trees_df(df, site, mig_type, seed):

    df = df.astype({'site': 'string', 'mig_type': 'string'})
    subset = df[(df['site'] == site) & (df['mig_type'] == mig_type) & (df['seed'] == seed)]
    subset = subset[(subset['loss'] == subset.loss.min())]
    return [l-1 for l in list(subset.tree_num)]


if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <PREDICTIONS_DATA_DIR> <LOSS_OUTPUT_TXT> <RUN_NAME>\n" % sys.argv[0])
        sys.stderr.write("MACHINA_SIM_DATA_DIR: directory containing the true labelings\n")
        sys.stderr.write("PREDICTIONS_DATA_DIR: directory containing the predicted labelings\n")
        sys.stderr.write("LOSS_OUTPUT_TXT: txt file containing the loss values for all the mutation trees within a seed. Only trees with minimum loss value are used.\n")
        sys.stderr.write("RUN_NAME: Name of the run to use in output png naming.\n")
        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    results = {s : {m : [] for m in mig_types} for s in sites}
    grad_m5_f1_scores = []
    grad_m8_f1_scores = []

    mach_sim_data_dir = sys.argv[1]
    predictions_data_dir = sys.argv[2]
    loss_output_txt_fn = sys.argv[3]
    run_name = sys.argv[4]

    i = 0

    sgd_results_df = pd.read_csv(loss_output_txt_fn)
    print(sgd_results_df.head())

    for site in sites:
        for mig_type in mig_types:
            true_site_mig_type_data_dir = os.path.join(mach_sim_data_dir, site, mig_type)
            predicted_site_mig_type_data_dir = os.path.join(predictions_data_dir, site, mig_type)
            filenames = fnmatch.filter(os.listdir(predicted_site_mig_type_data_dir), 'T_tree*.predicted.tree')
            seeds = set([int(s[s.find("seed")+4:s.find(".predicted")]) for s in filenames])

            for seed in seeds:
                seed_filenames = [f for f in filenames if seed == f[f.find("seed")+4:f.find(".predicted")]]
                #trees = [t[t.find("tree")+4:t.find("_seed")] for t in filenames if seed == t[t.find("seed")+4:t.find(".predicted")]]
                tree_nums = get_min_loss_trees_df(sgd_results_df, site, mig_type, seed)
                for tree_num in tree_nums:

                    #print(f"Evaluating history for seed {seed} {site} {mig_type} tree {tree}")

                    recall, precision, F, contains_resolved_polytomy = evaluate_seeding_clones(os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.tree"),
                                                                   os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.vertex.labeling"),
                                                                   os.path.join(predicted_site_mig_type_data_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"),
                                                                   os.path.join(predicted_site_mig_type_data_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"))

                    # if contains_resolved_polytomy:
                    #     continue

                    recall_G, precision_G, F_G = evaluate_migration_graph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                          os.path.join(predicted_site_mig_type_data_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))

                    recall_G2, precision_G2, F_G2 = evaluate_migration_multigraph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                  os.path.join(predicted_site_mig_type_data_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))


                    scores = [recall, precision, F, recall_G, precision_G, F_G, recall_G2, precision_G2, F_G2]
                    #print(",".join(map(str, scores)))

                    # rename "S", "M", "R" -> "pS", "pM", "pR"
                    mig_name = mig_type if len(mig_type) == 2 else "p"+mig_type
                    if site == 'm5':
                        grad_m5_f1_scores.append([seed, mig_name, F, F_G2])
                    elif site == 'm8':
                        grad_m8_f1_scores.append([seed, mig_name, F,  F_G2])

                    i += 1
                    results[site][mig_type].append(scores)

    print("num trees:",  i)

    # Plot results
    grad_m5_df = pd.DataFrame(grad_m5_f1_scores, columns=["seed", "seeding pattern",  "migrating clones F1 score", "migration graph F1 score"])
    grad_m8_df = pd.DataFrame(grad_m8_f1_scores, columns=["seed", "seeding pattern",  "migrating clones F1 score", "migration graph F1 score"])

    grad_m5_df = grad_m5_df.groupby(['seeding pattern','seed']).mean().assign(method="Gradient-based")
    grad_m8_df = grad_m8_df.groupby(['seeding pattern','seed']).mean().assign(method="Gradient-based")

    print("\nGradient-based m5 avg F1 scores")
    print(grad_m5_df.groupby('seeding pattern').mean())

    print("\nGradient-based m8 avg F1 scores")
    print(grad_m8_df.groupby('seeding pattern').mean())

    # Load machina results
    machina_m5_df, machina_m8_df = load_machina_results(".")
    col_mapping = {"FscoreT": "migrating clones F1 score", "FscoreMultiG": "migration graph F1 score", "pattern": "seeding pattern", "seed":"seed"}
    machina_m5_df = machina_m5_df.rename(columns=col_mapping)
    machina_m8_df = machina_m8_df.rename(columns=col_mapping)

    machina_m5_df = machina_m5_df[col_mapping.values()]
    machina_m5_df = machina_m5_df.groupby(['seeding pattern','seed']).mean().assign(method="MACHINA")
    machina_m8_df = machina_m8_df[col_mapping.values()]
    machina_m8_df = machina_m8_df.groupby(['seeding pattern','seed']).mean().assign(method="MACHINA")

    print("MACHINA m5 avg F1 scores")
    print(machina_m5_df.groupby('seeding pattern').mean())

    print("MACHINA m8 avg F1 scores")
    print(machina_m8_df.groupby('seeding pattern').mean())

    joint_m5_df = pd.concat([grad_m5_df, machina_m5_df]).reset_index()
    joint_m8_df = pd.concat([grad_m8_df, machina_m8_df]).reset_index()
    print(joint_m5_df.reset_index())

    save_boxplot(joint_m5_df, "migration graph F1 score", 5, f"m5_migration_graph_f1_scores_{run_name}.png")
    save_boxplot(joint_m8_df, "migration graph F1 score", 8, f"m8_migration_graph_f1_scores_{run_name}.png")

    save_boxplot(joint_m5_df, "migrating clones F1 score", 5,  f"m5_migrating_clones_f1_scores_{run_name}.png")
    save_boxplot(joint_m8_df, "migrating clones F1 score", 8,  f"m8_migrating_clones_f1_scores_{run_name}.png")

    # Timing benchmarks
    # TODO: put this in a different script
    print(sgd_results_df['time'])
    print(sgd_results_df.dtypes)
    #sgd_results_df['time'] = pd.to_datetime(sgd_results_df['time'])
    print(len(sgd_results_df[(sgd_results_df['site']=='m8') & (sgd_results_df['seed']==0) & (sgd_results_df['mig_type']=='S')]))
    print(sgd_results_df.groupby(['site']).mean())
    print("m8 trees", len(sgd_results_df[(sgd_results_df['site']=='m8') & ((sgd_results_df['mig_type']=='mS') | (sgd_results_df['mig_type']=='S'))]))
    print("m5 trees", len(sgd_results_df[(sgd_results_df['site']=='m5')]))
    print("m8 trees", len(sgd_results_df[(sgd_results_df['site']=='m8')]))
    print(sgd_results_df["time"])
    # m5_total = 0.0
    # m8_total = 0.0
    # with open("pmh_ti_timing_8cores_11152022.txt", "r") as f:
    #     for line in f:
    #         if "Runtime" in line:
    #             secs = float(line[line.find(":")+2:])
    #             if "m5" in line:
    #                 m5_total += secs
    #             elif "m8" in line:
    #                 m8_total += secs
    # print(m5_total/586, m8_total/581)

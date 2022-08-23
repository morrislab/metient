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
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pc_map = {'mS':0, 'pS': 1, 'mM': 2, 'pM': 3, 'mR': 4, 'pR': 5}

from statannot import add_stat_annotation

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
    # print("edges_simulated", edges_simulated)
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

    return recall, precision, F

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

def save_boxplot(df, y, fig_name):
    seeding_pattern_order = ["mS", "pS", "pM", "pR"]
    box_pairs = []

    for seeding_pattern in seeding_pattern_order:
        box_pairs.append(((seeding_pattern, "SGD"),(seeding_pattern, "MACHINA")))
    ax = sns.boxplot(x="seeding pattern", y=y, hue="method", data=df, order=seeding_pattern_order, palette=sns.color_palette("Set2"))
    add_stat_annotation(ax, data=df, x="seeding pattern", y=y, hue="method",
                        box_pairs=box_pairs,
                        test='t-test_welch', text_format='star', loc='inside', verbose=0, order=seeding_pattern_order, fontsize='large', comparisons_correction=None)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))
    ax.set(ylim=(0.0, 1.1))
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



if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <LOSS_OUTPUT_TXT>\n" % sys.argv[0])
        sys.stderr.write("MACHINA_SIM_DATA_DIR: directory containing the true labelings and predicted labelings")
        sys.stderr.write("LOSS_OUTPUT_TXT: txt file containing the loss values for all the mutation trees within a seed. Only trees with minimum value are used")
        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    results = {s : {m : [] for m in mig_types} for s in sites}
    grad_m5_f1_scores = []
    grad_m8_f1_scores = []

    mach_sim_data_dir = sys.argv[1]
    loss_output_txt_fn = sys.argv[2]

    min_loss_trees = extract_minimum_loss_trees(loss_output_txt_fn, sites, mig_types)
    i = 0

    for site in sites:
        for mig_type in mig_types:
            mig_type_data_dir = os.path.join(mach_sim_data_dir, site, mig_type)
            filenames = fnmatch.filter(os.listdir(mig_type_data_dir), 'T_tree*.predicted.tree')
            seeds = set([s[s.find("seed")+4:s.find(".predicted")] for s in filenames])

            for seed in seeds:
                seed_filenames = [f for f in filenames if seed == f[f.find("seed")+4:f.find(".predicted")]]
                #trees = [t[t.find("tree")+4:t.find("_seed")] for t in filenames if seed == t[t.find("seed")+4:t.find(".predicted")]]

                trees = min_loss_trees[site][mig_type][seed]
                for tree in trees:

                    #print(f"Evaluating history for seed {seed} {site} {mig_type} tree {tree}")

                    recall, precision, F = evaluate_seeding_clones(os.path.join(mig_type_data_dir, f"T_seed{seed}.tree"),
                                                                   os.path.join(mig_type_data_dir, f"T_seed{seed}.vertex.labeling"),
                                                                   os.path.join(mig_type_data_dir, f"T_tree{tree}_seed{seed}.predicted.tree"),
                                                                   os.path.join(mig_type_data_dir, f"T_tree{tree}_seed{seed}.predicted.vertex.labeling"))

                    recall_G, precision_G, F_G = evaluate_migration_graph(os.path.join(mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                          os.path.join(mig_type_data_dir, f"G_tree{tree}_seed{seed}.predicted.tree"))

                    recall_G2, precision_G2, F_G2 = evaluate_migration_multigraph(os.path.join(mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                  os.path.join(mig_type_data_dir, f"G_tree{tree}_seed{seed}.predicted.tree"))


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

    grad_m5_df = grad_m5_df.groupby(['seeding pattern','seed']).mean().assign(method="SGD")
    grad_m8_df = grad_m8_df.groupby(['seeding pattern','seed']).mean().assign(method="SGD")

    print("\nSGD m5 avg F1 scores")
    print(grad_m5_df)
    print(grad_m5_df.groupby('seeding pattern').mean())

    print("\nSGD m8 avg F1 scores")
    print(grad_m8_df)
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
    print(machina_m5_df)
    print(machina_m5_df.groupby('seeding pattern').mean())

    print("MACHINA m8 avg F1 scores")
    print(machina_m8_df)
    print(machina_m8_df.groupby('seeding pattern').mean())

    joint_m5_df = pd.concat([grad_m5_df, machina_m5_df]).reset_index()
    joint_m8_df = pd.concat([grad_m8_df, machina_m8_df]).reset_index()
    print(joint_m5_df.reset_index())

    save_boxplot(joint_m5_df, "migration graph F1 score", "m5_migration_graph_f1_scores.png")
    save_boxplot(joint_m8_df, "migration graph F1 score", "m8_migration_graph_f1_scores.png")

    save_boxplot(joint_m5_df, "migrating clones F1 score", "m5_migrating_clones_f1_scores.png")
    save_boxplot(joint_m8_df, "migrating clones F1 score", "m8_migrating_clones_f1_scores.png")

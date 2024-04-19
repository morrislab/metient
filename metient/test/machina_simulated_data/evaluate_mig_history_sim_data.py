#!/usr/bin/python
import sys
import os
import fnmatch
import numpy as np
import re 

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
from metient.util import eval_util as eutil

def save_m5_m8_boxplot(joint_m5_df, joint_m8_df, run_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    ys = ["Migration graph F1 score", "Migrating clones F1 score"]

    seeding_pattern_order = ["mS", "pS", "pM", "pR"]

    i = 0
    for y in ys:
        for num_sites, df in [(5, joint_m5_df), (8, joint_m8_df)]:

            box_pairs = []
            for seeding_pattern in seeding_pattern_order:
                box_pairs.append(((seeding_pattern, "Metient"),(seeding_pattern, "MACHINA")))
            flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='darkgrey')
            pos = str(np.binary_repr(i, width=2))
            ax = axes[int(pos[0]),int(pos[1])]
            sns.boxplot(ax=ax, x="seeding pattern", y=y, hue="method", data=df, order=seeding_pattern_order, 
                        palette={"Metient":"#2496c8ff", "MACHINA":"#ff915fff"}, flierprops=flierprops, linewidth=1.5)
            add_stat_annotation(ax, data=df, x="seeding pattern", y=y, hue="method",
                                box_pairs=box_pairs,
                                test='t-test_paired', text_format='star', loc='inside', verbose=0, order=seeding_pattern_order, fontsize=18, comparisons_correction=None)
            ax.set(ylim=(-0.1, 1.1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Seeding pattern', fontsize=16,)
            ax.set_ylabel(y, fontsize=16,)
            ax.set_title(f"{num_sites} Anatomical Sites", fontsize=16, y=1.1)
            ax.get_legend().remove()

            i += 1

    lines, labels = fig.axes[0].get_legend_handles_labels()
    
    fig.suptitle("F1-scores on Simulated Data",  fontweight='bold')
    fig.legend(lines, labels, ncol=2, loc='upper right', bbox_to_anchor=(.75, 0.98), frameon=False, fontsize=20)
    ax.get_figure().savefig("../output_plots/"+f"m5_m8_f1_scores_{run_name}.png")
    plt.clf()

def save_boxplot(joint_df, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    ys = ["Migration graph F1 score", "Migrating clones F1 score"]
    print(axes)
    seeding_pattern_order = ["mS", "pS", "pM", "pR"]
    for i,y in enumerate(ys):
        box_pairs = []
        for seeding_pattern in seeding_pattern_order:
            box_pairs.append(((seeding_pattern, "Metient"),(seeding_pattern, "MACHINA")))
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='darkgrey')
        
        ax = axes[i]

        snsfig = sns.boxplot(ax=ax, x="seeding pattern", y=y, hue="method", data=joint_df, order=seeding_pattern_order, 
                    palette={"Metient":"#40908e", "MACHINA":"#ff915fff"}, flierprops=flierprops, linewidth=1.5)
        # for i,box in enumerate([p for p in snsfig.patches if not p.get_label()]): 
        #     color = box.get_facecolor()
        #     box.set_edgecolor(color)
        #     box.set_facecolor((0, 0, 0, 0))
        #     # iterate over whiskers and median lines
        #     for j in range(5*i,5*(i+1)):
        #         snsfig.lines[j].set_color(color)

        add_stat_annotation(ax, data=joint_df, x="seeding pattern", y=y, hue="method",
                            box_pairs=box_pairs,
                            test='t-test_welch', text_format='star', loc='inside', verbose=0, order=seeding_pattern_order, fontsize=18, comparisons_correction=None)
        ax.set(ylim=(-0.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Seeding pattern', fontsize=16,)
        ax.set_ylabel(y, fontsize=16,)
        ax.get_legend().remove()

    lines, labels = fig.axes[0].get_legend_handles_labels()
    
    fig.suptitle("F1-scores on Simulated Data",  fontweight='bold')
    fig.legend(lines, labels, ncol=2, loc='upper right', bbox_to_anchor=(.75, 0.98), frameon=False, fontsize=20)
    ax.get_figure().savefig("../output_plots/"+f"all_f1_scores_{run_name}.png")
    plt.clf()

if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <PREDICTIONS_DATA_DIR> <RUN_NAME>\n" % sys.argv[0])
        sys.stderr.write("MACHINA_SIM_DATA_DIR: directory containing the true labelings\n")
        sys.stderr.write("PREDICTIONS_DATA_DIR: directory containing the predicted labelings\n")
        sys.stderr.write("RUN_NAME: Name of the run to use in output png naming.\n")
        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    results = {s : {m : [] for m in mig_types} for s in sites}
    grad_m5_f1_scores = []
    grad_m8_f1_scores = []
    all_f1_scores = []

    mach_sim_data_dir = sys.argv[1]
    predictions_data_dir = sys.argv[2]
    run_name = sys.argv[3]

    x = 0
    k = float("inf")
    loss_thres = 0.0 # floating point approximation leeway
    suffix = "_evaluate"

    print(f"Finding {k} best trees within loss threshold of {loss_thres}")
    print(f"Matching files ending in {suffix}")

    for site in sites:
        for mig_type in mig_types:
            print(site, mig_type)
            true_site_mig_type_data_dir = os.path.join(mach_sim_data_dir, site, mig_type)
            predicted_site_mig_type_data_dir = os.path.join(predictions_data_dir, site, mig_type)
            filenames = fnmatch.filter(os.listdir(predicted_site_mig_type_data_dir), 'T_tree*.predicted.tree')
            seeds = set([int(re.findall(r'seed(\d+)', filename)[0]) for filename in filenames])
            print(seeds)
            assert(len(seeds)==10)
            for seed in seeds:
                print(site, mig_type, seed)
                #trees = [t[t.find("tree")+4:t.find("_seed")] for t in filenames if seed == t[t.find("seed")+4:t.find(".predicted")]]
                tree_info = eutil.get_metient_min_loss_trees(predicted_site_mig_type_data_dir, seed, k, loss_thres=loss_thres, suffix=suffix)
                for loss, results_dict, met_tree_num in tree_info:
                    
                    recall, precision, F, has_resolved_polytomy = eutil.evaluate_seeding_clones(os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.tree"),
                                                                  os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.vertex.labeling"),
                                                                  results_dict, met_tree_num)


                    recall_G, precision_G, F_G = eutil.evaluate_migration_graph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                results_dict, met_tree_num)

                    recall_G2, precision_G2, F_G2 = eutil.evaluate_migration_multigraph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                        results_dict, met_tree_num)

                    scores = [recall, precision, F, recall_G, precision_G, F_G, recall_G2, precision_G2, F_G2]

                    # rename "S", "M", "R" -> "pS", "pM", "pR"
                    mig_name = mig_type if len(mig_type) == 2 else "p"+mig_type
                    if site == 'm5':
                        grad_m5_f1_scores.append([seed, mig_name, F, F_G2])
                    elif site == 'm8':
                        grad_m8_f1_scores.append([seed, mig_name, F,  F_G2])
                    all_f1_scores.append([seed, mig_name, F,  F_G2])

                    x += 1
                    results[site][mig_type].append(scores)

    print("num trees:",  x)

    # Plot results
    grad_m5_df = pd.DataFrame(grad_m5_f1_scores, columns=["seed", "seeding pattern",  "Migrating clones F1 score", "Migration graph F1 score"])
    grad_m8_df = pd.DataFrame(grad_m8_f1_scores, columns=["seed", "seeding pattern",  "Migrating clones F1 score", "Migration graph F1 score"])
    grad_all_df = pd.DataFrame(all_f1_scores, columns=["seed", "seeding pattern",  "Migrating clones F1 score", "Migration graph F1 score"])

    # print(grad_m5_df)
    # print(grad_m8_df)
    # print(grad_all_df)
    grad_m5_df = grad_m5_df.groupby(['seeding pattern','seed']).mean().assign(method="Metient")
    grad_m8_df = grad_m8_df.groupby(['seeding pattern','seed']).mean().assign(method="Metient")
    grad_all_df = grad_all_df.groupby(['seeding pattern','seed']).mean().assign(method="Metient")

    # print("\nGradient-based m5 avg F1 scores")
    # print(grad_m5_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean())

    # print("\nGradient-based m8 avg F1 scores")
    # print(grad_m8_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean())
    print("*"*50)
    print("\nMetient all avg F1 scores")
    order = ["mS", "pS", "pM", "pR"]
    grad_all_df_summary = grad_all_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean().reindex(order)
    print(grad_all_df_summary)

    grad_micro_f1 = grad_all_df[["Migrating clones F1 score", "Migration graph F1 score"]].mean()

    print("Migrating clone scores")
    clone_scores = list(grad_all_df_summary['Migrating clones F1 score'])
    lst = ["{:.3f}".format(x) for x in clone_scores]+["{:.3f}".format(sum(clone_scores)/len(clone_scores))]+["{:.3f}".format(grad_micro_f1["Migrating clones F1 score"].item())]
    print(" & ".join(lst))
    print("Migration graph scores")
    graph_scores = list(grad_all_df_summary['Migration graph F1 score'])
    lst = ["{:.3f}".format(x) for x in graph_scores]+["{:.3f}".format(sum(graph_scores)/len(graph_scores))]+["{:.3f}".format(grad_micro_f1["Migration graph F1 score"].item())]
    print(" & ".join(lst))

    # Load machina results
    machina_m5_df, machina_m8_df = eutil.load_machina_results(".")
    col_mapping = {"FscoreT": "Migrating clones F1 score", "FscoreMultiG": "Migration graph F1 score", "pattern": "seeding pattern", "seed":"seed"}
    machina_m5_df = machina_m5_df.rename(columns=col_mapping)
    machina_m8_df = machina_m8_df.rename(columns=col_mapping)

    machina_m5_df = machina_m5_df[col_mapping.values()]
    machina_m8_df = machina_m8_df[col_mapping.values()]
    machina_all_df = pd.concat([machina_m5_df, machina_m8_df], axis=0)
    machina_m5_df = machina_m5_df.groupby(['seeding pattern','seed']).mean().assign(method="MACHINA")
    machina_m8_df = machina_m8_df.groupby(['seeding pattern','seed']).mean().assign(method="MACHINA")
    machina_all_df = machina_all_df.groupby(['seeding pattern','seed']).mean().assign(method="MACHINA")

    # print("MACHINA m5 avg F1 scores")
    # print(machina_m5_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean())

    # print("MACHINA m8 avg F1 scores")
    # print(machina_m8_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean())
    
    print("MACHINA all avg F1 scores")
    print(machina_all_df.groupby('seeding pattern')[["Migrating clones F1 score", "Migration graph F1 score"]].mean())
    
    mach_micro_f1 = machina_all_df[["Migrating clones F1 score", "Migration graph F1 score"]].mean()
    print("Micro-F1", mach_micro_f1)

    joint_m5_df = pd.concat([grad_m5_df, machina_m5_df]).reset_index()
    joint_m8_df = pd.concat([grad_m8_df, machina_m8_df]).reset_index()
    joint_all_df = pd.concat([grad_all_df, machina_all_df]).reset_index()

    save_m5_m8_boxplot(joint_m5_df, joint_m8_df, run_name)
    save_boxplot(joint_all_df, run_name)

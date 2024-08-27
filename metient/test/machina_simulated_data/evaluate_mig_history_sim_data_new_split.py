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

from scipy.stats import shapiro, normaltest, anderson
import scipy.stats as stats

# xaxis_labels = ['monoclonal\nprimary-only', 'polyclonal\nprimary-only', 'monoclonal\nmet-to-met', 'polyclonal\nmet-to-met']
xaxis_labels = ['Primary-only', 'Met-to-met']
seeding_pattern_order = ["prim_only", "met_to_met"]
mig_clones_f1_mach_key = "Seeding clones F1 score"
founding_clones_f1_met_key = "Founding clones F1 score (Metient def.)"
mig_graph_f1_key = "Migration graph F1 score"

def save_boxplot(joint_df, run_name, ys):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), dpi=200)
    print(axes)

    for i,y in enumerate(ys):
        box_pairs = []
        for seeding_pattern in seeding_pattern_order:
            box_pairs.append(((seeding_pattern, "Metient"),(seeding_pattern, "MACHINA")))
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='darkgrey')
        
        ax = axes[i]

        snsfig = sns.boxplot(ax=ax, x="seeding pattern", y=y, hue="method", data=joint_df, order=seeding_pattern_order, 
                    palette={"Metient":"#40908e", "MACHINA":"#ff915fff"}, flierprops=flierprops, linewidth=1.5)


        add_stat_annotation(ax, data=joint_df, x="seeding pattern", y=y, hue="method",
                            box_pairs=box_pairs,test='Wilcoxon', text_format='star', loc='inside', 
                            verbose=2, order=seeding_pattern_order, fontsize=18, comparisons_correction=None)
        ax.set(ylim=(-0.1, 1.1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Seeding pattern', fontsize=16,)
        ax.set_ylabel(y, fontsize=16,)
        ax.get_legend().remove()
        ax.set_xticklabels(xaxis_labels)
        ax.tick_params(axis='both', which='major', labelsize=13)


    lines, labels = fig.axes[0].get_legend_handles_labels()
    
    fig.suptitle("F1-scores on Simulated Data",  fontweight='bold')
    fig.legend(lines, labels, ncol=2, loc='upper right', bbox_to_anchor=(.75, 0.98), frameon=False, fontsize=20)
    ax.get_figure().savefig("../output_plots/"+f"all_f1_scores_{run_name}.png")
    plt.clf()

def test_normality(df,metrics):
    for metric in metrics:
        for pattern in df['seeding pattern'].unique():
            f1_scores = list(df[df['seeding pattern']==pattern][metric])
            shapiro_test = shapiro(f1_scores)
            print(f"Shapiro-Wilk Test: p-value={shapiro_test.pvalue}", "is normal:",shapiro_test.pvalue>0.05)

def collect_metient_results(all_f1_scores, cols):
    # Create dataframe
    grad_all_df = pd.DataFrame(all_f1_scores, columns=cols)
    grad_all_df = grad_all_df.groupby(['seed', 'site', 'mig_type',"seeding pattern"]).mean().assign(method="Metient")

    # Print a summary of the results
    print("*"*50)
    print("\nMetient all avg F1 scores")
    grad_all_df_summary = grad_all_df.groupby('seeding pattern')[[mig_clones_f1_mach_key,founding_clones_f1_met_key,mig_graph_f1_key]].mean().reindex(seeding_pattern_order)

    grad_micro_f1 = grad_all_df[[mig_clones_f1_mach_key,founding_clones_f1_met_key,mig_graph_f1_key]].mean()

    print("Seeding clone scores (MACH def.)")
    clone_scores = list(grad_all_df_summary[mig_clones_f1_mach_key])
    lst = ["{:.3f}".format(x) for x in clone_scores]+["{:.3f}".format(sum(clone_scores)/len(clone_scores))]+["{:.3f}".format(grad_micro_f1[mig_clones_f1_mach_key].item())]
    print(" & ".join(lst))
    print("Colonizing clone scores (Metient def.)")
    clone_scores = list(grad_all_df_summary[founding_clones_f1_met_key])
    lst = ["{:.3f}".format(x) for x in clone_scores]+["{:.3f}".format(sum(clone_scores)/len(clone_scores))]+["{:.3f}".format(grad_micro_f1[founding_clones_f1_met_key].item())]
    print(" & ".join(lst))
    print("Migration graph scores")
    graph_scores = list(grad_all_df_summary[mig_graph_f1_key])
    lst = ["{:.3f}".format(x) for x in graph_scores]+["{:.3f}".format(sum(graph_scores)/len(graph_scores))]+["{:.3f}".format(grad_micro_f1[mig_graph_f1_key].item())]
    print(" & ".join(lst))
    return grad_all_df.reset_index()

def collect_machina_results():
    machina_m5_df, machina_m8_df = eutil.load_machina_results_new_split(".")
    col_mapping = {"FscoreT": "Seeding clones F1 score", "FscoreMultiG": "Migration graph F1 score", "new_gt_pattern": "seeding pattern", "seed":"seed", 'pattern':'mig_type'}
    machina_m5_df = machina_m5_df.rename(columns=col_mapping)
    machina_m8_df = machina_m8_df.rename(columns=col_mapping)

    machina_m5_df = machina_m5_df[col_mapping.values()]
    machina_m5_df['site'] = 'm5'
    machina_m8_df = machina_m8_df[col_mapping.values()]
    machina_m8_df['site'] = 'm8'
    machina_all_df = pd.concat([machina_m5_df, machina_m8_df], axis=0)
    machina_all_df = machina_all_df.groupby(['seed', 'site', 'mig_type',"seeding pattern"]).mean().assign(method="MACHINA")

    # Print a summary of the results
    print("\nMACHINA all avg F1 scores")
    machina_all_df_summary = machina_all_df.groupby('seeding pattern')[["Seeding clones F1 score", "Migration graph F1 score"]].mean().reindex(seeding_pattern_order)

    mach_micro_f1 = machina_all_df[["Seeding clones F1 score", "Migration graph F1 score"]].mean()
    print("Seeding clone scores")
    clone_scores = list(machina_all_df_summary['Seeding clones F1 score'])
    lst = ["{:.3f}".format(x) for x in clone_scores]+["{:.3f}".format(sum(clone_scores)/len(clone_scores))]+["{:.3f}".format(mach_micro_f1["Seeding clones F1 score"].item())]
    print(" & ".join(lst))
    print("Migration graph scores")
    graph_scores = list(machina_all_df_summary['Migration graph F1 score'])
    lst = ["{:.3f}".format(x) for x in graph_scores]+["{:.3f}".format(sum(graph_scores)/len(graph_scores))]+["{:.3f}".format(mach_micro_f1["Migration graph F1 score"].item())]
    print(" & ".join(lst))
    return machina_all_df.reset_index()


if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <PREDICTIONS_DATA_DIR> <RUN_NAME>\n" % sys.argv[0])
        sys.stderr.write("MACHINA_SIM_DATA_DIR: directory containing the true labelings\n")
        sys.stderr.write("PREDICTIONS_DATA_DIR: directory containing the predicted labelings\n")
        sys.stderr.write("RUN_NAME: Name of the run to use in output png naming.\n")
        sys.stderr.write("SUFFIX: suffix to match pkl.gz files\n") # e.g. _calibrate
        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    results = {s : {m : [] for m in mig_types} for s in sites}

    mach_sim_data_dir = sys.argv[1]
    predictions_data_dir = sys.argv[2]
    run_name = sys.argv[3]

    x = 0
    k = float("inf")
    loss_thres = 0.0
    suffix = sys.argv[4]

    print(f"Finding {k} best trees within loss threshold of {loss_thres}")
    print(f"Matching files ending in {suffix}")

    gt_df = pd.read_csv("/data/morrisq/divyak/projects/metient/metient/data/machina_sims/gt_pattern.csv")
    gt_df['site'] = gt_df['site'].astype(str)
    gt_df['mig_type'] = gt_df['mig_type'].astype(str)
    gt_df['seed'] = gt_df['seed'].astype(int)
    all_f1_scores = []

    for site in sites:
        for mig_type in mig_types:
            print(site, mig_type)
            true_site_mig_type_data_dir = os.path.join(mach_sim_data_dir, site, mig_type)
            predicted_site_mig_type_data_dir = os.path.join(predictions_data_dir, site, mig_type)
            filenames = fnmatch.filter(os.listdir(predicted_site_mig_type_data_dir), 'T_tree*.predicted.tree')
            seeds = set([int(re.findall(r'seed(\d+)', filename)[0]) for filename in filenames])
            assert(len(seeds)==10)
            for seed in seeds:
                print(site, mig_type, seed)
                gt_pattern = gt_df[(gt_df['site']==site)&(gt_df['mig_type']==mig_type)&(gt_df['seed']==seed)]['gt_pattern'].item()
                tree_info = eutil.get_metient_min_loss_trees(predicted_site_mig_type_data_dir, seed, k, loss_thres=loss_thres, suffix=suffix)
                for loss, results_dict, met_tree_num, clone_tree_num in tree_info:
                    recall_sc_mach, precision_sc_mach, F_sc_mach = eutil.evaluate_seeding_clones_mach(os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.tree"),
                                                                                                      os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.vertex.labeling"),
                                                                                                      results_dict, met_tree_num)
                    recall_sc_met, precision_sc_met, F_sc_met = eutil.evaluate_seeding_clones_met(os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.tree"),
                                                                                                  os.path.join(true_site_mig_type_data_dir, f"T_seed{seed}.vertex.labeling"),
                                                                                                  results_dict, met_tree_num)

                    recall_G, precision_G, F_G = eutil.evaluate_migration_graph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                results_dict, met_tree_num)

                    recall_G2, precision_G2, F_G2 = eutil.evaluate_migration_multigraph(os.path.join(true_site_mig_type_data_dir, f"G_seed{seed}.tree"),
                                                                                        results_dict, met_tree_num)

                    scores = [recall_sc_mach, precision_sc_mach, F_sc_mach, recall_sc_met, precision_sc_met, F_sc_met, recall_G, precision_G, F_G, recall_G2, precision_G2, F_G2]
                    
                    all_f1_scores.append([seed, site, mig_type, gt_pattern, F_sc_mach, F_sc_met, F_G2])

                    x += 1
                    results[site][mig_type].append(scores)

    # Plot results
    
    cols = ["seed", "site", "mig_type", "seeding pattern", mig_clones_f1_mach_key,founding_clones_f1_met_key,mig_graph_f1_key]
    
    # Load Metient results

    metient_result_df = collect_metient_results(all_f1_scores, cols)

    # Load machina results
    machina_result_df = collect_machina_results()
   
    joint_all_df = pd.concat([metient_result_df, machina_result_df]).reset_index()

    # test_normality(metient_result_df, ["Seeding clones F1 score", "Migration graph F1 score"])
    # test_normality(machina_result_df, ["Seeding clones F1 score", "Migration graph F1 score"])

    save_boxplot(joint_all_df, run_name, [mig_clones_f1_mach_key, mig_graph_f1_key])

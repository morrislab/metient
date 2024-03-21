import sys
import os
import torch
import matplotlib
import argparse
import datetime
import fnmatch
import pandas as pd
from metient.metient import *
from metient.lib.vertex_labeling import get_migration_history
from metient.util import plotting_util as plot_util
from metient.util.globals import *

def get_index_to_cluster_label_from_corrected_sim_tsv(ref_var_fn):
    df = pd.read_csv(ref_var_fn, sep="\t")
    clstr_idx_to_label = {}
    labels = df['character_label'].unique()
    for label in labels:
        idx = int(df[df['character_label']==label]['cluster_index'].unique().item())
        if idx not in clstr_idx_to_label:
            clstr_idx_to_label[idx] = []
        clstr_idx_to_label[idx].append(str(label))
    clstr_idx_to_label = {k:";".join(v) for k,v in clstr_idx_to_label.items()}
    return clstr_idx_to_label

def predict_vertex_labeling(machina_sims_data_dir, site, mig_type, seed, out_dir, weights, batch_size, weight_init_primary, perf_stats_fn, mode, solve_polytomies):
    print("solve_polytomies", solve_polytomies)
    print("wip", wip)
    print("batch_size", batch_size)
    print("mode", mode)
    cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
    all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
    trees = fnmatch.filter(os.listdir(os.path.join(machina_sims_data_dir, f"{site}_clustered_input_corrected")), f"cluster_{mig_type}_seed{seed}_tree*.tsv")

    idx_to_cluster_label = get_idx_to_cluster_label(cluster_fn, ignore_polytomies=True)
    data = get_adj_matrices_from_spruce_mutation_trees(all_mut_trees_fn, idx_to_cluster_label, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    perf_stats = []
    for tree_num in range(len(trees)):
        start_time = datetime.datetime.now()

        ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input_corrected", f"cluster_{mig_type}_seed{seed}_tree{tree_num}.tsv")
        corrected_idx_to_cluster_label = get_index_to_cluster_label_from_corrected_sim_tsv(ref_var_fn)
        data = dutil.get_adj_matrices_from_spruce_mutation_trees(all_mut_trees_fn, idx_to_cluster_label, is_sim_data=True)
        assert(data[tree_num][1] == corrected_idx_to_cluster_label)

        T = torch.tensor(data[tree_num][0], dtype = torch.float32)

        print_config = PrintConfig(visualize=False, verbose=False, k_best_trees=batch_size, save_outputs=True)

        T_edges, labeling, G_edges, loss_info, time = get_migration_history(T,  ref_var_fn, 'P', weights, print_config, out_dir, f"tree{tree_num}_seed{seed}_{mode}", 
                                                                            max_iter=100, batch_size=batch_size, custom_colors=custom_colors, 
                                                                            bias_weights=weight_init_primary, mode=mode, solve_polytomies=solve_polytomies)

        time_with_plotting = (datetime.datetime.now() - start_time).total_seconds()
        
        perf_stats.append([site, mig_type, seed, tree_num, time, time_with_plotting])

        plot_util.write_tree(T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}_{mode}.predicted.tree"), add_germline_node=True)
        plot_util.write_tree_vertex_labeling(labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}_{mode}.predicted.vertex.labeling"), add_germline_node=True)
        plot_util.write_migration_graph(G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}_{mode}.predicted.tree"))

    with open(os.path.join(perf_stats_fn, f"perf_stats_{site}_{mig_type}_{seed}.txt"), 'w') as f:
        for line in perf_stats:
            f.write("\t".join([str(stat) for stat in line]))
            f.write("\n")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
    parser.add_argument('--num_sites', type=str, help="m5 or m8")
    parser.add_argument('--mig_type', type=str, help="M, mS, R or S")
    parser.add_argument('--seed', type=int, help="tree seed")

    parser.add_argument('--wdata_fit', type=float, help="Weight on data fit")
    parser.add_argument('--wmig', type=float, help="Weight on migration number")
    parser.add_argument('--wcomig', type=float, help="Weight on comig")
    parser.add_argument('--wseed', type=float, help="Weight on seeding site number")
    parser.add_argument('--wreg', type=float, help="Weight on regularization")
    parser.add_argument('--wgen', type=float, help="Weight on genetic distance")

    parser.add_argument('--wip', type=str, help="Initialize weights higher to favor vertex labeling of primary for all internal nodes")
    parser.add_argument('--mode', type=str, help="calibrate or evaluate")
    parser.add_argument('--solve_polys', type=str, help="Whether or not to solve polytomies")
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--out_dir', type=str, help="Where to save this trees output.")
    
    args = parser.parse_args()
    # # Use the 'w' mode to create the file if it doesn't exist
    perf_stats_fn = os.path.join(args.out_dir).replace(args.num_sites, "").replace(args.mig_type, "")
    
    sys.stdout = open(os.path.join(args.out_dir, f"output.txt").replace(args.num_sites, "").replace(args.mig_type, ""), 'a')
    
    if args.mode == 'calibrate':
        weights = Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                     seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, gen_dist=args.wgen)

    elif args.mode == 'evaluate':
        weights = Weights(data_fit=args.wdata_fit, mig=[args.wmig], comig=args.wcomig, seed_site=[args.wseed], reg=args.wreg, gen_dist=args.wgen)
    
    wip = True if args.wip == "True" else False
    solve_polytomies = True if args.solve_polys == "True" else False

    print()
    print("Predicting for", args.num_sites, args.mig_type, args.seed)
    print(f"Weights:\nmig: {weights.mig}, comig: {weights.comig}, seeding: {weights.seed_site}, gen: {weights.gen_dist}")

    predict_vertex_labeling(args.sim_data_dir, args.num_sites, args.mig_type, args.seed, args.out_dir, weights, args.bs, wip, perf_stats_fn, args.mode, solve_polytomies)

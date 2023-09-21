import sys
import os
import torch
import matplotlib
import argparse

from src.lib.metient import *
from src.util import plotting_util as plot_util

def predict_vertex_labeling(machina_sims_data_dir, site, mig_type, seed, out_dir, weights, batch_size, weight_init_primary, lr_sched):
    cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
    all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
    ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.tsv")

    idx_to_cluster_label = get_idx_to_cluster_label(cluster_fn, ignore_polytomies=True)
    data = get_adj_matrices_from_spruce_mutation_trees(all_mut_trees_fn, idx_to_cluster_label, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]

    for tree_num, (adj_matrix, pruned_idx_to_label) in enumerate(data):
        print(f"Tree {tree_num}")
        T = torch.tensor(adj_matrix, dtype = torch.float32)

        ref_matrix, var_matrix, unique_sites= get_ref_var_matrices_from_machina_sim_data(ref_var_fn, pruned_idx_to_label, T)

        G = get_genetic_distance_matrix_from_adj_matrix(T, pruned_idx_to_label, ";")
        print_config = PrintConfig(visualize=False, verbose=False, viz_intermeds=False, k_best_trees=10)
        T_edges, labeling, G_edges, loss_info, time = get_migration_history(T, ref_matrix, var_matrix, unique_sites, 'P', pruned_idx_to_label,
                                                                            weights, print_config, out_dir, f"tree{tree_num}_seed{seed}", 
                                                                            G=G, max_iter=100, batch_size=batch_size, custom_colors=custom_colors, 
                                                                            weight_init_primary=weight_init_primary, lr_sched=lr_sched)

        plot_util.write_tree(T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"), add_germline_node=True)
        plot_util.write_tree_vertex_labeling(labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"), add_germline_node=True)
        plot_util.write_migration_graph(G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
    parser.add_argument('--num_sites', type=str, help="m5 or m8")
    parser.add_argument('--mig_type', type=str, help="M, mS, R or S")
    parser.add_argument('--seed', type=int, help="tree seed")

    parser.add_argument('--wdata_fit', type=float, help="Weight on data fit", default=0.2)
    parser.add_argument('--wmig', type=float, help="Weight on migration number", default=10.0)
    parser.add_argument('--wcomig', type=float, help="Weight on comigration number", default=7.0)
    parser.add_argument('--wseed', type=float, help="Weight on seeding site number", default=5.0)
    parser.add_argument('--wreg', type=float, help="Weight on regularization", default=1.0)
    parser.add_argument('--wgen', type=float, help="Weight on genetic distance", default=0.0)

    parser.add_argument('--wip', type=bool, help="Initialize weights higher to favor vertex labeling of primary for all internal nodes", default=False)
    parser.add_argument('--bs', type=int, help="Batch size", default=32)
    parser.add_argument('--lr_sched', type=str, help="Learning rate schedule. See vertex_labeling.gumbel_softmax_optimization for details", default="step")
    parser.add_argument('--out_dir', type=str, help="Where to save this trees output.")
    
    args = parser.parse_args()
    
    weights = Weights(data_fit=args.wdata_fit, mig=args.wmig, comig=args.wcomig, seed_site=args.wseed, reg=args.wreg, gen_dist=args.wgen)

    predict_vertex_labeling(args.sim_data_dir, args.num_sites, args.mig_type, args.seed, args.out_dir, weights, args.bs, args.wip, args.lr_sched)
                

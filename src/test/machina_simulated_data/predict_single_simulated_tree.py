import sys
import os
import torch
import matplotlib
import argparse

from src.lib import vertex_labeling
import src.util.data_extraction_util as data_util
import src.util.vertex_labeling_util as vert_util
from src.util import plotting_util as plot_util

def predict_vertex_labeling(machina_sims_data_dir, site, mig_type, seed, out_dir, weights, batch_size, weight_init_primary, lr_sched):
    cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
    all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
    ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.tsv")

    cluster_label_to_idx = data_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies=True)
    
    data = data_util.get_adj_matrices_from_all_mutation_trees(all_mut_trees_fn, cluster_label_to_idx, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    tree_num = 0

    for adj_matrix, pruned_cluster_label_to_idx in data:
        print(f"Tree {tree_num}")
        T = torch.tensor(adj_matrix, dtype = torch.float32)
        idx_to_label = {v:k for k,v in pruned_cluster_label_to_idx.items()}

        ref_matrix, var_matrix, unique_sites= data_util.get_ref_var_matrices_from_machina_sim_data(ref_var_fn,
                                                                                                   pruned_cluster_label_to_idx=pruned_cluster_label_to_idx,
                                                                                                   T=T)

        primary_idx = unique_sites.index('P')
        r = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
        G = data_util.get_genetic_distance_tensor_from_adj_matrix(T, idx_to_label, ";")
        print_config = plot_util.PrintConfig(visualize=False, verbose=False, viz_intermeds=False, k_best_trees=5)
        T_edges, labeling, G_edges, loss_info, time = vertex_labeling.get_migration_history(T, ref_matrix, var_matrix, unique_sites, r, idx_to_label,
                                                                                            weights, print_config, out_dir, f"tree{tree_num}_seed{seed}", 
                                                                                            G=G, max_iter=100, batch_size=batch_size, custom_colors=custom_colors, 
                                                                                            weight_init_primary=weight_init_primary, lr_sched=lr_sched)

        plot_util.write_tree(T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"), add_germline_node=True)
        plot_util.write_tree_vertex_labeling(labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"), add_germline_node=True)
        plot_util.write_migration_graph(G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))
        tree_num += 1

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
    
    weights = vertex_labeling.Weights(data_fit=args.wdata_fit, mig=args.wmig, comig=args.wcomig, seed_site=args.wseed, reg=args.wreg, gen_dist=args.wgen)

    predict_vertex_labeling(args.sim_data_dir, args.num_sites, args.mig_type, args.seed, args.out_dir, weights, args.bs, args.wip, args.lr_sched)
                

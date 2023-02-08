import sys
import os
import fnmatch
import torch
import matplotlib
import argparse
import datetime
import concurrent.futures
import multiprocessing
import pandas as pd

from src.lib import vertex_labeling
import src.util.data_extraction_util as data_util
import src.util.vertex_labeling_util as vert_util

import cProfile
results = []

def predict_vertex_labelings(machina_sims_data_dir, site, mig_type, seed, out_dir):
    cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
    all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
    ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.tsv")

    cluster_label_to_idx = data_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies=True)
    
    data = data_util.get_adj_matrices_from_all_mutation_trees(all_mut_trees_fn, cluster_label_to_idx, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    tree_num = 0

    for adj_matrix, pruned_cluster_label_to_idx in data:
        #print(f"Tree {tree_num}")
        T = torch.tensor(adj_matrix, dtype = torch.float32)
        B = vert_util.get_mutation_matrix_tensor(T)
        idx_to_label = {v:k for k,v in pruned_cluster_label_to_idx.items()}

        ref_matrix, var_matrix, unique_sites= data_util.get_ref_var_matrices_from_machina_sim_data(ref_var_fn,
                                                                                                   pruned_cluster_label_to_idx=pruned_cluster_label_to_idx,
                                                                                                   T=T)


        primary_idx = unique_sites.index('P')
        r = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
        # TODO: add these as args
        weights = vertex_labeling.Weights(data_fit=1.0, mig=10.0, comig=5.0, seed_site=1.0, reg=1.0, gen_dist=0.5)
        G = data_util.get_genetic_distance_tensor_from_sim_adj_matrix(T, pruned_cluster_label_to_idx)

        best_T_edges, best_labeling, best_G_edges, best_loss_info, time = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites=unique_sites,
                                                                                                weights=weights, p=r, node_idx_to_label=idx_to_label, G=G,
                                                                                                max_iter=150, batch_size=32, init_temp=30, final_temp=0.01,
                                                                                                custom_colors=custom_colors, visualize=False, verbose=False)

        vert_util.write_tree(best_T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"))
        vert_util.write_tree_vertex_labeling(best_labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"))
        vert_util.write_migration_graph(best_G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))
        tree_num += 1
        tree_info = {**{"site": site, "mig_type": mig_type, "seed":seed, "tree_num": tree_num, "time": time}, **best_loss_info}
        global results
        results.append(tree_info)
        print("results length", len(results))

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
    parser.add_argument('run_name', type=str, help="Name of this run")
    parser.add_argument('--cores', '-c', type=int, default=1, help="Number of cores to use (default 1)")
    args = parser.parse_args()

    site = "m8"
    mig_type = "M"
    seeds = ['76', '7']

    machina_sims_data_dir = args.sim_data_dir
    run_name = args.run_name

    predictions_dir = f"predictions_{run_name}"
    os.mkdir(predictions_dir)

    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print(f"Using {args.cores} cores.")
 
    os.mkdir(os.path.join(predictions_dir, site))

    out_dir = os.path.join(predictions_dir, site, mig_type)
    os.mkdir(out_dir)
    site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)

    print(seeds)
    for seed in seeds:
        #predict_vertex_labelings(machina_sims_data_dir, site, mig_type, seed, out_dir)
        # Are we IO bound or CPU bound? maybe we should use a thread pool...?
        cProfile.run("predict_vertex_labelings(machina_sims_data_dir, site, mig_type, seed, out_dir)", sort="tottime")
    end_time = datetime.datetime.now()

    results_df = pd.DataFrame(list(results))
    print(results_df.head())
    results_df.to_csv(os.path.join(predictions_dir, f"results_{run_name}.txt"), sep=',', index=False)
 
    print(f"Finished running {len(results)} simulations.")
    print(f"End time: {end_time}. Time elapsed: {end_time - start_time}")

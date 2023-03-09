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
from pprint import pprint

from src.lib import vertex_labeling
import src.util.data_extraction_util as data_util
import src.util.vertex_labeling_util as vert_util

results = []


def predict_vertex_labelings(machina_sims_data_dir, site, mig_type, seed, out_dir, weights, batch_size, weight_init_primary):
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
        
        G = data_util.get_genetic_distance_tensor_from_sim_adj_matrix(T, pruned_cluster_label_to_idx)
        print_config = vertex_labeling.PrintConfig(visualize=False, verbose=False, viz_intermeds=False)

        T_edges, labeling, G_edges, loss_info, time = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, 
                                                                                                 ordered_sites=unique_sites,
                                                                                                 weights=weights, p=r, node_idx_to_label=idx_to_label, 
                                                                                                 G=G, max_iter=150, batch_size=batch_size, 
                                                                                                 custom_colors=custom_colors, print_config=print_config, 
                                                                                                 weight_init_primary=weight_init_primary)

        vert_util.write_tree(T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"))
        vert_util.write_tree_vertex_labeling(labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"))
        vert_util.write_migration_graph(G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))
        tree_num += 1
        tree_info = {**{"site": site, "mig_type": mig_type, "seed":seed, "tree_num": tree_num, "time": time}, **loss_info}
        global results
        results.append(tree_info)
            
        print("Number of seeds run:", len(results))

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
    parser.add_argument('run_name', type=str, help="Name of this run")

    parser.add_argument('--data_fit', type=float, help="Weight on data fit", default=1.0)
    parser.add_argument('--mig', type=float, help="Weight on migration number", default=1.0)
    parser.add_argument('--comig', type=float, help="Weight on comigration number", default=1.0)
    parser.add_argument('--seed', type=float, help="Weight on seeding site number", default=1.0)
    parser.add_argument('--reg', type=float, help="Weight on regularization", default=1.0)
    parser.add_argument('--gen', type=float, help="Weight on genetic distance", default=0.0)

    parser.add_argument('--primary_weight', action='store_true', help="If passed, initialize weights higher to favor vertex labeling of primary for all internal nodes", default=False)
    parser.add_argument('--bs', type=int, help="Batch size", default=32)
    parser.add_argument('--cores', '-c', type=int, default=1, help="Number of cores to use (default 1)")
    args = parser.parse_args()

    machina_sims_data_dir = args.sim_data_dir
    run_name = args.run_name

    predictions_dir = f"predictions_{run_name}"
    os.mkdir(predictions_dir)

    sys.stdout = open(os.path.join(predictions_dir, f"output__{run_name}.txt"), 'w')

    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]
    
    weights = vertex_labeling.Weights(data_fit=args.data_fit, mig=args.mig, comig=args.comig, seed_site=args.seed, reg=args.reg, gen_dist=args.gen)
    print("Weights:")
    pprint(vars(weights))
    batch_size = args.bs
    print(f"Batch size: {batch_size}")
    print(f"Placing higher weight on primary vertex labeling for all internal nodes: {args.primary_weight}")

    

    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print(f"Using {args.cores} cores.")
    executor = concurrent.futures.ThreadPoolExecutor(args.cores)
    #manager = multiprocessing.Manager()
    #results = manager.list()
    futures = []
    for site in sites:
        os.mkdir(os.path.join(predictions_dir, site))

        for mig_type in mig_types:
            out_dir = os.path.join(predictions_dir, site, mig_type)
            os.mkdir(out_dir)
            print(out_dir)
            site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)

            seeds = fnmatch.filter(os.listdir(site_mig_data_dir), 'reads_seed*.tsv')
            seeds = [s.replace(".tsv", "").replace("reads_seed", "") for s in seeds]
            print(seeds)
            for seed in seeds:
                #predict_vertex_labelings(machina_sims_data_dir, site, mig_type, seed, out_dir)
                # Are we IO bound or CPU bound? maybe we should use a thread pool...?
                futures.append(executor.submit(predict_vertex_labelings, machina_sims_data_dir, site, mig_type, seed, out_dir, weights, batch_size, args.primary_weight))
    print("Number of tasks:", len(futures))
    concurrent.futures.wait(futures)
    end_time = datetime.datetime.now()

    results_df = pd.DataFrame(list(results))
    print(results_df.head())
    results_df.to_csv(os.path.join(predictions_dir, f"results_{run_name}.txt"), sep=',', index=False)
 
    print(f"Finished running {len(results)} simulations.")
    print(f"End time: {end_time}. Time elapsed: {end_time - start_time}")

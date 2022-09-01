import sys
import os
import fnmatch
import torch
import matplotlib

from src.lib import vertex_labeling
import src.util.machina_data_extraction_util as mach_util
import src.util.vertex_labeling_util as vert_util

num_sims_runs = 0


def predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, out_dir):
    cluster_label_to_idx = mach_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies=True)
    global num_sims_runs

    data = mach_util.get_adj_matrices_from_all_mutation_trees(all_mut_trees_fn, cluster_label_to_idx, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    tree_num = 0
    for adj_matrix, pruned_cluster_label_to_idx in data:
        print(f"n: {num_sims_runs}")
        print(f"Tree {tree_num}")
        T = torch.tensor(adj_matrix, dtype = torch.float32)
        B = vert_util.get_mutation_matrix_tensor(T)
        idx_to_label = {v:k for k,v in pruned_cluster_label_to_idx.items()}

        ref_matrix, var_matrix, unique_sites= mach_util.get_ref_var_matrices_from_machina_sim_data(ref_var_fn,
                                                                                                   pruned_cluster_label_to_idx=pruned_cluster_label_to_idx,
                                                                                                   T=T)


        print(unique_sites)
        primary_idx = unique_sites.index('P')
        r = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T

        weights = vertex_labeling.Weights(data_fit=0.8, mig=10.0, comig=5.0, seed_site=1.0, l1=1.0, gen_dist=0.5)
        G = mach_util.get_genetic_distance_tensor_from_sim_adj_matrix(T, pruned_cluster_label_to_idx)

        best_T_edges, best_labeling, best_G_edges = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites=unique_sites,
                                                                                                weights=weights, p=r, node_idx_to_label=idx_to_label, G=G,
                                                                                                max_iter=150, batch_size=64,
                                                                                                custom_colors=custom_colors, visualize=False)

        vert_util.write_tree(best_T_edges, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.tree"))
        vert_util.write_tree_vertex_labeling(best_labeling, os.path.join(out_dir, f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"))
        vert_util.write_migration_graph(best_G_edges, os.path.join(out_dir, f"G_tree{tree_num}_seed{seed}.predicted.tree"))
        num_sims_runs += 1
        tree_num += 1

if __name__=="__main__":

    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <run_name>\n" % sys.argv[0])
        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    machina_sims_data_dir = sys.argv[1]
    run_name = sys.argv[2]

    predictions_dir = f"predictions_{run_name}"
    os.mkdir(predictions_dir)

    for site in sites:
        os.mkdir(os.path.join(predictions_dir, site))

        for mig_type in mig_types:
            out_dir = os.path.join(predictions_dir, site, mig_type)
            os.mkdir(out_dir)
            print(out_dir)
            site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)

            seeds = fnmatch.filter(os.listdir(site_mig_data_dir), 'seed*_0.95.tsv')
            seeds = [s.replace("_0.95.tsv", "").replace("seed", "") for s in seeds]

            for seed in seeds:

                print("="*150)
                print(f"Predicting vertex labeling for {site} {mig_type} seed {seed}.")

                cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
                all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
                ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.tsv")

                predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, out_dir)

                num_sims_runs += 1

    print(f"{num_sims_runs} simulations run")

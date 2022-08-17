import sys
import os
import fnmatch
import torch
import matplotlib

from src.lib import vertex_labeling
import src.util.machina_data_extraction_util as mach_util
import src.util.vertex_labeling_util as vert_util

num_sims_runs = 0

def predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, data_dir, site):
    cluster_label_to_idx = mach_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies=True)
    global num_sims_runs

    data = mach_util.get_adj_matrices_from_all_mutation_trees(all_mut_trees_fn, cluster_label_to_idx)
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

        best_T_edges, best_labeling, best_G_edges = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites=unique_sites,
                                                                                                p=r, node_idx_to_label=idx_to_label,
                                                                                                w_e=0.01, w_l=0.8, w_m=10, max_iter=150, batch_size=64,
                                                                                                custom_colors=custom_colors, visualize=False)

        vert_util.write_tree(best_T_edges, os.path.join(data_dir, f"{site}_predictions", f"T_tree{tree_num}_seed{seed}.predicted.tree"))
        vert_util.write_tree_vertex_labeling(best_labeling, os.path.join(data_dir, f"{site}_predictions", f"T_tree{tree_num}_seed{seed}.predicted.vertex.labeling"))
        vert_util.write_migration_graph(best_G_edges, os.path.join(data_dir, f"{site}_predictions", f"G_tree{tree_num}_seed{seed}.predicted.tree"))
        num_sims_runs += 1
        tree_num += 1

if __name__=="__main__":

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR>\n" % sys.argv[0])
        sys.exit(1)



    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    MACHINA_DATA_DIR = sys.argv[1]

    for site in sites:
        for mig_type in mig_types:
            SIM_DATA_DIR = os.path.join(MACHINA_DATA_DIR, site, mig_type)
            seeds = fnmatch.filter(os.listdir(SIM_DATA_DIR), 'seed*_0.95.tsv')
            seeds = [s.replace("_0.95.tsv", "").replace("seed", "") for s in seeds]

            for seed in seeds:

                print("="*150)
                print(f"Predicting vertex labeling for {site} {mig_type} seed {seed}.")

                cluster_fn = os.path.join(MACHINA_DATA_DIR, f"clustered_input_{site}", f"cluster_{mig_type}_seed{seed}.txt")
                all_mut_trees_fn = os.path.join(MACHINA_DATA_DIR, f"mut_trees_{site}", f"mut_trees_{mig_type}_seed{seed}.txt")
                ref_var_fn = os.path.join(MACHINA_DATA_DIR, f"clustered_input_{site}", f"cluster_{mig_type}_seed{seed}.tsv")
                predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, MACHINA_DATA_DIR, site)

                num_sims_runs += 1

    print(f"{num_sims_runs} simulations run")

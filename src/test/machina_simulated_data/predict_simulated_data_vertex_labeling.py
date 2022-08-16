import sys
import os
import fnmatch
import torch
import matplotlib

from src.lib import vertex_labeling
import src.util.machina_data_extraction_util as mach_util
import src.util.vertex_labeling_util as vert_util

def predict_vertex_labeling(cluster_fn, tree_fn, ref_var_fn, ignore_polytomies):
    cluster_label_to_idx = mach_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies)
    idx_to_label = {v:k for k,v in cluster_label_to_idx.items()}

    T = torch.tensor(mach_util.get_adj_matrix_from_machina_tree(cluster_label_to_idx, tree_fn, skip_polytomies=ignore_polytomies), dtype = torch.float32)
    B = vert_util.get_mutation_matrix_tensor(T)
    ref_matrix, var_matrix, unique_sites = mach_util.get_ref_var_matrices_from_machina_sim_data(ref_var_fn,
                                                                                                cluster_label_to_idx=cluster_label_to_idx,
                                                                                                T=T)

    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    primary_idx = unique_sites.index('P')
    r = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T

    best_T_edges, best_labeling, best_G_edges = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites=unique_sites,
                                                                                            p=r, node_idx_to_label=idx_to_label,
                                                                                            w_e=0.01, w_l=0.8, w_m=100, max_iter=100, batch_size=64,
                                                                                            custom_colors=custom_colors, visualize=False)

    vert_util.write_tree(best_T_edges, os.path.join(SIM_DATA_DIR, f"T_seed{seed}.predicted.tree"))
    vert_util.write_tree_vertex_labeling(best_labeling, os.path.join(SIM_DATA_DIR, f"T_seed{seed}.predicted.vertex.labeling"))
    vert_util.write_migration_graph(best_G_edges, os.path.join(SIM_DATA_DIR, f"G_seed{seed}.predicted.tree"))

if __name__=="__main__":

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s <MACHINA_DATA_DIR>\n" % sys.argv[0])
        sys.exit(1)


    num_sims_runs = 0
    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]

    MACHINA_DATA_DIR = sys.argv[1]

    for site in sites:
        for mig_type in mig_types:

            SIM_DATA_DIR = os.path.join(MACHINA_DATA_DIR, "sims", site, mig_type)

            seeds = fnmatch.filter(os.listdir(SIM_DATA_DIR), 'seed*_0.95.tsv')
            seeds = [s.replace("_0.95.tsv", "").replace("seed", "") for s in seeds]

            for seed in seeds:

                print("="*150)
                print(f"Tree {num_sims_runs}")
                print(f"Predicting vertex labeling for {site} {mig_type} seed {seed}.")

                cluster_fn = os.path.join(SIM_DATA_DIR, f"clustering_observed_seed{seed}.txt")
                tree_fn = os.path.join(SIM_DATA_DIR, f"T_seed{seed}.tree")
                ref_var_fn = os.path.join(SIM_DATA_DIR, f"seed{seed}_0.95.tsv")
                predict_vertex_labeling(cluster_fn, tree_fn, ref_var_fn, ignore_polytomies=True)

                num_sims_runs += 1

    print(f"{num_sims_runs} simulations run")

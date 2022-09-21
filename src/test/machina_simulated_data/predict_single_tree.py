import sys
import os
import fnmatch
import torch
import matplotlib

from src.lib import vertex_labeling
import src.util.machina_data_extraction_util as mach_util
import src.util.vertex_labeling_util as vert_util

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, site_mig_data_dir):
    cluster_label_to_idx = mach_util.get_cluster_label_to_idx(cluster_fn, ignore_polytomies=True)

    data = mach_util.get_adj_matrices_from_all_mutation_trees(all_mut_trees_fn, cluster_label_to_idx, is_sim_data=True)
    custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]
    tree_num = 0
    for adj_matrix, pruned_cluster_label_to_idx in data:
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
        weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=0.5)

        best_T_edges, best_labeling, best_G_edges, best_loss_info = vertex_labeling.gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites=unique_sites,
                                                                                                weights=weights, p=r, node_idx_to_label=idx_to_label,
                                                                                                max_iter=100, batch_size=32,
                                                                                                custom_colors=custom_colors, visualize=False)

        tree_num += 1

if __name__=="__main__":

    if len(sys.argv) != 5:
        sys.stderr.write("Usage: %s <MACHINA_SIM_DATA_DIR> <site> <pattern> <seed>\n" % sys.argv[0])
        sys.exit(1)


    machina_sims_data_dir = sys.argv[1]

    site = sys.argv[2]
    mig_type = sys.argv[3]
    seed = sys.argv[4]
    site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)

    print("="*150)
    print(f"Predicting vertex labeling for {site} {mig_type} seed {seed}.")

    cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
    all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
    ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.tsv")

    predict_vertex_labelings(cluster_fn, all_mut_trees_fn, ref_var_fn, site_mig_data_dir)

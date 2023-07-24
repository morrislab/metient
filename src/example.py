import torch
from src.lib import vertex_labeling
from src.util import data_extraction_util as data_util
from src.util import pairtree_data_extraction_util as pt_util
from src.util import vertex_labeling_util as vert_util
from src.util import plotting_util as plot_util
import os

cd = os.getcwd()
os.chdir(cd)
data_dir = os.path.join(cd, "data/gundem_neuroblastoma_2023/patient_driver_genes/")
output_dir = os.path.join(cd, "metient_example_outputs/")
if not os.path.exists(output_dir):
   os.makedirs(output_dir)
patient_id = "H103207_clustered"

# Process input reference and variant reads
ref_var_fn = os.path.join(data_dir, f"{patient_id}_SNVs.tsv")
ref_matrix, var_matrix, unique_sites, idx_to_cluster_label = data_util.get_ref_var_matrices_from_real_data(ref_var_fn)

# Process input trees (this is for PairTree or Orchard trees)
tree_data = pt_util.get_adj_matrices_from_pairtree_results(os.path.join(data_dir, "orchard_trees", f"{patient_id}.results.npz"))

# Specify anatomical site label of primary site
primary_site = "left adrenal - D"

# Enumerate all possible trees (if there are multiple) and run migration history analysis
for i, (T, llh) in enumerate(tree_data):
    print(f"Solving migration history for tree {i}")
    if not vert_util.is_tree(T):
        print("Invalid tree was provided, skipping: \n", T)
        continue

    # Configure this to diplay mutations in a custom way 
    idx_to_cluster_label = {k:(" ").join(v.split('_')[0:2]) for k,v in idx_to_cluster_label.items()}
    primary_idx = unique_sites.index(primary_site)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
    
    weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=0.0)
    print_config = plot_util.PrintConfig(visualize=False, verbose=False, viz_intermeds=False, k_best_trees=4)
    vertex_labeling.get_migration_history(T, ref_matrix, var_matrix, unique_sites, p, idx_to_cluster_label, weights,
                                          print_config, output_dir, patient_id, batch_size=64, max_iter=200)


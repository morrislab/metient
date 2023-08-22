import sys
import os
import argparse
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.lib import vertex_labeling
from src.util import data_extraction_util as data_util
from src.util import pairtree_data_extraction_util as pt_util
from src.util import vertex_labeling_util as vert_util
from src.util import plotting_util as plot_util

custom_colors = ["#6aa84fff","#c27ba0ff", "#e69138ff", "#be5742e1", "#2496c8ff", "#674ea7ff"] + sns.color_palette("Paired").as_hex()

def find_labeling(ref_var_fn, tree, custom_colors, primary_site, patient_name, output_dir, weights, weight_init_primary):    
    ref_matrix, var_matrix, unique_sites, idx_to_full_cluster_label = data_util.get_ref_var_matrices_from_real_data(ref_var_fn)

    idx_to_cluster_label = dict()
    for ix in idx_to_full_cluster_label:
        og_label_muts = idx_to_full_cluster_label[ix].split(';') # e.g. CUL3:2:225371655:T;TRPM6:9:77431650:C
        if len(og_label_muts) > 3:
            og_label_muts = og_label_muts[:3]
        gene_names = []
        for mut_label in og_label_muts:
            gene_names.append(mut_label.split(":")[0])
        idx_to_cluster_label[ix] = ("_").join(gene_names)
    print(idx_to_cluster_label)
    print(tree.shape)

    print(f"Anatomical sites: {unique_sites}")   
    primary_idx = unique_sites.index(primary_site)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
    G = data_util.get_genetic_distance_tensor_from_adj_matrix(tree,idx_to_full_cluster_label, ";")
    print_config = plot_util.PrintConfig(visualize=False, verbose=False, viz_intermeds=False, k_best_trees=4)
    vertex_labeling.get_migration_history(tree, ref_matrix, var_matrix, unique_sites, p, idx_to_cluster_label,
                                          weights, print_config, output_dir, patient_name, G=G, 
                                          weight_init_primary=weight_init_primary, custom_colors=custom_colors, 
                                          batch_size=32, max_iter=100, lr_sched='step')

    
def run_conipher_patient(patient, weights, weight_init_primary, tsv_dir, tree_dir, output_dir):
    space = "x"*44
    tsv_fn = os.path.join(tsv_dir, f"{patient}_clustered_SNVs.tsv")
    print(f"{space} PATIENT {patient} {space}")
    df = pd.read_csv(tsv_fn, delimiter="\t")
    primary_sites = list(df[df['sample_type']=='primary']['anatomical_site_label'].unique())
    if (len(primary_sites) > 1):
        print("*Multiple primary samples, running metient once for each possible primary*")
    for primary_site in primary_sites:
        print(f"Primary site: {primary_site}")
        run_name = f"{patient}_{primary_site}"
        tree_fn = os.path.join(tree_dir, f"{patient}_conipher_SNVsallTrees_cleaned.txt")
        trees = data_util.get_adj_matrices_from_all_conipher_trees(tree_fn)
        find_labeling(tsv_fn, trees[0], custom_colors, primary_site, run_name, output_dir, weights, weight_init_primary)


if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='run metient on conipher generated trees (with and without genetic distance).')
	parser.add_argument('patient', type=str,
                    	help='an integer for the accumulator')
	parser.add_argument('tsv_dir', type=str,
                    	help='directory with clustered tsvs')
	parser.add_argument('tree_dir', type=str,
                    	help='directory with conipher trees')
	parser.add_argument('output_dir', type=str,
                        help='parent output directory, where subdirectories for each configuration will get created')
	args = parser.parse_args()
	patient = args.patient

	# (1) Maximum parsimony
	weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=0.0)
	output_dir = os.path.join(args.output_dir, "max_pars")
    print(output_dir)
	run_conipher_patient(patient, weights, False, args.tsv_dir, args.tree_dir, output_dir)

	# (2) Maximum parsimony + weight init primary
	weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=0.0)
	output_dir = os.path.join(args.output_dir, "max_pars_wip")
    print(output_dir)
	run_conipher_patient(patient, weights, True, args.tsv_dir, args.tree_dir, output_dir)

	# (3) Maximum parsimony + genetic distance
	weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=1.0)
	output_dir = os.path.join(args.output_dir, "max_pars_genetic_distance")
    print(output_dir)
	run_conipher_patient(patient, weights, False, args.tsv_dir, args.tree_dir, output_dir)

	# (4) Maximum parsimony + genetic distance + weight init primary
	weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0, gen_dist=1.0)
    print(output_dir)
	output_dir = os.path.join(args.output_dir, "max_pars_genetic_distance_wip")
	run_conipher_patient(patient, weights, True, args.tsv_dir, args.tree_dir, output_dir)


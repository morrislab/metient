import sys
import os
import argparse
import seaborn as sns
import pandas as pd
import torch
import glob
import shutil
import matplotlib.pyplot as plt
from src.lib.metient import *
from src.util import data_extraction_util as data_util

COLORS = ["#6aa84fff","#c27ba0ff", "#e69138ff", "#be5742e1", "#2496c8ff", "#674ea7ff"] + sns.color_palette("Paired").as_hex()

BATCH_SIZE=6000
SOLVE_POLYTOMIES=True

def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)    
    os.makedirs(directory_path)

def get_matching_basenames(directory, pattern, suffix):
    file_paths = glob.glob(os.path.join(directory, pattern))
    basenames = [os.path.basename(file_path).replace(suffix, "") for file_path in file_paths]
    return basenames

def shorten_cluster_names(idx_to_full_cluster_label):
	idx_to_cluster_label = dict()
	for ix in idx_to_full_cluster_label:
		og_label_muts = idx_to_full_cluster_label[ix].split(';') # e.g. CUL3:2:225371655:T;TRPM6:9:77431650:C
		if len(og_label_muts) > 2:
			og_label_muts = og_label_muts[:2]
		gene_names = []
		for mut_label in og_label_muts:
			gene_names.append(mut_label.split(":")[0])
		idx_to_cluster_label[ix] = ("_").join(gene_names)
	return idx_to_cluster_label

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='run metient on conipher generated trees (with and without genetic distance).')
	parser.add_argument('tsv_dir', type=str,
						help='directory with clustered tsvs')
	parser.add_argument('tree_dir', type=str,
						help='directory with conipher trees')
	parser.add_argument('output_dir', type=str,)
	args = parser.parse_args()

	create_directory(args.output_dir)
	sys.stdout = open(os.path.join(args.output_dir, f"output.txt"), 'a')

	patients = get_matching_basenames(args.tsv_dir, "*_clustered_SNVs.tsv", "_clustered_SNVs.tsv")
	print(f"{len(patients)} patients")
	print(f"Batch size: {BATCH_SIZE}, solve_polytomies: {SOLVE_POLYTOMIES}")

	# Collect all data
	Ts, ref_matrices, var_matrices, ordered_sites, all_primary_sites, node_idx_to_labels,run_names,Gs = [],[],[],[],[],[],[],[]
	total = 0
	for patient in patients:
		tsv_fn = os.path.join(args.tsv_dir, f"{patient}_clustered_SNVs.tsv")
		ref_matrix, var_matrix, unique_sites, idx_to_full_cluster_label = get_ref_var_matrices(tsv_fn)

		idx_to_shortened_label = shorten_cluster_names(idx_to_full_cluster_label)
		
		df = pd.read_csv(tsv_fn, delimiter="\t")
		primary_sites = list(df[df['sample_type']=='primary']['anatomical_site_label'].unique())

		for primary_site in primary_sites:
			tree_fn = os.path.join(args.tree_dir, f"{patient}_conipher_SNVsallTrees_cleaned.txt")
			trees = data_util.get_adj_matrices_from_all_conipher_trees(tree_fn)
			# Use the default tree
			tree = trees[0]
			Ts.append(tree)
			ref_matrices.append(ref_matrix)
			var_matrices.append(var_matrix)
			ordered_sites.append(unique_sites)
			node_idx_to_labels.append(idx_to_shortened_label)
			run_names.append(f"{patient}_{primary_site}")
			all_primary_sites.append(primary_site)
			G = get_genetic_distance_matrix_from_adj_matrix(tree,idx_to_full_cluster_label, ";")
			Gs.append(G)
			total += 1
	print("total trees", total)
	print(len(Ts),len(ref_matrices), len(var_matrices), len(ordered_sites), len(all_primary_sites), len(node_idx_to_labels), len(run_names))
	weights = Weights(gen_dist=1.0)
	print_config = PrintConfig(visualize=True, verbose=False, viz_intermeds=False, k_best_trees=BATCH_SIZE)
	calibrate(Ts,ref_matrices, var_matrices, ordered_sites, all_primary_sites,node_idx_to_labels,
              weights, print_config, args.output_dir, run_names, Gs=Gs,batch_size=BATCH_SIZE,custom_colors=COLORS,
              solve_polytomies=SOLVE_POLYTOMIES)



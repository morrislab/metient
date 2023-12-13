import sys
import os
import argparse
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.lib.metient import *
from src.util import data_extraction_util as data_util

custom_colors = ["#6aa84fff","#c27ba0ff", "#e69138ff", "#be5742e1", "#2496c8ff", "#674ea7ff"] + sns.color_palette("Paired").as_hex()

BATCH_SIZE=10000
WEIGHT_INIT_PRIMARY=True
MODE='calibrate'
SOLVE_POLYTOMIES=True

def find_labeling(ref_var_fn, tree, custom_colors, primary_site, patient_name, output_dir, weights):    
	ref_matrix, var_matrix, unique_sites, idx_to_full_cluster_label = get_ref_var_matrices(ref_var_fn)

	idx_to_cluster_label = dict()
	for ix in idx_to_full_cluster_label:
		og_label_muts = idx_to_full_cluster_label[ix].split(';') # e.g. CUL3:2:225371655:T;TRPM6:9:77431650:C
		if len(og_label_muts) > 2:
			og_label_muts = og_label_muts[:2]
		gene_names = []
		for mut_label in og_label_muts:
			gene_names.append(mut_label.split(":")[0])
		idx_to_cluster_label[ix] = ("_").join(gene_names)

	G = get_genetic_distance_matrix_from_adj_matrix(tree,idx_to_full_cluster_label, ";")
	print_config = PrintConfig(visualize=True, verbose=False, viz_intermeds=False, k_best_trees=10000)
	get_migration_history(tree, ref_matrix, var_matrix, unique_sites, primary_site, idx_to_cluster_label,
						  weights, print_config, output_dir, patient_name, G=G, 
						  weight_init_primary=WEIGHT_INIT_PRIMARY, custom_colors=custom_colors, 
						  batch_size=BATCH_SIZE, max_iter=100, mode=MODE, solve_polytomies=SOLVE_POLYTOMIES)

	
def run_conipher_patient(patient, weights, tsv_dir, tree_dir, output_dir):
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
		find_labeling(tsv_fn, trees[0], custom_colors, primary_site, run_name, output_dir, weights)


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
	# weights = Weights(data_fit=0.2, mig=(10.0, 8.0), mig_delta=0.8, seed_site=(1.0, 10.0), reg=2.0, gen_dist=0.0)
	# output_dir = os.path.join(args.output_dir, "max_pars")
	# run_conipher_patient(patient, weights, args.tsv_dir, args.tree_dir, output_dir)

	# (2) Maximum parsimony + genetic distance
	if MODE == 'evaluate':
		weights = Weights(data_fit=0.2, mig=[2.0], mig_delta=0.8, seed_site=[1.0], reg=2.0, gen_dist=0.2)
	elif MODE == 'calibrate':
		weights = Weights(data_fit=0.2, mig=[3,2,1,1,1], mig_delta=0.8, seed_site=[1,1,1,2,3], reg=2.0, gen_dist=1.0)
	output_dir = os.path.join(args.output_dir, "max_pars_genetic_distance")
	sys.stdout = open(os.path.join(output_dir, f"output.txt"), 'a')
	print(f"Weights:\nmig: {weights.mig}, mig_delts: {weights.mig_delta}, seeding: {weights.seed_site}, gen: {weights.gen_dist}")
	print(f"Batch size: {BATCH_SIZE}, wip: {WEIGHT_INIT_PRIMARY}, mode: {MODE}, solve_polytomies: {SOLVE_POLYTOMIES}")
	run_conipher_patient(patient, weights, args.tsv_dir, args.tree_dir, output_dir)



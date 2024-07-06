import sys
import os
import subprocess
import fnmatch
import argparse
import datetime
import math
import time

from metient.metient import *
from metient.lib.migration_history_inference import rank_solutions
from metient.util import eval_util as eutil
from metient.util import plotting_util as plot_util
from metient.util.vertex_labeling_util import create_reweighted_solution_set_from_pckl

from metient.util.globals import *
import torch
import pickle
import gzip
import matplotlib
import json
import pandas as pd

VISUALIZE=False

def get_num_mut_trees(mut_tree_fn):
    with open(mut_tree_fn, 'r') as f:
        # look for line w/ "3 #trees" as an example
        for line in f:
            if  "#trees" in line:
                return int(line.strip().split()[0])

def recalibrate(seed, tree_num, out_dir, ref_var_fn, weights, print_config, custom_colors, solve_polys):
    df = pd.read_csv(ref_var_fn, delimiter="\t", index_col=False)  
    # Extract unique site labels in the order of site_index
    sorted_tuples = sorted(zip(df['anatomical_site_index'].unique(), df['anatomical_site_label'].unique()))
    unique_sites = [t[1] for t in sorted_tuples]

    with gzip.open(os.path.join(out_dir, f"tree{tree_num}_seed{seed}_calibrate.pkl.gz"), 'rb') as f:
        pckl = pickle.load(f)
    
    saved_U = torch.tensor(pckl[OUT_OBSERVED_CLONES_KEY])
    primary_idx = unique_sites.index('P')
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
    final_solutions = rank_solutions(create_reweighted_solution_set_from_pckl(pckl, None, p, weights),
                                     print_config, needs_pruning=False)
    print("final_solutions:", len(final_solutions))

    plot_util.save_best_trees(final_solutions, saved_U, None, weights, unique_sites, print_config, 
                              custom_colors, 'P', out_dir, f"tree{tree_num}_seed{seed}_calibrate")

    return 
                


parser = argparse.ArgumentParser()
parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
parser.add_argument('run_name', type=str, help="Name of this run")

parser.add_argument('--mig', type=float, help="Weight on migration number", default=10.0)
parser.add_argument('--comig', type=float, help="Weight on comig", default=0.8)
parser.add_argument('--seed', type=float, help="Weight on seeding site number", default=1.0)
parser.add_argument('--gen', type=float, help="Weight on genetic distance", default=0.0)

parser.add_argument('--wip', action='store_true', help="If passed, initialize weights higher to favor vertex labeling of primary for all internal nodes", default=False)
parser.add_argument('--solve_polys', action='store_true', help="If passed, solve polytomies", default=False)
parser.add_argument('--bs', type=int, help="Batch size", default=32)

args = parser.parse_args()

machina_sims_data_dir = args.sim_data_dir
run_name = args.run_name

predictions_dir = os.path.join('/data/morrisq/divyak/data/metient_prediction_results', f"predictions_{run_name}")
os.mkdir(predictions_dir)
sys.stdout = open(os.path.join(predictions_dir, f"output.txt"), 'a')

sites = ["m8", "m5"]
mig_types = ["M", "mS", "R", "S"]

print(f"Mode: calibrate")
print(f"Solving polytomies: {args.solve_polys}")
print(f"Batch size: {args.bs}")
print(f"Weight biasing: {args.wip}")

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

# 1. Run trees in calibrate mode
files_to_check = []
for site in sites:
    os.mkdir(os.path.join(predictions_dir, site))

    for mig_type in mig_types:
        out_dir = os.path.join(predictions_dir, site, mig_type)
        os.mkdir(out_dir)
        #print(out_dir)
        site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)
        seeds = fnmatch.filter(os.listdir(site_mig_data_dir), 'reads_seed*.tsv')
        seeds = [s.replace(".tsv", "").replace("reads_seed", "") for s in seeds]
        #print(seeds)
        for seed in seeds:
            files_to_check.append(os.path.join(predictions_dir, f"perf_stats_{site}_{mig_type}_{seed}.txt"))
            num_trees = get_num_mut_trees(os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt"))
            job_time = math.ceil(num_trees*(1/2))+2 
            python_cmd = [f"bsub -J metient_sim_{site}_{mig_type}_{seed}_{run_name} -n 8 -W {job_time}:00  -R 'rusage[mem=8] span[hosts=1]' -o output_metient_sims_{run_name}.log -e error_metient_sims_{run_name}.log ./predict_single_simulated_tree.sh",
                          machina_sims_data_dir, site, mig_type, seed, str(args.mig),str(args.comig), str(args.seed), str(args.gen),
                          'True' if args.wip else 'False', str(args.bs), 'calibrate', 'True' if args.solve_polys else 'False', out_dir]
            cmd = " ".join(python_cmd)
            print(f"Submitting command: {cmd}")
            subprocess.run(" ".join(python_cmd), shell=True)

print(files_to_check)
assert(len(files_to_check)==80)
# Wait until all files exist
while not all(os.path.exists(file) for file in files_to_check):
    print("Waiting for files to exist...")
    time.sleep(10)  # Adjust the sleep duration as needed

print("All processes finished")
start_time = datetime.datetime.now()
# 2. Find the best thetas for each migration type category 
mig_type_to_trees = {m:[] for m in mig_types}
for site in sites:
    for mig_type in mig_types:
        site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)
        seeds = fnmatch.filter(os.listdir(site_mig_data_dir), 'reads_seed*.tsv')
        seeds = [s.replace(".tsv", "").replace("reads_seed", "") for s in seeds]
        for seed in seeds:
            num_trees = get_num_mut_trees(os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt"))
            for tree_num in range(num_trees):
                seeding_fn = os.path.join(predictions_dir, site, mig_type, f"tree{tree_num}_seed{seed}_calibrate.pkl.gz")
                mig_type_to_trees[mig_type].append(seeding_fn)
mig_type_to_calibration_time = {m:(datetime.datetime.now() - start_time).total_seconds() for m in mig_types}

mig_type_to_best_theta = {}
for mig_type in mig_type_to_trees:
    start_time = datetime.datetime.now()
    best_theta = eutil.get_max_cross_ent_thetas(pickle_file_list=mig_type_to_trees[mig_type], suffix="_calibrate", use_min_tau=False)
    print("BEST THETA", best_theta, "mig_type", mig_type)
    # eutil.plot_cross_ent_chart(all_theta_to_cross_ent_sum, mig_type, predictions_dir)
    mig_type_to_best_theta[mig_type] = best_theta
    mig_type_to_calibration_time[mig_type] += (datetime.datetime.now() - start_time).total_seconds()

rounded_mig_type_to_best_theta = {k:[round(v,3) for v in mig_type_to_best_theta[k]] for k in mig_type_to_best_theta}
print(rounded_mig_type_to_best_theta)
with open(os.path.join(predictions_dir, "mig_type_to_best_thetas.json"), 'w') as json_file:
    json.dump(rounded_mig_type_to_best_theta, json_file, indent=2)

print_config = PrintConfig(visualize=VISUALIZE, verbose=False, k_best_trees=args.bs, save_outputs=True)
custom_colors = [matplotlib.colors.to_hex(c) for c in ['limegreen', 'royalblue', 'hotpink', 'grey', 'saddlebrown', 'darkorange', 'purple', 'red', 'black', 'black', 'black', 'black']]

# import concurrent.futures
# with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
# futures = []
num_completed = 0
for site in sites:
    for mig_type in mig_types:
        start_time = datetime.datetime.now()
        # Recalibrate trees using the best thetas
        weights = Weights(mig=[mig_type_to_best_theta[mig_type][0]*50], comig=mig_type_to_best_theta[mig_type][1]*50, seed_site=[mig_type_to_best_theta[mig_type][2]*50],
                             gen_dist=args.gen, organotrop=0.0)
        out_dir = os.path.join(predictions_dir, site, mig_type)
        site_mig_data_dir = os.path.join(machina_sims_data_dir, site, mig_type)
        seeds = fnmatch.filter(os.listdir(site_mig_data_dir), 'reads_seed*.tsv')
        seeds = [s.replace(".tsv", "").replace("reads_seed", "") for s in seeds]
        for seed in seeds:
            cluster_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input", f"cluster_{mig_type}_seed{seed}.txt")
            all_mut_trees_fn = os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt")
            num_trees = get_num_mut_trees(os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt"))
            for tree_num in range(num_trees):
                ref_var_fn = os.path.join(machina_sims_data_dir, f"{site}_clustered_input_corrected", f"cluster_{mig_type}_seed{seed}_tree{tree_num}.tsv")
                recalibrate(seed, tree_num, out_dir,ref_var_fn, 
                            weights, print_config, custom_colors,args.solve_polys)
                if num_completed % 10 == 0:
                    print("***Num. recalibrated:***", num_completed)
                num_completed += 1
        mig_type_to_calibration_time[mig_type] += (datetime.datetime.now() - start_time).total_seconds()

print("Migration type to calibration time\n", mig_type_to_calibration_time)
with open(os.path.join(predictions_dir, f"mig_type_to_calibration_time.json"), 'w') as f:
    json.dump(mig_type_to_calibration_time, f, indent=2)


    #                 futures.append(executor.submit(recalibrate, site, mig_type, seed, tree_num, out_dir,
    #                                                ref_var_fn, data, weights, print_config, custom_colors))

    # num_complete = 0
    # # Wait for all processes to complete
    # for future in concurrent.futures.as_completed(futures):
    #     print("Num recalibrated:", num_complete)
    #     num_complete += 1
    #     pass  # Do nothing, just wait for completion


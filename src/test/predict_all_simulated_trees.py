import random
import sys
import os
import subprocess
import fnmatch
import argparse
import datetime
import concurrent.futures
import multiprocessing
from src.lib import vertex_labeling

def get_num_mut_trees(mut_tree_fn):
    with open(mut_tree_fn, 'r') as f:
        # look for line w/ "3 #trees" as an example
        for line in f:
            if  "#trees" in line:
                return int(line.strip().split()[0])
parser = argparse.ArgumentParser()
parser.add_argument('sim_data_dir', type=str, help="Directory containing machina simulated data")
parser.add_argument('run_name', type=str, help="Name of this run")

parser.add_argument('--data_fit', type=float, help="Weight on data fit", default=0.2)
parser.add_argument('--mig', type=float, help="Weight on migration number", default=10.0)
parser.add_argument('--comig', type=float, help="Weight on comigration number", default=7.0)
parser.add_argument('--seed', type=float, help="Weight on seeding site number", default=5.0)
parser.add_argument('--reg', type=float, help="Weight on regularization", default=1.0)
parser.add_argument('--gen', type=float, help="Weight on genetic distance", default=0.0)

parser.add_argument('--wip', action='store_true', help="If passed, initialize weights higher to favor vertex labeling of primary for all internal nodes", default=False)
parser.add_argument('--bs', type=int, help="Batch size", default=32)
parser.add_argument('--lr_sched', type=str, help="Learning rate schedule. See vertex_labeling.gumbel_softmax_optimization for details", default="step")
parser.add_argument('--cores', '-c', type=int, default=1, help="Number of cores to use (default 1)")

args = parser.parse_args()

machina_sims_data_dir = args.sim_data_dir
run_name = args.run_name

predictions_dir = f"predictions_{run_name}"
os.mkdir(predictions_dir)
sys.stdout = open(os.path.join(predictions_dir, f"output_{run_name}.txt"), 'w')

sites = ["m8", "m5"]
mig_types = ["M", "mS", "R", "S"]

print("Weights:")
weights = vertex_labeling.Weights(data_fit=args.data_fit, mig=args.mig, comig=args.comig, seed_site=args.seed, reg=args.reg, gen_dist=args.gen)

print(vars(weights))
print(f"Batch size: {args.bs}")
print(f"Learning rate schedule: {args.lr_sched}")
print(f"Placing higher weight on primary vertex labeling for all internal nodes: {args.wip}")

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")
print(f"Using {args.cores} cores.")

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
            num_trees = get_num_mut_trees(os.path.join(machina_sims_data_dir, f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt"))
            for tree_num in num_trees:
                python_cmd = [f"bsub -J metient_sim_{site}_{mig_type}_{seed} -n 8 -W 30:00 -o output_metient_sims.log -e error_metient_sims.log ./predict_single_simulated_tree.sh",
                              machina_sims_data_dir, site, mig_type, seed, tree_num, str(args.data_fit), str(args.mig),
                              str(args.comig), str(args.seed), str(args.reg), str(args.gen),
                              'True' if args.wip else 'False', str(args.bs), args.lr_sched, out_dir]
                cmd = " ".join(python_cmd)
                print(f"Submitting command: {cmd}")
                subprocess.run(" ".join(python_cmd), shell=True)


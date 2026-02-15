#!/usr/bin/python
'''
Evaluate how well solutions match when run 10 times, with varying batch size
'''
import sys
import os
import fnmatch
import numpy as np
from src.util.vertex_labeling_util import LabeledTree
from src.util.globals import *

import re
import pickle

# plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="whitegrid")
sns.set(font_scale=1.5)

sns.set_style("whitegrid")
sns.set_style("ticks")
sns.despine()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pc_map = {'mS':0, 'pS': 1, 'mM': 2, 'pM': 3, 'mR': 4, 'pR': 5}

from statannot import add_stat_annotation
from src.util import eval_util as eutil

def get_num_mut_trees(mut_tree_fn):
    with open(mut_tree_fn, 'r') as f:
        # look for line w/ "3 #trees" as an example
        for line in f:
            if  "#trees" in line:
                return int(line.strip().split()[0])


def get_matching_directories(pattern):
    # Use a regular expression to find matching directories
    matching_directories = {}

    for root, dirs, _ in os.walk("./"):
        for directory in dirs:
            match = re.match(re.compile(pattern), directory)
            if match:
                batch = int(directory.split("_")[2])
                if batch not in matching_directories:
                    matching_directories[batch] = []
                matching_directories[batch].append(os.path.join(root, directory))
    return matching_directories


# def get_
if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.stderr.write("Usage: %s <REGEX_PATTERN_PREDICTIONS_DIR> <MACHINA_SIM_DATA_DIR> <PLOT_NAME>\n" % sys.argv[0])
        sys.stderr.write("REGEX_PATTERN_PREDICTIONS_DIR: what string to match when looking for directories with multiple batch sizes/runs. e.g. predictions_bs_\d+_gd_r\d+_wip_09212023\n")
        sys.stderr.write("MACHINA_SIM_DATA_DIR: dir for sim data.\n")
        sys.stderr.write("PLOT_NAME: What to use in output png naming.\n")

        sys.exit(1)


    sites = ["m8", "m5"]
    mig_types = ["M", "mS", "R", "S"]
    topk = 5 # how many of the top solutions to check for 
    print(f"matching top {topk} solutions")
    matching_directories = get_matching_directories(sys.argv[1])
    print(matching_directories)
    first_dir = matching_directories[list(matching_directories.keys())[0]][0] # grab any of the predictions
    print(first_dir)
    bs_to_num_matches = {}

    for bs in matching_directories:
        bs_to_num_matches[bs] = []
        for site in sites:
            for mig_type in mig_types:
                # Get all seeds for mig_type + site combo
                seeds = fnmatch.filter(os.listdir(os.path.join(first_dir, site, mig_type)), '*.pickle')
                seeds = list(set([s.replace(".pickle", "").replace("seed", "").split("_")[1] for s in seeds]))
                print(seeds)
                for seed in seeds:
                    # Get all the clone trees for this seed
                    num_trees = get_num_mut_trees(os.path.join(sys.argv[2], f"{site}_mut_trees", f"mut_trees_{mig_type}_seed{seed}.txt"))
                    for tree_num in range(num_trees):
                        # across all the runs for this mig_type + site + seed + clone tree combo, 
                        # see how many solutions are the same
                        solution_sets = []
                        for predictions_dir in matching_directories[bs]:
                            predicted_site_mig_type_data_dir = os.path.join(predictions_dir, site, mig_type)
                            metient_pickle = open(os.path.join(predicted_site_mig_type_data_dir, f"tree{tree_num}_seed{seed}.pickle"), "rb")
                            pckl = pickle.load(metient_pickle)
                            Vs = pckl[OUT_LABElING_KEY]
                            k = topk if len(Vs) >= topk else len(Vs)
                            if topk != k:
                                print("didn't find topk solutions",topk, k, bs, predictions_dir)
                            Vs = Vs[:k]
                            A = pckl[OUT_ADJ_KEY]
                            single_run_solution_set = set()
                            for V in Vs:
                                single_run_solution_set.add(LabeledTree(A, V))
                            solution_sets.append(single_run_solution_set)
                        # how many solutions appear across all x runs?
                        intersection_set = solution_sets[0]  # Initialize with the first set
                        for s in solution_sets[1:]:
                            intersection_set = intersection_set.intersection(s)
                        bs_to_num_matches[bs].append(len(intersection_set))
        print(bs, bs_to_num_matches[bs])

    avg_matches = {bs:sum(bs_to_num_matches[bs])/len(bs_to_num_matches[bs]) for bs in bs_to_num_matches}
    print(avg_matches)

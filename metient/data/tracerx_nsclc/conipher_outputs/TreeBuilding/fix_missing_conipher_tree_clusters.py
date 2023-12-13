# The clusters produced from CONIPHER (as indicated by {patient}treeTable.tsv treeCLUSTER column) 
# are not all used in the final trees (as indicated by {patient}allTrees.txt)
# This makes indexing etc downstream unanoyyingly complicated, so we preprocess to remedy the issue
# before later use

import os
import pandas as pd
import argparse
import glob

from src.util import data_extraction_util as dutil

def get_all_tree_edges(tree_fn):
    out = []
    tree_clusters_used = set()
    with open(tree_fn, 'r') as f:
        tree_edges = []
        for i, line in enumerate(f):
            if i < 2: continue
            # This marks the beginning of a tree
            if "# tree" in line:
                out.append(tree_edges)
                tree_edges = []
            else:
                nodes = line.strip().split()
                # -1 for 0-indexing
                tree_edges.append((int(nodes[0]), int(nodes[1])))
                tree_clusters_used.add(int(nodes[0]))
                tree_clusters_used.add(int(nodes[1]))
        out.append(tree_edges)
    return out, tree_clusters_used

def write_fixed_trees(tree_fn, patient_id, output_dir, old_clust_new_clust_map):
    fixed_output = []
    with open(tree_fn, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                fixed_output.append(line.strip())
            else:
                nodes = line.strip().split()
                fixed_output.append(f"{old_clust_new_clust_map[int(nodes[0])]}\t{old_clust_new_clust_map[int(nodes[1])]}")
    with open(os.path.join(output_dir, f"{patient_id}allTrees_cleaned.txt"), "w") as file:
        for line in fixed_output:
            file.write(line)
            file.write("\n")

def check_edges(all_tree_edges, true_clusters_to_keep):
    # Check that each individual tree uses the same clusters
    # as the ones we're keeping
    for edges in all_tree_edges:
        clusters_used = set()
        for edge in edges:
            clusters_used.add(edge[0])
            clusters_used.add(edge[1])
        assert(true_clusters_to_keep==clusters_used)

def fix_missing_conipher_tree_clusters(patient_id, tree_dir, output_dir):
    tree_info = pd.read_csv(os.path.join(tree_dir, f"{patient_id}treeTable.tsv"), sep="\t")
    tree_info_clusters = set(tree_info['treeCLUSTER'].unique())
    print(tree_info_clusters)

    tree_fn = os.path.join(tree_dir, f"{patient_id}allTrees.txt")
    all_tree_edges, true_clusters_to_keep = get_all_tree_edges(tree_fn)

    print(true_clusters_to_keep)
    check_edges(all_tree_edges, true_clusters_to_keep)

    clusters_to_remove = tree_info_clusters - true_clusters_to_keep 
    print("clusters_to_remove", clusters_to_remove)
    if len(clusters_to_remove) == 0:
        print("no clusters to remove")

    old_clust_new_clust_map = dict()
    for i, c in enumerate(true_clusters_to_keep):
        old_clust_new_clust_map[c] = i
    print("old_clust_new_clust_map", old_clust_new_clust_map)

    tree_info = tree_info[~tree_info['treeCLUSTER'].isin(clusters_to_remove)]
    tree_info['treeCLUSTER'] = tree_info.apply(lambda row: old_clust_new_clust_map[row['treeCLUSTER']], axis=1)
    tree_info.to_csv(os.path.join(output_dir, f"{patient_id}treeTable_cleaned.tsv"), sep="\t")

    write_fixed_trees(tree_fn, patient_id, output_dir, old_clust_new_clust_map)

parser = argparse.ArgumentParser()
parser.add_argument('tree_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()

matching_files = glob.glob(os.path.join(args.tree_dir, '*' + "treeTable.tsv"))
patient_ids = [os.path.splitext(os.path.basename(file))[0].replace("treeTable","") for file in matching_files]

for patient_id in patient_ids:
    print("\n")
    print(patient_id)
    fix_missing_conipher_tree_clusters(patient_id, args.tree_dir, args.output_dir)

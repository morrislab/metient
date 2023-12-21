

## Installation


```bash
# mamba used for speed, can use conda instead if mamba is not installed
mamba create -n "met" python=3.8.8 ipython
mamba activate met
pip install metient
```

## Test your installation
```bash
cd src
python example.py
```
If everything goes well with installation, you should have output saved at `src/metient_example_outputs/H103207_clustered.png`

## Code example for running metient

```python
import torch
from src.lib import metient
import os

# SET THESE VARIABLES
ref_var_fn = "/path/to/ref/var/tsv"
tree_fn = "/path/to/tree/results/npz"
output_dir = "/path/to/save/outputs"
run_name = "patient_id"
primary_site = "primary" # Specify anatomical site label of primary site
weights = vertex_labeling.Weights(data_fit=1.0, mig=3.0, comig=2.0, seed_site=1.0, reg=2.0)
print_config = plot_util.PrintConfig(visualize=True, verbose=False, viz_intermeds=False, k_best_trees=4)

# Process input reference and variant reads
ref_matrix, var_matrix, unique_sites, idx_to_cluster_label = data_util.get_ref_var_matrices_from_real_data(ref_var_fn)

# Process input trees (this is for PairTree or Orchard trees)
tree_data = pt_util.get_adj_matrices_from_pairtree_results(tree_fn)

# Enumerate all possible trees (if there are multiple) and run migration history analysis
for i, (T, llh) in enumerate(tree_data):
    print(f"TREE {i}, llh {llh}")
    if not vert_util.is_tree(T):
        print("Invalid tree was provided, skipping: \n", T)
        continue

    # Configure this to diplay mutations in a custom way 
    primary_idx = unique_sites.index(primary_site)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(unique_sites)).T
    vertex_labeling.get_migration_history(T, ref_matrix, var_matrix, unique_sites, p, idx_to_cluster_label, weights,
                                          print_config, output_dir, patient_id)
```


## Example of using the results of Metient for downstream analysis

```python
import pickle
from src.util import plotting_util as plot_util

file = open("/path/to/your/pickle/file","rb")
pckl = pickle.load(file)
print(pckl.keys())
# V is the best ancestral labeling
V = pckl['ancestral_labelings'][0]
# A is the adjacency matrix
A = pckl['full_adjacency_matrices'][0]
# G represents the migration graph
G = plot_util.get_migration_graph(V, A)
# Get the seeding pattern for this patient (e.g. "polyconal single-source seeding")
seeding_pattern = plot_util.get_seeding_pattern_from_migration_graph(G)
```

In the pickle file you'll find the following keys:
* `ordered_anatomical_sites`: a list of anatomical sites in the order used for the matrices detailed below.
* `full_node_idx_to_label`: list of dictionaries, in order from best to worst tree. Each dictionary maps node index (as used for the matrices detailed below) to the label used on the tree. The reason this is different from what is inputted is that metient adds leaf nodes which correspond to the inferred subclonal presence of each node in anatomical sites. Each tree has a different set of possible leaf nodes, and each leaf node is labeled as <parent_node_name>_<anatomical_site>.
* `ancestral_labelings`: list of tensors, in order from best to worst tree. Each tensor is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. Each column is a one-hot vector representing the location inferred by Metient for that node.
* `subclonal_presence_matrices`: list of tensors, in order from best to worst tree. Each tensor is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. A value at i,j greater than 0.05 indicates that that node is present in that antomical site. These are the nodes that get added as leaf nodes.
* `full_adjacency_matrices`: list of tensors, in order from best to worst tree. Each tensor is a matrix (shape: `len(full_node_idx_to_label.values())`, `len(full_node_idx_to_label.values())`). A 1 at index i,j indicates an edge from i to j.



## Installation


```bash
# mamba used for speed, can use conda instead if mamba is not installed
mamba create -n "met" python=3.8.8 ipython
mamba activate met
pip install metient
```

## Tutorial
A tutorial for running calibrate mode is available [here](https://github.com/divyakoyy/metient/blob/main/metient_calibrate_tutorial.ipynb)


## Example of using the results of Metient for downstream analysis

```python
import pickle
from metient.util import plotting_util as pl

file = open("/path/to/your/pickle/file","rb")
pckl = pickle.load(file)
print(pckl.keys())
# V is the best ancestral labeling
V = pckl['ancestral_labelings'][0]
# A is the adjacency matrix
A = pckl['full_adjacency_matrices'][0]
# G represents the migration graph
G = pl.get_migration_graph(V, A)
# Get the seeding pattern for this patient (e.g. "polyconal single-source seeding")
seeding_pattern = plot_util.get_seeding_pattern_from_migration_graph(G)
```

In the pickle file you'll find the following keys:
* `ordered_anatomical_sites`: a list of anatomical sites in the order used for the matrices detailed below.
* `full_node_idx_to_label`: list of dictionaries, in order from best to worst tree. Each dictionary maps node index (as used for the matrices detailed below) to the label used on the tree. The reason this is different from what is inputted is that metient adds leaf nodes which correspond to the inferred subclonal presence of each node in anatomical sites. Each tree has a different set of possible leaf nodes, and each leaf node is labeled as <parent_node_name>_<anatomical_site>.
* `ancestral_labelings`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. Each column is a one-hot vector representing the location inferred by Metient for that node.
* `subclonal_presence_matrices`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. A value at i,j greater than 0.05 indicates that that node is present in that antomical site. These are the nodes that get added as leaf nodes.
* `full_adjacency_matrices`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(full_node_idx_to_label.values())`, `len(full_node_idx_to_label.values())`). A 1 at index i,j indicates an edge from i to j.

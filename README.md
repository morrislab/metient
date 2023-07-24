

## Installation

```bash
git clone git@github.com:divyakoyy/met_history_prediction.git
cd met_history_prediction
conda create -n "met" python=3.8.8 ipython
conda activate met
pip3 install -r requirements.txt
python setup.py install
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
from src.lib import vertex_labeling
from src.util import data_extraction_util as data_util
from src.util import pairtree_data_extraction_util as pt_util
from src.util import vertex_labeling_util as vert_util
from src.util import plotting_util as plot_util
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

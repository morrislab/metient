# Metient Guide

Metient takes a clone tree and mutation data from one or more tumor sites as input, and infers where each clone originated to produce a migration history that describes how cancer spread between anatomical sites in a patient.

> Start with a [tutorial notebook](../tutorial/) for a hands-on walkthrough. Use this guide as a reference alongside it for understanding parameters, input formats, and interpreting outputs.

## Step 1: Choose your function

Metient exposes four main functions. Pick yours based on two questions.

**How many patients do you have?** If you have a cohort of ~5 or more patients with the same cancer type, use **Metient-calibrate** — it learns optimal parsimony weights from your data. If you have fewer patients, or want to use pre-calibrated or custom weights, use **Metient-evaluate**. 

**Do you know which clones are present at which sites?** Metient needs to know which mutation clusters (clones) are present at each anatomical site. If you have reference and variant read counts, Metient can estimate this for you. If you already have binary present/absent calls per clone per site, you can provide those directly (e.g. from single-cell data, or the output of your tree estimation).

**Do you have barcode-based lineage-tracing data?** We recommend using **Metient-evaluate**. Since you know where each cell is sampled, follow tutorial 4.

|  | **I have ref/var read counts** and want Metient to estimate which clones are present at each site | **I already know which clones are present** at each site (binary present/absent per clone per site) |
|--|---|---|
| **Evaluate** (pre-set weights, any # of patients) | `met.evaluate()` — [Tutorial 3](../tutorial/3_evaluate_infer_observed_clones_label_clone_tree_tutorial.ipynb) | `met.evaluate_label_clone_tree()` — [Tutorial 4](../tutorial/4_evaluate_label_clone_tree_tutorial.ipynb) |
| **Calibrate** (learn weights, cohort of ~5+) | `met.calibrate()` — [Tutorial 1](../tutorial/1_calibrate_infer_observed_clones_label_clone_tree_tutorial.ipynb) | `met.calibrate_label_clone_tree()` — [Tutorial 2](../tutorial/2_calibrate_label_clone_tree_tutorial.ipynb) |

---

## Step 2: Prepare your input files

Each patient needs two files: a TSV with mutation/sample data and a TXT file with clone tree edges.

### Tree TXT file

Each line is a space-separated edge `parent_index child_index`. Indices must match `cluster_index` in the TSV. [Example](../tutorial/inputs/A_tree.txt).

### TSV file

Which TSV format you need depends on whether you already know which clones are present in which sites. If so, use 1b. If you want Metient to infer this from ref/var read counts, use 1a.

#### Format 1a: Read count TSV

Used by: `evaluate()`, `calibrate()` — [example](../tutorial/inputs/A_SNVs.tsv)

Each row = reference and variant read counts at a single locus in a single tumor sample.

| Column | Description |
|--------|-------------|
| **anatomical_site_index** | Zero-based index for anatomical_site_label. Rows with same site index and cluster_index are pooled. |
| **anatomical_site_label** | Name of the anatomical site |
| **character_index** | Zero-based index for character_label |
| **character_label** | Mutation name (short, no colons) |
| **cluster_index** | Cluster index, must match tree TXT file. Rows with same site index and cluster_index are pooled. |
| **ref** | Reference allele read count |
| **var** | Variant allele read count |
| **site_category** | `primary` or `metastasis`. Multiple primaries triggers one run per candidate. |
| **var_read_prob** | Variant read correction factor for copy number effects on variant allele frequency. See [details below](#var_read_prob-details). |

#### Format 1b: Known clone presence TSV

Used by: `evaluate_label_clone_tree()`, `calibrate_label_clone_tree()` — [example](../tutorial/inputs/CRUK0003_SNVs.tsv)

Each row = a single mutation/cluster in a single tumor sample.

| Column | Description |
|--------|-------------|
| **anatomical_site_index** | Zero-based index for anatomical_site_label. Rows with same site index and cluster_index are pooled. |
| **anatomical_site_label** | Name of the anatomical site |
| **cluster_index** | Cluster index, must match tree TXT file. Rows with same site index and cluster_index are pooled. |
| **cluster_label** | Mutation/cluster name (short, no colons) |
| **present** | `0` or `1` — whether this clone is present at this site |
| **site_category** | `primary` or `metastasis`. Multiple possible primaries triggers one run per candidate. |
| **num_mutations** | [Optional] Number of mutations in this cluster. NOTE: We do not recommend using this for barcode-based lineage tracing data. |

#### var_read_prob details

`var_read_prob` corrects for the effect that copy number alterations (CNAs) have on the relationship between variant allele frequency (VAF, the proportion of alleles with the mutation) and subclonal frequency (the proportion of cells with the mutation).

For a given mutation j, `var_read_prob` is the probability of observing a read from the variant allele in a cell that carries mutation j. This depends on how many copies of the reference vs. variant allele exist in that cell:

| Scenario | var_read_prob | Why |
|----------|---------------|-----|
| Diploid locus, no CNAs | `0.5` | 1 variant + 1 reference allele |
| Haploid locus (e.g., male sex chromosome), no CNAs | `1.0` | 1 variant allele, no reference |
| CNA duplicated the reference allele before mutation j occurred | `0.333` | 1 variant + 2 reference alleles |

**Using a copy number caller:** If your CN caller reports major and minor copy number, see B.1 in [Metient's supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-025-02924-8/MediaObjects/41592_2025_2924_MOESM1_ESM.pdf) on how to compute `var_read_prob`.

---

## Step 3: Configure weights

How you set weights depends on whether you're using **evaluate** or **calibrate**.

### If using evaluate: pick or create weights

You can use a pre-calibrated preset:

| Preset | How the weights were fit | Recommended for |
|--------|-------------|-----------------|
| `Weights.pancancer_genetic_organotropism_uniform_weighting()` | Genetic + organotropism, uniform cohort weighting | **Human data** |
| `Weights.pancancer_genetic_uniform_weighting()` | Genetic only, uniform cohort weighting | **Non-human data** |
| `Weights.pancancer_genetic_cohort_size_weighting()` | Genetic only, weighted by cohort size | |
| `Weights.pancancer_genetic_organotropism_cohort_size_weighting()` | Genetic + organotropism, weighted by cohort size | |

```python
weights = met.Weights.pancancer_genetic_organotropism_uniform_weighting()
```

Or define custom weights:

```python
weights = met.Weights(mig=0.5, comig=0.3, seed_site=0.2, gen_dist=0.01, organotrop=0.005)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mig` | — | Penalizes total migration number |
| `comig` | — | Penalizes co-migration number |
| `seed_site` | — | Penalizes number of seeding sites |
| `gen_dist` | `0.0` | Penalizes genetic distance (requires `num_mutations` in input). NOTE: not recommended for use when using barcode-based lineage tracing data. |
| `organotrop` | `0.0` | Penalizes deviation from organotropism priors |

> [!TIP]
> Use much higher penalties for parsimony weights (`mig`, `comig`, `seed_site`) than for `gen_dist` or `organotrop`. If using both genetic distance and organotropism, set both > 0.

### If using calibrate: choose a calibration mode

Calibrate learns weights for you. The `calibration_type` parameter controls which objectives are used:

| `calibration_type` | Description |
|-------------------|-------------|
| `"genetic"` | Calibrates using genetic distance between clones |
| `"organotropism"` | Calibrates using organ-specific metastasis frequencies |
| `"both"` | Calibrates using both genetic distance and organotropism |

### Organotropism dictionaries (optional)

If using organotropism (in either evaluate or calibrate), provide a dictionary per patient mapping site names to metastasis frequencies. Values should be normalized (sum to 1).

```python
Os = [
    {"Liver": 0.5, "Lung": 0.4, "Brain": 0.1},   # Patient 1
    {"Lymph": 0.7, "Bone": 0.3},                 # Patient 2
]
```
> [!TIP]
> If you don't have organotropism frequencies for your cancer type, you can compute them from our [MSK-MET metastasis counts table](https://github.com/morrislab/metient/blob/main/data/msk_met/msk_met_freq_by_cancer_type.csv), derived from [Nguyen et al.](https://www.cell.com/cell/fulltext/S0092-8674(22)00003-4). This table has counts by cancer type × metastatic site. Make sure to normalize so the frequencies provided for each patient sum to 1. Note that your cancer type and the patient's tumor sites need to be mappable to the cancer types and sites in this table.
 
---

## Step 4: Set run options

### Performance parameters
 
| Parameter | Default | What it does |
|-----------|---------|--------------|
| `sample_size` | `-1` (auto) | How many solutions to explore in parallel per run. Limited by memory. |
| `num_runs` | `-1` (auto) | How many times to repeat the full algorithm. Runs are sequential, so doesn't cost extra memory. |
| `solve_polytomies` | `False` | Try resolving nodes with >2 children into binary splits. Slower, and not tested on trees beyond 100 nodes. |
 
The defaults auto-calculate both `sample_size` and `num_runs` based on your tree size and number of sites. **This is fine for trees <30 nodes.**
 
`sample_size` and `num_runs` are the main drivers of runtime. Each run processes `sample_size` solutions in parallel, and Metient repeats this `num_runs` times sequentially. If Metient is running slower than expected, these are the knobs to turn down, but doing so may affect result quality.
 
If your results change significantly between runs, increase `num_runs`. If you want to override, set both explicitly — the total samples Metient considers is `num_runs × sample_size`.
 
> [!IMPORTANT]
> **Use a GPU for trees with >100 nodes.** Metient is much faster on GPU and large inputs may not be practical on CPU. If you run into GPU memory issues on really large trees, reduce `sample_size` and increase `num_runs` to compensate.

### PrintConfig

Controls visualization and output saving:

```python
print_config = met.PrintConfig(
    visualize=True,       # Visualize loss, best tree, and migration graph
    verbose=False,         # Print debug info
    k_best_trees=10,       # Number of solutions to visualize (all are saved to pkl)
    save_outputs=True,     # Save pngs and pickle files
    custom_colors=None,    # Array of hex strings (length = number of anatomical sites)
    display_labels=True    # Display node labels on migration history tree
)
```
---

## Step 5: Run Metient

### Evaluate (single patient)

```python
import metient as met

weights = met.Weights.pancancer_genetic_organotropism_uniform_weighting()
print_config = met.PrintConfig(visualize=True, save_outputs=True, k_best_trees=5)

results = met.evaluate(
    tree_fn="path/to/tree.txt",
    tsv_fn="path/to/mutations.tsv",
    weights=weights,
    print_config=print_config,
    output_dir="./output",
    run_name="patient_1"
)
```

### Calibrate (cohort)

```python
import metient as met

print_config = met.PrintConfig(visualize=True, save_outputs=True, k_best_trees=5)

results = met.calibrate(
    tree_fns=["patient1_tree.txt", "patient2_tree.txt", "patient3_tree.txt"],
    tsv_fns=["patient1.tsv", "patient2.tsv", "patient3.tsv"],
    print_config=print_config,
    output_dir="./output",
    run_names=["patient_1", "patient_2", "patient_3"],
    calibration_type="both",
    Os=[{"Liver": 0.5, "Lung": 0.4, "Brain": 0.1},
        {"Lymph": 0.7, "Bone": 0.3},
        {"Liver": 0.6, "Lung": 0.4}]
)
```

> [!NOTE]
> `tree_fns[i]`, `tsv_fns[i]`, `run_names[i]`, and `Os[i]` must all correspond to patient i.

### Full function signatures

```python
met.evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name,
             O=None, sample_size=-1, solve_polytomies=False, num_runs=5)

met.evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name,
                              O=None, sample_size=-1, solve_polytomies=False, num_runs=5)

met.calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
              Os=None, sample_size=-1, solve_polytomies=False, num_runs=5)

met.calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
                               Os=None, sample_size=-1, solve_polytomies=False, num_runs=5)
```

---

## Step 6: Interpret your results

Metient outputs a pickle file per patient in your `output_dir`.

### Quick inspection

```python
import pickle, gzip
import metient as met

with gzip.open("output/patient_1.pkl.gz", "rb") as f:
    pkl = pickle.load(f)

print(met.weighted_seeding_pattern(pkl))
print(met.weighted_phyleticity(pkl))
print(met.weighted_site_clonality(pkl))
print(met.weighted_genetic_clonality(pkl))
```

### Pickle file contents

| Key | Description |
|-----|-------------|
| **anatomical_sites** | List of sites in the order used for all matrices below |
| **node_info** | List of dicts (best → worst solution). Maps node index → `(label, is_leaf, is_polytomy_resolver_node)`. Leaf nodes added by Metient are labeled `<parent>_<site>`. |
| **node_labels** | List of arrays (best → worst solution). Shape `(num_sites, num_nodes)`. Each column is a one-hot site assignment. |
| **parents** | List of 1-D arrays (best → worst solution). `parents[i]` = parent of node i; root = `-1`. |
| **observed_proportions** | Array `(num_sites, num_clusters)`. Values > 0.05 = clone present at site. |
| **losses** | List of losses, best → worst solution|
| **probabilities** | List of probabilities, best → worst solution|
| **primary_site** | Anatomical site used as primary |
| **loss_info** | List of dicts with unweighted loss components (migration number, comigration number, etc.) |

### Analysis utilities

These functions help characterize the migration history. The **weighted** variants (recommended) operate on the full pickle dict and weight across all solutions by probability:

| Function | Returns |
|----------|---------|
| `met.weighted_seeding_pattern(pkl)` | Weighted seeding pattern |
| `met.weighted_site_clonality(pkl)` | Weighted site clonality |
| `met.weighted_genetic_clonality(pkl)` | Weighted genetic clonality |
| `met.weighted_phyleticity(pkl, sites=None)` | Weighted phyleticity (optionally restricted to specific sites) |

For single-solution analysis, extract matrices from the pickle and use these functions (`V` = vertex labeling matrix, `A` = adjacency matrix, `node_info` = dict from pickle):

| Function | Returns |
|----------|---------|
| `met.migration_graph(V, A)` | Migration graph |
| `met.seeding_pattern(V, A)` | {primary single-source, single-source, multi-source, reseeding} |
| `met.site_clonality(V, A)` | Monoclonal / polyclonal |
| `met.genetic_clonality(V, A, node_info)` | Monoclonal / polyclonal |
| `met.phyleticity(V, A, node_info)` | Monophyletic / polyphyletic |
| `met.seeding_clusters(V, A, node_info)` | Nodes whose parent has a different site |
| `met.adjacency_matrix_from_parents(parents)` | Sparse adjacency matrix from parents vector |

```python
# Example: per-solution analysis
A = met.adjacency_matrix_from_parents(pkl["parents"][0])
V = pkl["node_labels"][0]
node_info = pkl["node_info"][0]

print(met.seeding_pattern(V, A))
print(met.phyleticity(V, A, node_info))
```

---

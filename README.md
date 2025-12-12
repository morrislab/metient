# Metient
<!-- <p align="center">
<img src="metient/logo.png" width="150">
</p> -->

<p align="center">
  <img src="metient/method_overview.png?cachebust=12345" width="1000" style="padding:20px;">
</p>

**Metient** (**MET**astasis + gradi**ENT**) is a tool for inferring the metastatic migrations of a patient's cancer. You can find our preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.09.602790).

## Table of contents
1. [System requirements](#system-requirements)
2. [Installation](#installation)
3. [Tutorial](#tutorial)
4. [Inputs](#inputs)
5. [Usage](#usage)
6. [Outputs](#outputs)


## System requirements

### Hardware requirements
Metient compute requirements depend on the input size of the data. Inputs with less than ~50 tree nodes and 6 tumor sites can be run on any computer with sufficient RAM. Inputs with larger tree sizes or tumor sites should use a GPU along with a larger amount of CPU RAM. No extra configuration is needed to run Metient on GPU (Metient will automatically detect and use a GPU if one is available).

### Software requirements
Metient has been tested on macOS Sonoma (14.4) and CentOS Linux 7 (Core).

## Installation

Installing and running a tutorial for Metient should take ~5 minutes.

Metient is available as a python library, installable via pip.
```bash
# Mamba or conda can be used
# Create and activate environment
mamba create -n met -c conda-forge python=3.9
mamba activate met

# Install graphviz dependencies first via mamba
mamba install -c conda-forge graphviz pygraphviz ipython

# Install metient 
pip install metient
```

> [!TIP]
> If `pip install metient` fails due to `fatal error: graphviz/cgraph.h: No such file or directory`, you need to set environment variables to point to your graphviz header files.
> Locate the path to your mamba environment (for e.g. using `mamba env list`), and run the following:
> ```bash
> export CFLAGS="-I/path/to/mamba/env/include"
> export LDFLAGS="-L/path/to/mamba/env/lib"
> pip install pygraphviz --user
> pip install metient
> ```

## Tutorial

To run the tutorial notebooks, clone this repo:
```bash
git clone git@github.com:morrislab/metient.git
cd metient/tutorial/
```

There are different Jupyter Notebook tutorials based on your use case:
1. I have a cohort of patients (~5 or more patients) with the same cancer type. (Metient-calibrate)
   - I want Metient to estimate which mutations/mutation clusters are present in which anatomical sites. [Tutorial 1](tutorial/1_calibrate_infer_observed_clones_label_clone_tree_tutorial.ipynb)
   - I know which mutations/mutation clusters are present in which anatomical sites.  [Tutorial 2](tutorial/2_calibrate_label_clone_tree_tutorial.ipynb)
3. I have a small number of patients, or I want to enforce my own parsimony metric weights. (Metient-evaluate)
   - I want Metient to estimate which mutations/mutation clusters are present in which anatomical sites. [Tutorial 3](tutorial/3_evaluate_infer_observed_clones_label_clone_tree_tutorial.ipynb)
   - I know which mutations/mutation clusters are present in which anatomical sites. [Tutorial 4](tutorial/4_evaluate_label_clone_tree_tutorial.ipynb)
> [!TIP]
> If your jupyter notebook does not automatically recognize your conda environment, run the following:
> ```bash
> python -m ipykernel install --user --name met --display-name "met"
> ```
> Then in the jupyter notebook, select Kernel > Change kernel > met.

<details>
<summary><h2>🔽 Input format</h2></summary>
   
## Inputs
There are two required inputs, a tsv file with information for each sample and mutation/mutation cluster, and a txt file specifying the edges of the clone tree.

### 1. **Tsv file**

There are two types of tsvs that are accepted, depending on if you'd like Metient to estimate the presence of cancer clones in each tumor site (1a), or if you'd like to input this yourself (1b). 

#### 1a. If you would like Metient to estimate the prevalance of each cancer clone in each tumor site, use the following input tsv format.

[1a example tsv](tutorial/inputs/A_SNVs.tsv)

Each row in this tsv should correspond to the reference and variant read counts at a single locus in a single tumor sample:
| Column name | Description |
|----------|----------|
| **anatomical_site_index** | Zero-based index for anatomical_site_label column. Rows with the same anatomical site index and cluster_index will get pooled together.| 
| **anatomical_site_label** | Name of the anatomical site |
| **character_index** | Zero-based index for character_label column |
| **character_label** | Name of the mutation. This is used in visualizations, so it should be short. NOTE: due to graphing dependencies, this string cannot contain colons. |
| **cluster_index** | If using a clustering method, the cluster index that this mutation belongs to. NOTE: this must correspond to the indices used in the tree txt file. Rows with the same anatomical site index and cluster_index will get pooled together.|
| **ref** | The number of reads that map to the reference allele for this mutation or mutation cluster in this anatomical site. |
| **var** | The number of reads that map to the variant allele for this mutation or mutation cluster in this anatomical site. |
| **site_category** | Must be one of `primary` or `metastasis`. If multiple primaries are specified, such that the `primary` label is used for multiple different anatomical site indices (i.e., the true primary is not known), we will run Metient multiple times with each primary used as the true primary. Output files are saved with the suffix `_{anatomical_site_label}` to indicate which primary was used in that run. |
| **var_read_prob** | This gives Metient the ability to correct for the effect copy number alterations (CNAs) have on the relationship between variant allele frequency (VAF, i.e., the proportion of alleles bearing the mutation) and subclonal frequency (i.e., the proportion of cells bearing the mutation). Let j = character_index. var_read_prob is the probabilty of observing a read from the variant allele for mutation at j in a cell bearing the mutation. Thus, if mutation at j occurred at a diploid locus with no CNAs, this should be 0.5. In a haploid cell (e.g., male sex chromosome) with no CNAs, this should be 1.0. If a CNA duplicated the reference allele in the lineage bearing mutation j prior to j occurring, there will be two reference alleles and a single variant allele in all cells bearing j, such that var_read_prob = 0.3333. If using a CN caller that reports major and minor CN: `var_read_prob = (p*maj)/(p*(maj+min)+(1-p)*2)`, where `p` is tumor purity, `maj` is major CN, `min` is minor CN, and we're assuming the variant allele has major CN. For more information, see S2.2 of [PairTree's supplementary info](https://aacr.silverchair-cdn.com/aacr/content_public/journal/bloodcancerdiscov/3/3/10.1158_2643-3230.bcd-21-0092/9/bcd-21-0092_supplementary_information_suppsm_sf1-sf21.pdf?Expires=1709221974&Signature=dJH6~Dg-6gEb-S88i0wDGW28QZn16keQj34Vo2tAvJL2cUJrQo48afpHPp-a2zAwQa~ET6SDgw3hb3ITacB06GDUc3GYCdCgYtfPMjFGwygFj-Q9xf-c44VAvwiyliwsBXK1shZmURlFMwSjzkwRwasuWu50sMNmeJSoVyX3nQ-rRBlK93aDR5s9c0l-p4aGvTi6QmfKJPsxXaHB4Lz5yXSl3Xd~JPK-Y~ltC14epDRb~MiSPWUFCAiYetUXcQ7J7vd6b4XQKT9PnYkjQtUq55tLSoUkOGe5JkJ32NXCeoT~l-XD97pCeDYVDOYzAuOkAG0tDYrPebEh2TGTA3fnbA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) for more details. |


#### 1b. If you would like to input the prevalence of each cancer clone in each tumor site, use the following input tsv format.

[1b example tsv](tutorial/inputs/CRUK0003_SNVs.tsv)

Each row in this tsv should correspond to a single mutation/mutation cluster in a single tumor sample:
| Column name | Description |
|----------|----------|
| **anatomical_site_index** | Zero-based index for anatomical_site_label column. Rows with the same anatomical site index and cluster_index will get pooled together.| 
| **anatomical_site_label** | Name of the anatomical site |
| **cluster_index** | If using a clustering method, the cluster index that this mutation belongs to. NOTE: this must correspond to the indices used in the tree txt file. Rows with the same anatomical site index and cluster_index will get pooled together.|
| **cluster_label** | Name of the mutation or cluster of mutations. This is used in visualizations, so it should be short. NOTE: due to graphing dependencies, this string cannot contain colons. |
| **present** | Must be one of `0` or `1`. `1` indicates that this mutation/mutation cluster is present in this anatomical site, and `0` indicates that it is not. |
| **site_category** | Must be one of `primary` or `metastasis`. If multiple primaries are specified, such that the `primary` label is used for multiple different anatomical site indices (i.e., the true primary is not known), we will run Metient multiple times with each primary used as the true primary. Output files are saved with the suffix `_{anatomical_site_label}` to indicate which primary was used in that run. |
| **num_mutations** | The number of mutations in this cluster. |


### 2. **Tree txt file**
A .txt file where each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv. 

[Example tree .txt file](tutorial/inputs/A_tree.txt)
</details>
   
<details>
<summary><h2>🔽 Usage</h2></summary>

## Usage

Below is a guide to the most important parameters for `Metient-calibrate` and `Metient-evaluate`. Each parameter is summarized with what it does, when to use it, and recommended settings.

### Core Parameter Summary Table

| Parameter | Default | What it Does | When to Use | Notes / Recommended Values |
|----------|---------|--------------|-------------|-----------------------------|
| **solve_polytomies** | `False` | Attempts to refine trees by resolving nodes with >2 children (polytomies) into binary resolutions. | Use when exploring alternative refinements of a tree with polytomies. | Can provide more parsimonious migration histories, but increases compute time. Not tested on trees >100 nodes. |
| **sample_size** | `-1` (auto) | Number of parallel solutions explored per run. | Use `-1` for Metient to auto-calculate a sample size for you; increase to reduce run-to-run variability. | Auto ≈ (num_sites)^(num_nodes). Trees with <20 nodes: ~4096 is typically sufficient. Consider using 10,000+ samples for larger trees. Increase until results stabilize; must fit available memory. |
| **num_runs** | `1` | Number of full algorithm repeats. | Use 2-5 for quick exploratory analysis; use much larger values (>50) for stable results or when memory limits sample_size. | If results vary by run, increase sample_size or num_runs. |
| **run_names** | User-defined | Controls output labels. | Always used. | Use unique, descriptive names; avoid special characters. For calibrate, run_names must match the order of inputs. |

**BEST PRACTICE**: The full number of samples considered by Metient is num_runs × sample_size. In practice, set sample_size as large as your memory allows, and rely on num_runs to stabilize the results through repeated, sequential runs.

---

### Weights
#### Preset Parimsony Models

We provide parsimony models pre-calibrated to patient data, that provide weights on migration number, comigration number, and seeding site number. 
| Preset Function | Description | Recommended For |
|-----------------|-------------|-----------------|
| **pancancer_genetic_organotropism_uniform_weighting()** | Combined genetic + organotropism model; uniform cohort weighting. | Recommended for human data. |
| **pancancer_genetic_uniform_weighting()** | Genetic-only model; uniform cohort weighting. | Recommended for non-human data. |
| **pancancer_genetic_cohort_size_weighting()** | Genetic-only; weighted by cohort size. |  |
| **pancancer_genetic_organotropism_cohort_size_weighting()** | Genetic + organotropism; weighted by cohort size. | |

**BEST PRACTICE**: If you want to also use genetic distance and organotropism in the model, you must set non-zero values for those weights (see below). We recommend using a much higher penalty on the parsimony metric weights (mig, comig, seed_site) than gen_dist and organotrop.

Example usage:
```python
weights = met.Weights.pancancer_genetic_organotropism_uniform_weighting()
```

#### Custom Weights

| Parameter | Meaning | Notes / Guidelines |
|-----------|---------|---------------------|
| **mig** | Penalizes the total number of migrations. |  |
| **comig** | Penalizes co-migrations. |  |
| **seed_site** | Penalizes the number of seeding sites. | |
| **gen_dist** | Penalizes genetic distance. | Default 0; requires branch lengths (see ``num_mutations`` in Inputs). |
| **organotrop** | Penalizes deviation from organotropism priors. | Default 0; requires organotropism dictionaries. |

**BEST PRACTICE**: Use much higher penalties for the parsimony-related weights (`mig`, `comig`, `seed_site`) than for `gen_dist` or `organotrop`. If using genetic distance and organotropism together, set both `gen_dist > 0` and `organotrop > 0`.

Example usage:
```python
weights = met.Weights(mig=0.5, comig=0.3, seed_site=0.2, gen_dist=0.01)
```

---
### Os (Organotropism Dictionaries)
```python
Os = {
    "Liver": 0.5,
    "Lung": 0.4,
    "Brain": 0.1,
}, # Organotropism dictionary for patient 1
{
    "Lymph": 0.7,
    "Bone": 0.3,
},  # Organotropism dictionary for patient 2
```
- **What it does**: Specifies known frequencies of metastasis to different sites. Must have a frequency for all metastatic sites for all patients to be used.
- **Note**: Values should be normalized (sum to 1)

---

### PrintConfig 
```python
print_config = met.PrintConfig(
    visualize=True,      
    verbose=True,        # Enable for debugging
    k_best_trees=10,     # The number of solutions to visualize (all solutions are saved to a pkl file)
    save_outputs=True,
    custom_colors=None,  # Array of hex strings (with length = number of anatomical sites) to be used as custom colors in visualization
)
```

</details>

<details>
<summary><h2>🔽 Output format</h2></summary>
   
## Outputs
   
Metient will output a pickle file in the specificed output directory for each patient that is inputted. 

In the pickle file you'll find the following keys:
| Pkl key name | Description |
|----------|----------|
| **anatomical_sites** | a list of anatomical sites **in the order** used for the matrices detailed below.| 
| **node_info** | list of dictionaries, in order from best to worst solution. This is solution specific because reolving polytomies can change the tree. Each dictionary maps node index (as used for the matrices detailed below) to a tuple: (label, is_leaf, is_polytomy_resolver_node) used on the tree. The reason labels can be different from what is inputted into Metient is that Metient adds leaf nodes which correspond to the inferred presence of each node in anatomical sites. Each leaf node is labeled as <parent_node_name>_<anatomical_site>. |
| **node_labels** | list of numpy ndarrays, in order from best to worst solution. Each numpy array is a matrix (shape: `len(ordered_anatomical_sites)`, `len(node_info[x])`), where `x` is the `x`th best solution. Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `node_info[x][j][0]`. Each column is a one-hot vector representing the location inferred by Metient for that node. |
| **parents** | list of numpy 1-D arrays, in order from best to worst tree. Each is a an array (shape: `len(node_info[x])`), where `x` is the `x`th best solution. The value at index i is the parent of node i. The root node will have a -1 at its index. | 
| **observed_proportions** | numpy ndarray (shape: `len(ordered_anatomical_sites)`, `num_clusters`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `node_info[x][j][0]`. A value at i,j greater than 0.05 indicates that that node is present in that antomical site. These are the nodes that get added as leaf nodes. |
|**losses** | a list of the losses, from best to worst solution.|
|**probabilities** | a list of the probabilities, from best to worst solution.|
|**primary_site**|str, the name of the anatomical site used as the primary site.|
|**loss_info**| a list of the dicts, from best to worst solution. Each dictionary contains the unweighted components of the loss (e.g. migration number, comigration number, etc.)|

</details>

## Questions

Please email any questions you have to divyakoyy@gmail.com, or open a GitHub issue!

## Citation
If you use Metient, please cite our paper:
```
@article{koyyalagunta2025inferring,
  title={Inferring cancer type-specific patterns of metastatic spread using Metient},
  author={Koyyalagunta, Divya and Ganesh, Karuna and Morris, Quaid},
  journal={bioRxiv},
  pages={2024--07},
  year={2025}
}
```





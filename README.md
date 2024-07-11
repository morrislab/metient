# Metient
<img src="metient/logo.png" width="150">

**Metient** (**MET**astasis + gradi**ENT**) is a tool for migration history inference. You can find our preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.09.602790v1).

Metient is available as a python library, installable via pip. It has been tested on Linux. Installation and running one of the tutorials should take ~5 minutes.

## Installation

```bash
# mamba used for speed, can use conda instead if mamba is not installed
mamba create -n "met" python=3.8.8 ipython
mamba activate met
pip install metient
```

## Tutorial
To run the tutorial notebooks, clone this repo:
```bash
git clone git@github.com:morrislab/metient.git
cd metient/tutorial/
```

There are different Jupyter Notebook tutorials based on your use case:
1. I have a cohort of patients (~5 or more patients) with the same cancer type. (Metient-calibrate)
   - I want Metient to estimate which mutations/mutation clusters are present in which anatomical sites. [Tutorial 1](tutorial/1_metient_calibrate_infer_observed_clones_label_clone_tree_tutorial.ipynb)
   - I know which mutations/mutation clusters are present in which anatomical sites.  [Tutorial 2](tutorial/2_metient_calibrate_label_clone_tree_tutorial.ipynb)
3. I have a small number of patients, or I want to enforce my own parsimony metric weights. (Metient-evaluate)
   - I want Metient to estimate which mutations/mutation clusters are present in which anatomical sites. [Tutorial 3](tutorial/3_metient_evaluate_infer_observed_clones_label_clone_tree_tutorial.ipynb)
   - I know which mutations/mutation clusters are present in which anatomical sites. [Tutorial 4](tutorial/4_metient_evaluate_label_clone_tree_tutorial.ipynb)
  
If your jupyter notebook does not automatically recognize your conda environment, run the following:
```bash
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "met"
```
Then in the jupyter notebook, select Kernel > Change kernel > met.

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

## Outputs

Metient will output a pickle file in the specificed output directory for each patient that is inputted. 

In the pickle file you'll find the following keys:
* `ordered_anatomical_sites`: a list of anatomical sites in the order used for the matrices detailed below.
* `full_tree_node_idx_to_labels`: list of dictionaries, in order from best to worst solution. This is solution specific because reolving polytomies can change the tree. Each dictionary maps node index (as used for the matrices detailed below) to the label used on the tree. The reason this is different from what is inputted is that metient adds leaf nodes which correspond to the inferred presence of each node in anatomical sites. Each leaf node is labeled as <parent_node_name>_<anatomical_site>.
* `clone_tree_labelings`: list of numpy ndarrays, in order from best to worst solution. Each numpy array is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. Each column is a one-hot vector representing the location inferred by Metient for that node.
* `full_adjacency_matrices`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(full_node_idx_to_label.values())`, `len(full_node_idx_to_label.values())`). A 1 at index i,j indicates an edge from i to j.
* `observed_clone_matrix`: numpy ndarray (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. A value at i,j greater than 0.05 indicates that that node is present in that antomical site. These are the nodes that get added as leaf nodes.

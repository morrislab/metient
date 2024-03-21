

## Installation


```bash
# mamba used for speed, can use conda instead if mamba is not installed
mamba create -n "met" python=3.8.8 ipython
mamba activate met
pip install metient
```

## Tutorial
A tutorial for running calibrate mode is available [here](https://github.com/divyakoyy/metient/blob/main/tutorial/metient_calibrate_tutorial.ipynb)


## Inputs
There are two required inputs, a tsv file with the reference and variant counts for each sample and mutation, and a txt file specifying the edges of the clone tree.

### 1. **Tsv file**
Each row in the tsv should correspond to the reference and variant read counts at a single locus in a single tumor sample.  

The required fields for the tsv file:
| Column name | Description |
|----------|----------|
| **anatomical_site_index** | Zero-based index for anatomical_site_label column. Rows with the same anatomical site index and cluster_index will get pooled together.| 
| **anatomical_site_label** | Name of the anatomical site |
| **character_label** | Zero-based index for character_label column |
| **character_label** | Name of the mutation or cluster of mutations. This is used in visualizations, so it should be short. NOTE: due to graphing dependencies, this string cannot contain colons. |
| **cluster_index** | If using a clustering method, the cluster index that this mutation belongs to. NOTE: this must correspond to the indices used in the tree txt file. Rows with the same anatomical site index and cluster_index will get pooled together.|
| **ref** | The number of reads that map to the reference allele for this mutation or mutation cluster in this anatomical site. |
| **var** | The number of reads that map to the variant allele for this mutation or mutation cluster in this anatomical site. |
| **site_category** | Must be one of `primary` or `metastasis`. If multiple primaries are specified, such that the `primary` label is used for multiple different anatomical site indices (i.e., the true primary is not known), we will run Metient multiple times with each primary used as the true primary. Output files are saved with the suffix `_{anatomical_site_label}` to indicate which primary was used in that run. |
| **var_read_prob** | This gives Metient the ability to correct for the effect copy number alterations CNAs) have on the relationship between VAF (i.e., the proportion of alleles bearing the mutation) and subclonal frequency (i.e., the proportion of cells bearing the mutation). Let j = character_index. This is the probabilty of observing a read from the variant allele for mutation at j in a cell bearing the mutation. Thus, if mutation at j occurred at a diploid locus, this should be 0.5. In a haploid cell (e.g., male sex chromosome), this should be 1.0. If a CNA duplicated the reference allele in the lineage bearing mutation j prior to j occurring, there will be two reference alleles and a single variant allele in all cells bearing j, such that var_read_prob = 0.3333. If using a CN caller that reports major and minor CN: `var_read_prob = (p*maj)/(p*(maj+min)+(1-p)*2)`, where `p` is tumor purity, `maj` is major CN, `min` is minor CN, and we're assuming the variant allele has major CN. For more information, see S2.2 of [PairTree's supplementary info](https://aacr.silverchair-cdn.com/aacr/content_public/journal/bloodcancerdiscov/3/3/10.1158_2643-3230.bcd-21-0092/9/bcd-21-0092_supplementary_information_suppsm_sf1-sf21.pdf?Expires=1709221974&Signature=dJH6~Dg-6gEb-S88i0wDGW28QZn16keQj34Vo2tAvJL2cUJrQo48afpHPp-a2zAwQa~ET6SDgw3hb3ITacB06GDUc3GYCdCgYtfPMjFGwygFj-Q9xf-c44VAvwiyliwsBXK1shZmURlFMwSjzkwRwasuWu50sMNmeJSoVyX3nQ-rRBlK93aDR5s9c0l-p4aGvTi6QmfKJPsxXaHB4Lz5yXSl3Xd~JPK-Y~ltC14epDRb~MiSPWUFCAiYetUXcQ7J7vd6b4XQKT9PnYkjQtUq55tLSoUkOGe5JkJ32NXCeoT~l-XD97pCeDYVDOYzAuOkAG0tDYrPebEh2TGTA3fnbA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) for more details. |


        anatomical_site_index    anatomical_site_label    cluster_index    character_label    ref    var     var_read_prob    site_category    num_mutations
        0    breast    0    HER2    982    78    0.43    primary   54 
        0    breast    0    LAMA2   782    0    0.36    primary   15
        0    breast    1    A2BP1   221    0    0.31    primary   8
        1    spinal    0    HER2    897    892    0.53    metastasis   54 
        1    spinal    0    LAMA2   124    101    0.42    metastasis   15
        1    spinal    1    A2BP1   341    89    0.22    metastasis   8
        
### 2. **Tree txt file**
A .txt file where each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv. 

        0   1
        0   2


## Outputs

Metient will output a pickle file in the specificed output directory for each patient that is inputted. 

In the pickle file you'll find the following keys:
* `ordered_anatomical_sites`: a list of anatomical sites in the order used for the matrices detailed below.
* `full_tree_node_idx_to_labels`: list of dictionaries, in order from best to worst tree. Each dictionary maps node index (as used for the matrices detailed below) to the label used on the tree. The reason this is different from what is inputted is that metient adds leaf nodes which correspond to the inferred subclonal presence of each node in anatomical sites. Each tree has a different set of possible leaf nodes, and each leaf node is labeled as <parent_node_name>_<anatomical_site>.
* `ancestral_labelings`: list of numpy ndarrays, in order from best to worst tree. Each numpy array is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. Each column is a one-hot vector representing the location inferred by Metient for that node.
* `full_adjacency_matrices`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(full_node_idx_to_label.values())`, `len(full_node_idx_to_label.values())`). A 1 at index i,j indicates an edge from i to j.
* `subclonal_presence_matrices`: list of numpy ndarrays, in order from best to worst tree. Each tensor is a matrix (shape: `len(ordered_anatomical_sites)`, `len(full_node_idx_to_label.values())`). Row i corresponds to the site at index i in `ordered_anatomical_sites`, and column j corresponds to the node with label `full_node_idx_to_label[j]`. A value at i,j greater than 0.05 indicates that that node is present in that antomical site. These are the nodes that get added as leaf nodes.

### Wrapper API

from metient.lib import vertex_labeling as vert_label
from metient.util import plotting_util as plutil
from metient.util import data_extraction_util as dutil
from metient.util import pairtree_data_extraction_util as ptutil

from metient.util.globals import *


def evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O=None, batch_size=-1, custom_colors=None, bias_weights=True, solve_polytomies=False):
    return vert_label.evaluate(tree_fn, tsv_fn,weights, print_config, output_dir, run_name, 
                               O=O, batch_size=batch_size, custom_colors=custom_colors, 
                               bias_weights=bias_weights, solve_polytomies=solve_polytomies)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, 
              Os=None, batch_size=-1, custom_colors=None, bias_weights=True,  solve_polytomies=False):
    return vert_label.calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, Os=Os, batch_size=batch_size, 
                                custom_colors=custom_colors, bias_weights=bias_weights, solve_polytomies=solve_polytomies)

class PrintConfig:
    def __init__(self, visualize=True, verbose=False, k_best_trees=10, save_outputs=True):
        '''
        visualize: bool, whether to visualize loss, best tree, and migration graph
        verbose: bool, whether to print debug info
        k_best_trees: int, number of best tree solutions to visualize (if 1, only show best tree)
        save_outputs: bool, whether to save pngs and pickle files 
        '''
        self.visualize = visualize
        self.verbose = verbose 
        self.k_best_trees = k_best_trees
        self.save_outputs = save_outputs

class Weights:
    def __init__(self,  mig=10.0, comig=5.0, seed_site=1.0, gen_dist=0.0, organotrop=0.0, data_fit=0.2, reg=5.0, entropy=0.1):
        if not isinstance(mig, list):
            mig = [mig]
        if not isinstance(seed_site, list):
            seed_site = [seed_site]
        self.data_fit = data_fit
        self.mig = mig
        self.comig = comig
        self.seed_site = seed_site
        self.reg = reg
        self.gen_dist = gen_dist
        self.organotrop = organotrop
        self.entropy = entropy

def get_verbose_seeding_pattern(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    returns: one of: {monoclonal, polyclonal} {primary single-source, single-source, multi-source, reseeding}
    '''
    return plutil.get_verbose_seeding_pattern(V, A)

def get_migration_graph(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    '''
    return plutil.get_migration_graph(V, A)


########### Data Extraction Utilities ############

def get_adj_matrices_from_pairtree_results(pairtee_results_filepaths):
    '''
    Args:
        - pairtee_results_filepaths: either a single path or list of paths to pairtree/orchard clone tree
        results (end with ".results.npz")

    Returns:
        If a list of pairtee_results_filepaths is supplied, a list of adjacency matrices is returned
        (otherwise, a single adjacency matrix is returned)

    '''
    return ptutil.get_adj_matrices_from_pairtree_results(pairtee_results_filepaths)

def get_genetic_distance_matrices_from_adj_matrices(adj_matrices, idx_to_character_labels, split_char, normalize=True):
    '''
    Get the genetic distances between nodes by counting the number of mutations between parent and child. 
    
    Args:
        - adj_matrices: either a single adjacency matrix or list of adjacency matrices
        - idx_to_character_labels: either a single dictionary or list of dictionaries.
        idx_to_character_label's keys are expected to be mutation/cluster indices with
        the mutation or mutations in that cluster (e.g. 'ENAM:4:71507837_DLG1:3:196793590'). 
        - split_char: indicates what the mutations in the cluster name are split by. e.g. "_". If
        split_char is None, it is assumed that these are single mutations and not clusters, and the distance
        between each parent and child node in the clone tree is 1. 

    If a single adj_matrix or idx_to_character_label is inputted, a single genetic distance matrix is returned. 
    Otherwise, a list of genetic distance matrices is returned.
    '''
    return dutil.get_genetic_distance_matrices_from_adj_matrices(adj_matrices, idx_to_character_labels, split_char, normalize=normalize)

def get_organotropism_matrix_from_msk_met(ordered_sites, cancer_type, frequency_csv, site_to_msk_met_map=None):
    '''
    Args:
        - ordered_sites: array of the anatomical site names (e.g. ["breast", "lung"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        - cancer_type: cancer type name, which is used as the 
        - frequency_csv: csv with frequency of metastasis by cancer type
        - site_to_msk_met_map: dictionary mapping site names to MSK-MET site names. if
        not provided, the names used in the ordered_sites array are used

    Returns:
        array of size len(ordered_sites) with the frequency with which
        primary cancer type seeds site i
    '''
    return dutil.get_organotropism_matrix_from_msk_met(ordered_sites, cancer_type, frequency_csv, site_to_msk_met_map=site_to_msk_met_map)

def get_adj_matrices_from_spruce_mutation_trees(mut_trees_filename, idx_to_character_label, is_sim_data=False):
    '''
    When running MACHINA's generatemutationtrees executable, it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of tuples, each containing (T, pruned_idx_to_character_label) for each
    tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - idx_to_character_label: a dict mapping indices of the adj matrix T to character
        labels 
    '''
    return dutil.get_adj_matrices_from_spruce_mutation_trees(mut_trees_filename, idx_to_character_label, is_sim_data=is_sim_data)

def get_idx_to_cluster_label(cluster_filepath, ignore_polytomies):
    '''
    Args:
	    cluster_filepath: path to cluster file for MACHINA simulated data in the format:
	    0
	    1
	    3;15;17;22;24;29;32;34;53;56
	    69;78;80;81

    	where each semi-colon separated number represents a mutation that belongs
    	to cluster i where i is the file line number.

	    ignore_polytomies: whether to include resolved polytomies (which were found by
	    running PMH-TR) in the returned dictionary

    Returns:
        (1) a dictionary mapping cluster number to cluster name
        for e.g. for the file above, this would return:
            {0: '0', 1: '69;78;80;81'}
    '''
    return dutil.get_idx_to_cluster_label(cluster_filepath, ignore_polytomies)

def get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_idx_to_cluster_label, T):
    '''
    tsv_filepath: path to tsv for machina simulated data (generated from create_conf_intervals_from_reads.py)

    tsv is expected to have columns: ['#sample_index', 'sample_label', 'anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'f_lb', 'f_ub', 'ref', 'var']

    pruned_idx_to_cluster_label:  dictionary mapping the cluster index to label, where 
    index corresponds to col index in the R matrix and V matrix returned. This isn't 1:1 
    with the 'character_label' to 'character_index' mapping in the tsv because we only keep the
    nodes which appear in the mutation tree, and re-index after removing unseen nodes
    (see _get_adj_matrix_from_machina_tree)

    T: adjacency matrix of the internal nodes.

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data
    '''
    return dutil.get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_idx_to_cluster_label, T)

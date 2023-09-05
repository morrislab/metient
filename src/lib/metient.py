### Wrapper API

from src.lib import vertex_labeling as vert_label
from src.util import plotting_util as plutil
from src.util import data_extraction_util as dutil

def get_migration_history(T, ref, var, ordered_sites, primary_site, node_idx_to_label,
                          weights, print_config, output_dir, run_name, 
                          G=None, O=None, max_iter=200, lr=0.1, init_temp=40, final_temp=0.01,
                          batch_size=64, custom_colors=None, weight_init_primary=False, lr_sched="step"):
    '''
    Args:
        T: numpy ndarray or torch tensor (shape: num_internal_nodes x num_internal_nodes). Adjacency matrix (directed) of the internal nodes.
        
        ref: numpy ndarray or torch tensor (shape: num_anatomical_sites x num_mutation_clusters). Reference matrix, i.e., num. reads that map to reference allele
        
        var: numpy ndarray or torch tensor (shape:  num_anatomical_sites x num_mutation_clusters). Variant matrix, i.e., num. reads that map to variant allele
        
        ordered_sites: list of the anatomical site names (e.g. ["breast", "lung_met"]) with length =  num_anatomical_sites) and 
        the order matches the order of sites in the ref and var
        
        primary_site: name of the primary site (must be an element of ordered_sites)

        weights: Weight object for how much to penalize each component of the loss
        
        print_config: PrintConfig object with options on how to visualize output
        
        node_idx_to_label: dictionary mapping vertex indices (corresponding to their index in T) to custom labels
        for plotting
        
        output_dir: path for where to save output trees to

        run_name: e.g. patient name, used for naming output files.

    Optional:
        G: numpy ndarray or torch tensor (shape: num_internal_nodes x num_internal_nodes).
        Matrix of genetic distances between internal nodes.
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        
        O: numpy ndarray or torch tensor (shape: num_anatomical_sites x  num_anatomical_sites).
        Matrix of organotropism values between sites.

        weight_init_primary: whether to initialize weights higher to favor vertex labeling of primary for all internal nodes
        
        lr_sched: how to weight the tasks of (1) leaf node inference and (2) internal vertex labeling. options:
            "bi-level": outer objective is to learn vertex labeling, inner objective is to learn leaf nodes (subclonal presence)
            "constant": default, (1) and (2) weighted equally at each epoch
            "step": (1) has weight=1 for first half of epochs and (2) has weight=0, and then we flip the weights for the last half of training
            "em": (1) has weight=1 and (2) has weight=0 for 20 epochs, and we flip every 20 epochs (kind of like E-M)
            "linear": (1) decreases linearly while (2) increases linearly at the same rate (1/max_iter)
    Returns:
        # TODO: return info for k best trees
        Corresponding info on the *best* tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
        (4) dictionary w/ loss values for each component of the loss
        (5) how long (in seconds) the algorithm took to run
    '''


    return vert_label.get_migration_history(T, ref, var, ordered_sites, primary_site, node_idx_to_label,
					                        weights, print_config, output_dir, run_name, 
					                        G=G, O=O, max_iter=max_iter, lr=lr, init_temp=init_temp, final_temp=final_temp,
					                        batch_size=batch_size, custom_colors=custom_colors, weight_init_primary=weight_init_primary, lr_sched=lr_sched)

class PrintConfig:
    def __init__(self, visualize=True, verbose=False, viz_intermeds=False, k_best_trees=1, save_imgs=False):
        '''
        visualize: bool, whether to visualize loss, best tree, and migration graph
        verbose: bool, whether to print debug info
        viz_intermeds: bool, whether to visualize intermediate solutions to best tree
        k_best_trees: int, number of best tree solutions to visualize (if 1, only show best tree)
        '''
        self.visualize = visualize
        self.verbose = verbose 
        self.viz_intermeds = viz_intermeds
        self.k_best_trees = k_best_trees
        self.save_imgs = save_imgs

class Weights:
    def __init__(self, data_fit=0.2, mig=10.0, comig=7.0, seed_site=5.0, reg=2.0, gen_dist=0.0, organotrop=0.0):
        self.data_fit = data_fit
        self.mig = mig
        self.comig = comig
        self.seed_site = seed_site
        self.reg = reg
        self.gen_dist = gen_dist
        self.organotrop = organotrop




########### Data Extraction Utilities ############

def get_ref_var_matrices(tsv_filepath):
    '''
    tsv_filepath: path to tsv with columns:

    ['#sample_index', 'sample_label', '#anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'ref', 'var']

    returns
    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
    (3) unique anatomical sites from the patient's data,
    (4) dictionary mapping index to character_label (based on input tsv, gives the index for each mutation name,
    where these indices are used in R matrix, V matrix
    '''
    return dutil.get_ref_var_matrices(tsv_filepath)

def get_genetic_distance_matrix_from_adj_matrix(adj_matrix, idx_to_character_label, split_char, normalize=True):
    '''
    Get the genetic distances between nodes by counting the number of mutations between
    parent and child. idx_to_character_label's keys are expected to be mutation/cluster indices with
    the mutation or mutations in that cluster (e.g. 'ENAM:4:71507837_DLG1:3:196793590'). split_char
    indicates what the mutations in the cluster name are split by (if it's a cluster).
    '''
    return dutil.get_genetic_distance_matrix_from_adj_matrix(adj_matrix, idx_to_character_label, split_char, normalize=normalize)

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

def get_organotropism_matrix_from_msk_met(ordered_sites, cancer_type, frequency_csv, site_to_msk_met_map=None):
    '''
    Args:
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        cancer_type: cancer type name, which is used as the 
        frequency_csv: csv with frequency of metastasis by cancer type
        site_to_msk_met_map: dictionary mapping site names to MSK-MET site names. if
        not provided, the names used in the ordered_sites array are used

    Returns:
        array of size len(ordered_sites) with the frequency with which
        primary cancer type seeds site i
    '''
    return dutil.get_organotropism_matrix_from_msk_met(ordered_sites, cancer_type, frequency_csv, site_to_msk_met_map=site_to_msk_met_map)

def get_cluster_label_to_idx(cluster_filepath, ignore_polytomies):
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
	    (1) a dictionary mapping cluster name to cluster number
	    for e.g. for the file above, this would return:
	        {'0': 0, '1': 1, '3;15;17;22;24;29;32;34;53;56': 2, '69;78;80;81': 3}
    '''
    dutil.get_cluster_label_to_idx(cluster_filepath, ignore_polytomies)

def get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_cluster_label_to_idx, T):
    '''
    tsv_filepath: path to tsv for machina simulated data (generated from create_conf_intervals_from_reads.py)

    tsv is expected to have columns: ['#sample_index', 'sample_label', 'anatomical_site_index',
    'anatomical_site_label', 'character_index', 'character_label', 'f_lb', 'f_ub', 'ref', 'var']

    pruned_cluster_label_to_idx:  dictionary mapping the cluster label to index which corresponds to
    col index in the R matrix and V matrix returned. This isn't 1:1 with the
    'character_label' to 'character_index' mapping in the tsv because we only keep the
    nodes which appear in the mutation tree, and re-index after removing unseen nodes
    (see _get_adj_matrix_from_machina_tree)

    T: adjacency matrix of the internal nodes.

    Returns:
	    (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
	    (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
	    (3) unique anatomical sites from the patient's data
    '''
    dutil.get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_cluster_label_to_idx, T)

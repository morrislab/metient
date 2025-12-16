### Wrapper API

from metient.lib import migration_history_inference as mig_hist
from metient.util import plotting_util as plutil
from metient.util import data_extraction_util as dutil

def evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O=None, sample_size=-1, solve_polytomies=False, num_runs=3):
    '''
    Runs Metient-evaluate, and infers the observed clone percentages and the labels of the clone tree.

    Args: 
        REQUIRED:
        tree_fn: Path to .txt file where each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv.
        tsv_fn: Path to .tsv file where each row in this tsv should correspond to a single mutation/mutation cluster in a single tumor sample. Required columns:
                anatomical_site_index, anatomical_site_label, cluster_index, character_index, character_label, ref, var, var_read_prob, site_category
        weights: Weights object which specifies the relative weighting to place on each part of the objective
        print_config: PrintConfig object which specifies saving/visualization configuration
        output_dir: Path for where to save outputs
        run_name: Name for this patient which will be used to save all outputs.

        OPTIONAL:
        O: a dictionary mapping anatomical site name (as used in tsv_fn) -> frequency of metastasis (these values should be normalized)
        sample_size: how many samples to have Metient solve in parallel
        solve_polytomies: bool, whether or not to resolve polytomies 

    Outputs migration history inferences for a single patient.
    '''
    return mig_hist.evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
                             O=O, sample_size=sample_size, solve_polytomies=solve_polytomies, 
                             num_runs=num_runs, bias_weights=True)

def evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
                              O=None, sample_size=-1, solve_polytomies=False, num_runs=3):
    '''
    Runs Metient-evaluate with observed clone percentages inputted, and only inferring the labels of the clone tree.

    Args: 
        REQUIRED:
        tree_fn: Path to .txt file where each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv.
        tsv_fn: Path to .tsv file where each row in this tsv should correspond to a single mutation/mutation cluster in a single tumor sample. Required columns:
                anatomical_site_index, anatomical_site_label, cluster_index, cluster_label, present, site_category, num_mutations
        weights: Weights object which specifies the relative weighting to place on each part of the objective
        print_config: PrintConfig object which specifies saving/visualization configuration
        output_dir: Path for where to save outputs
        run_name: Name for this patient which will be used to save all outputs.

        OPTIONAL:
        O: a dictionary mapping anatomical site name (as used in tsv_fn) -> frequency of metastasis (these values should be normalized)
        sample_size: how many samples to have Metient solve in parallel
        solve_polytomies: bool, whether or not to resolve polytomies 
    
    Outputs migration history inferences for a single patient.
    '''
    return mig_hist.evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
                                              O=O, sample_size=sample_size, bias_weights=True, 
                                              solve_polytomies=solve_polytomies, num_runs=num_runs)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type, 
              Os=None, sample_size=-1, solve_polytomies=False, num_runs=3):
    '''
    Runs Metient-calibrate on a cohort of patients. For each patient, we infer the observed clone percentages and the labels of the clone tree.

    Args: 
        REQUIRED:
        tree_fns: List of paths to .txt files. In each .txt file, each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv.
        tsv_fns: List of paths to .tsv files. In each .tsv file, each row in this tsv should correspond to a single mutation/mutation cluster in a single tumor sample. Required columns:
                anatomical_site_index, anatomical_site_label, cluster_index, character_index, character_label, ref, var, var_read_prob, site_category
        print_config: PrintConfig object which specifies saving/visualization configuration
        output_dir: Path for where to save outputs
        run_names: List of patient names which will be used to save all outputs.
        calibration_type: str, one of ["genetic", "organotropism", "both"] specifying which type of calibration to perform

        NOTE: tree_fns[i] and tsv_fns[i] and run_names[i] all correspond to patient i.

        OPTIONAL:
        Os: a list of dictionaries mapping anatomical site name (as used in tsv_fn) -> 
            frequency of metastasis (these values should be normalized), Os[i] corresponds to patient i.
        sample_size: how many samples to have Metient solve in parallel
        solve_polytomies: bool, whether or not to resolve polytomies 
    '''
    return mig_hist.calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, 
                              calibration_type, Os=Os, sample_size=sample_size, 
                              bias_weights=True, solve_polytomies=solve_polytomies, num_runs=num_runs)

def calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, 
                               calibration_type, Os=None, sample_size=-1, solve_polytomies=False, num_runs=3):
    '''
    Runs Metient-calibrate on a cohort of patients. For each patient, we use the inputted observed clone percentages, and only infer the labels of the clone tree.

    Args: 
        REQUIRED:
        tree_fns: List of paths to .txt files. In each .txt file, each line is an edge from the first index to the second index. Must correspond to the cluster_index column in the input tsv.
        tsv_fns: List of paths to .tsv files. In each .tsv file, each row in this tsv should correspond to a single mutation/mutation cluster in a single tumor sample. Required columns:
                anatomical_site_index, anatomical_site_label, cluster_index, cluster_label, present, site_category, num_mutations
        print_config: PrintConfig object which specifies saving/visualization configuration
        output_dir: Path for where to save outputs
        run_names: List of patient names which will be used to save all outputs.
        calibration_type: str, one of ["genetic", "organotropism", "both"] specifying which type of calibration to perform

        NOTE: tree_fns[i] and tsv_fns[i] and run_names[i] all correspond to patient i.

        OPTIONAL:
        Os: a list of dictionaries mapping anatomical site name (as used in tsv_fn) -> 
            frequency of metastasis (these values should be normalized), Os[i] corresponds to patient i.
        sample_size: how many samples to have Metient solve in parallel
        solve_polytomies: bool, whether or not to resolve polytomies 
    '''
    return mig_hist.calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
                                              Os=Os, sample_size=sample_size, bias_weights=True, 
                                              solve_polytomies=solve_polytomies, num_runs=num_runs)


class PrintConfig:
    def __init__(self, visualize=True, verbose=False, k_best_trees=float("inf"), 
                 save_outputs=True, custom_colors=None, display_labels=True):
        '''
        Args:
            visualize: bool, whether to visualize loss, best tree, and migration graph
            verbose: bool, whether to print debug info
            k_best_trees: int, number of best tree solutions to visualize (if 1, only show best tree)
            save_outputs: bool, whether to save pngs and pickle files
            custom_colors: array of hex strings (with length = number of anatomical sites) to be used as custom colors in output visualizations
            display_labels: bool, whether to display node labels on the migration history tree
        '''
        if k_best_trees <= 0:
            raise ValueError("k_best_trees must be >= 1")
        self.visualize = visualize
        self.verbose = verbose 
        self.k_best_trees = k_best_trees
        self.save_outputs = save_outputs
        self.custom_colors = custom_colors
        self.display_labels = display_labels

class Weights:
    """Weight configuration for Metient models"""
    
    @classmethod
    def pancancer_genetic_uniform_weighting(cls):
        """Default weights for genetic-only model (no tissue tropism),
            using uniform cohort weighting
        """
        return cls(
            mig=0.5448,
            comig=0.2727,
            seed_site=0.1825,
        )
    
    @classmethod
    def pancancer_genetic_cohort_size_weighting(cls):
        """Default weights for genetic-only model (no tissue tropism),
            using weighting by cohort size
        """
        return cls(
            mig=0.5398,
            comig=0.2837,
            seed_site=0.1764,
        )

    @classmethod
    def pancancer_genetic_organotropism_uniform_weighting(cls):
        """Default weights for combined genetic and tissue tropism model"""
        return cls(
            mig=0.5437,
            comig=0.2712,
            seed_site=0.1852,
        )
    
    @classmethod
    def pancancer_genetic_organotropism_cohort_size_weighting(cls):
        """Default weights for combined genetic and tissue tropism model,
            using weighting by cohort size
        """
        return cls(
            mig=0.5363,
            comig=0.2848,
            seed_site=0.1789,
        )

    def __init__(self, mig, comig, seed_site, gen_dist=0.0, 
                 organotrop=0.0, data_fit=15.0, reg=0.5, entropy=0.0001):
        '''
        The higher the inputted weight, the higher the penalty on that metric.

        Args:
            mig: weight to place on migration number. Default is based on calibration to real data.
            comig: weight to place on comigration number. Default is based on calibration to real data.
            seed_site: weight to place on seeding site number. Default is based on calibration to real data.
            gen_dist: weight to place on genetic distance loss.
            organotrop: weight to place on organotropism loss.
            data_fit: weight to place on negative log likelihood loss of observed clone percentages.
            reg: weight to place on regularization loss for observed clone percentages.
            entropy: weight to place on negative entropy.
        '''
        self.data_fit = data_fit
        self.mig = mig
        self.comig = comig
        self.seed_site = seed_site
        self.reg = reg
        self.gen_dist = gen_dist
        self.organotrop = organotrop
        self.entropy = entropy

def migration_graph(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    '''
    return plutil.migration_graph(V, A)

def seeding_pattern(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns: verbal description of the seeding pattern, one of:
    {monoclonal, polyclonal} {single-source, multi-source, reseeding}
    '''
    return plutil.seeding_pattern(V, A)

def phyleticity(V, A, node_info):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    node_info: Dictionary mapping node indices to a tuple (label (str), is_witness_node (bool), is_polytomy_resolver_node (bool)). 
               This is outputted in Metient's pkl.gz file

    After determining which nodes perform seeding (i.e., nodes which have a different
    color from their parent), if all nodes can be reached from the highest level node 
    in the seeding clusters (closest to root), returns monophyletic, else polyphyletic
    '''
    return plutil.phyleticity(V, A, node_info)

def seeding_clusters(V, A, node_info):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    node_info: Dictionary mapping node indices to a tuple (label (str), is_witness_node (bool), is_polytomy_resolver_node (bool)). 
               This is outputted in Metient's pkl.gz file

    Returns: list of nodes whose parent is a different color than itself
    '''
    return plutil.seeding_clusters(V, A, node_info)

def site_clonality(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    '''
    return plutil.site_clonality(V, A)

def genetic_clonality(V, A, node_info):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
    node_info: Dictionary mapping node indices to a tuple (label (str), is_witness_node (bool), is_polytomy_resolver_node (bool)). 
               This is outputted in Metient's pkl.gz file

    Returns monoclonal if every site is seeded by the *same* clone,
    else returns polyclonal.
    '''
    return plutil.genetic_clonality(V, A, node_info)


def adjacency_matrix_from_parents(parents):
    """
    Convert parents vector to sparse adjacency matrix
    
    Args:
        parents: numpy array where index i contains the parent node of node i,
                with -1 indicating a root node
                
    Returns:
        torch.sparse_coo_tensor: Sparse adjacency matrix where entry (i,j)=1 
                                indicates i is the parent of j
    """
    return dutil.adjacency_matrix_from_parents(parents)


def weighted_phyleticity(pkl, sites=None):
    """
    Calculate weighted phyleticity classification across all solutions.
    Args:
        pkl (dict): Pickle file containing tree data
        sites (list[str] | None, optional): List of anatomical site names to restrict phyleticity classification to.
    """
    return plutil.weighted_phyleticity(pkl, sites=sites)

def weighted_genetic_clonality(pkl):
    """
    Calculate weighted genetic clonality classification across all solutions.
    Args:
        pkl (dict): Pickle file containing tree data
    """
    return plutil.weighted_genetic_clonality(pkl)   

def weighted_site_clonality(pkl):
    """
    Calculate weighted site clonality classification across all solutions.
    Args:
        pkl (dict): Pickle file containing tree data
    """
    return plutil.weighted_site_clonality(pkl)  

def weighted_seeding_pattern(pkl):
    """
    Calculate weighted seeding pattern classification across all solutions.
    Args:
        pkl (dict): Pickle file containing tree data
    """
    return plutil.weighted_seeding_pattern(pkl)
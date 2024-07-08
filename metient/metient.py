### Wrapper API

from metient.lib import migration_history_inference as mig_hist
from metient.util import plotting_util as plutil

def evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O=None, batch_size=-1, custom_colors=None, solve_polytomies=False):
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
        O: a 1 x n array (where n is number of anatomical sites) if using organotropism
        batch_size: how many samples to have Metient solve in parallel
        custom_colors: an array of hex strings (with length = number of anatomical sites) to be used as custom colors in output visualizations.
        solve_polytomies: bool, whether or not to resolve polytomies 

    Outputs migration history inferences for a single patient.
    '''
    return mig_hist.evaluate(tree_fn, tsv_fn,weights, print_config, output_dir, run_name, 
                             O=O, batch_size=batch_size, custom_colors=custom_colors, 
                             bias_weights=True, solve_polytomies=solve_polytomies)

def evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
                               O=None, batch_size=-1, custom_colors=None, solve_polytomies=False):
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
        O: a 1 x n array (where n is number of anatomical sites) if using organotropism
        batch_size: how many samples to have Metient solve in parallel
        custom_colors: an array of hex strings (with length = number of anatomical sites) to be used as custom colors in output visualizations.
        solve_polytomies: bool, whether or not to resolve polytomies 
    
    Outputs migration history inferences for a single patient.
    '''
    return mig_hist.evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, O=O, batch_size=batch_size, 
                                              custom_colors=custom_colors, bias_weights=True, solve_polytomies=solve_polytomies)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, 
              Os=None, batch_size=-1, custom_colors=None, solve_polytomies=False):
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

        NOTE: tree_fns[i] and tsv_fns[i] and run_names[i] all correspond to patient i.

        OPTIONAL:
        O: a 1 x n array (where n is number of anatomical sites) if using organotropism
        batch_size: how many samples to have Metient solve in parallel
        custom_colors: an array of hex strings (with length = number of anatomical sites) to be used as custom colors in output visualizations.
        solve_polytomies: bool, whether or not to resolve polytomies 

    Outputs migration history inferences for a full cohort.
    '''
    return mig_hist.calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, Os=Os, batch_size=batch_size, 
                              custom_colors=custom_colors, bias_weights=True, solve_polytomies=solve_polytomies)

def calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, 
                               Os=None, batch_size=-1, custom_colors=None,  solve_polytomies=False):
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

        NOTE: tree_fns[i] and tsv_fns[i] and run_names[i] all correspond to patient i.

        OPTIONAL:
        O: a 1 x n array (where n is number of anatomical sites) if using organotropism
        batch_size: how many samples to have Metient solve in parallel
        custom_colors: an array of hex strings (with length = number of anatomical sites) to be used as custom colors in output visualizations.
        solve_polytomies: bool, whether or not to resolve polytomies 

    Outputs migration history inferences for a full cohort.
    '''
    return mig_hist.calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, Os=Os, batch_size=batch_size, 
                                               custom_colors=custom_colors, bias_weights=True, solve_polytomies=solve_polytomies)


class PrintConfig:
    def __init__(self, visualize=True, verbose=False, k_best_trees=1000, save_outputs=True):
        '''
        Args:
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
    def __init__(self, mig=4.8, comig=3.0, seed_site=2.2, gen_dist=0.0, organotrop=0.0, data_fit=15.0, reg=0.5, entropy=0.01):
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

def phyleticity(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    After determining which nodes perform seeding (i.e., nodes which have a different
    color from their child), if all nodes can be reached from the highest level node 
    in the seeding clusters (closest to root), returns monophyletic, else polyphyletic
    '''
    return plutil.phyleticity(V, A)

def seeding_clusters(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns: list of nodes whose child is a different color than itself
    '''
    return plutil.seeding_clusters(V, A)

def site_clonality(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns monoclonal if every site is seeded by one clone,
    else returns polyclonal.
    '''
    return plutil.site_clonality(V, A)

def genetic_clonality(V, A):
    '''
    V: Vertex labeling matrix where columns are one-hot vectors representing the
    anatomical site that the node originated from (num_sites x num_nodes)
    A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)

    Returns monoclonal if every site is seeded by the *same* clone,
    else returns polyclonal.
    '''
    return plutil.genetic_clonality(V, A)
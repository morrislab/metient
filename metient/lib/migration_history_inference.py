import torch
import datetime
import os
import copy
import shutil
import pickle
import gzip
import json

from metient import metient as met
from metient.util import vertex_labeling_util as vutil
from metient.util import data_extraction_util as dutil
from metient.util import eval_util as eutil
from metient.util import plotting_util as putil
from metient.lib import polytomy_resolver as prutil
from metient.lib import v_optimizer as voptim
from metient.lib import u_optimizer as uoptim

from metient.util.globals import *

torch.set_printoptions(precision=2)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def prune_histories(solutions):
    '''
    Only keep the unique, Pareto front of trees
    '''

    # Collect each unique solution's parsimony metrics
    all_pars_metrics = []
    unique_labelings = set()
    final_solutions = []
    for soln in solutions:
        tree = vutil.LabeledTree(soln.T, soln.V)
        if tree not in unique_labelings:
            final_solutions.append(soln)
            unique_labelings.add(tree)
            all_pars_metrics.append((soln.m, soln.c, soln.s))
    # Keep the pareto front of trees
    pareto_metrics, pruned_solutions = vutil.pareto_front(final_solutions, all_pars_metrics)
    print("pareto_metrics", set(pareto_metrics))

    return pruned_solutions

def rank_solutions(solution_set, print_config, needs_pruning=True):
    '''
    Return the sorted, top k solutions
    '''        
    # 1. Only keep unique, Pareto optimal histories
    if needs_pruning:
        final_solutions = prune_histories(solution_set)
    else:
        final_solutions = solution_set

    # 2. Sort the solutions from lowest to highest loss
    final_solutions = sorted(list(final_solutions))

    # 3. Return the k best solutions
    k = print_config.k_best_trees if len(final_solutions) >= print_config.k_best_trees else len(final_solutions)
    final_solutions = final_solutions[:k]
    return final_solutions

def create_solution_set(best_Vs, best_soft_Vs, best_Ts, G, O, p, node_idx_to_label, weights):
    # Make a solution set
    losses, (ms,cs,ss) = vutil.clone_tree_labeling_objective(vutil.to_tensor(best_Vs), vutil.to_tensor(best_soft_Vs), vutil.to_tensor(best_Ts), torch.stack([G for _ in range(len(best_Vs))]), O, p, weights, True)
    solution_set = []
    # Did we find a solution with primary-only single source seeding?
    has_pss_solution = False
    for loss,m,c,s,V,soft_V,T in zip(losses,ms,cs,ss,best_Vs,best_soft_Vs,best_Ts):
        if s == 1:
            has_pss_solution = True
        soln = vutil.VertexLabelingSolution(loss, m, c, s, V, soft_V, T, G, node_idx_to_label)
        solution_set.append(soln)
    return solution_set, has_pss_solution

def get_best_final_solutions(results, G, O, p, weights, print_config, 
                             node_idx_to_label, num_internal_nodes, needs_pruning):
    '''
    Prune unecessary poly nodes (if they weren't used) and return the top k solutions
    '''

    multiresult_has_pss_solution = False
    full_solution_set = []
    unique_labelings = set()

    for result in results:
        
        best_Vs, best_soft_Vs, best_Ts, poly_res = result
        solution_set, has_pss_solution = create_solution_set(best_Vs, best_soft_Vs, best_Ts, G, O, p, node_idx_to_label, weights)
        multiresult_has_pss_solution = multiresult_has_pss_solution or has_pss_solution

        # 1. Make solutions unique before we do all this additional post-processing work which is time intensive
        unique_solution_set = []
        for soln in solution_set:
            tree = vutil.LabeledTree(soln.T, soln.V)
            if tree not in unique_labelings:
                unique_solution_set.append(soln)
                unique_labelings.add(tree)
        # Don't need to recover a primary-only single source solution if we have already found one
        if not multiresult_has_pss_solution:
            unique_solution_set = recover_prim_ss_solutions(unique_solution_set, unique_labelings, weights, O, p)
            multiresult_has_pss_solution = True
        
        #  Remove any extra resolver nodes that don't actually help
        unique_solution_set = prutil.remove_extra_resolver_nodes(unique_solution_set, num_internal_nodes, poly_res, weights, O, p)
        full_solution_set.extend(unique_solution_set)

    return rank_solutions(full_solution_set, print_config, needs_pruning=needs_pruning)

def recover_prim_ss_solutions(solution_set, unique_labelings, weights, O, p):
    '''
    In hard (i.e. usually large input) cases where we are unable to find a 
    primary-only seeding solution, see if we can recover one by post-processing
    final solutions and removing any met-to-met migration edges, and add these
    to our final solution set
    '''
    
    expanded_solution_set = []
    for solution in solution_set:
        expanded_solution_set.append(solution)
        clusters = putil.seeding_clusters(solution.V,solution.T)
        new_V = copy.deepcopy(solution.V)
        for s in clusters:
            new_V[:,s] = p.T 
        loss, (m,c,s) = vutil.clone_tree_labeling_objective(new_V, solution.soft_V, solution.T, solution.G, O, p, weights, True)
        new_solution = vutil.VertexLabelingSolution(loss, m, c, s, new_V, solution.soft_V, solution.T, solution.G, solution.idx_to_label)
        unique_labeled_tree = vutil.LabeledTree(solution.T, new_V)
        if unique_labeled_tree not in unique_labelings:
            expanded_solution_set.append(new_solution)
            unique_labelings.add(unique_labeled_tree)
    return expanded_solution_set

def prep_inputs(tree_fns, tsv_fns, run_names, estimate_observed_clones, output_dir):

    if not (len(tree_fns) == len(tsv_fns) == len(run_names)):
        raise ValueError("Inputs Ts, tsv_fns, and run_names must have equal length (length = number of patients in cohort")

    if isinstance(tree_fns[0], str):
        Ts = []
        for tree_fn in tree_fns:
            Ts.append(dutil.get_adjacency_matrix_from_txt_edge_list(tree_fn))
    else:
        Ts = tree_fns
    
    # If we're not estimating the observed clones, the tsv being inputted doesn't need to be pooled
    if estimate_observed_clones:
        pooled_tsv_fns = []
        for tsv_fn, run_name in zip(tsv_fns, run_names):
            pooled_tsv_fns.append(dutil.pool_input_tsv(tsv_fn, output_dir, f"tmp_{run_name}"))
    else:
        for tsv_fn in tsv_fns:
            dutil.validate_prepooled_tsv(tsv_fn)
        pooled_tsv_fns = tsv_fns
    
    return Ts, pooled_tsv_fns

def evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O, batch_size, custom_colors, bias_weights, solve_polytomies):
    '''
    Observed clone proportions are inputted (in tsv_fns), only labeling of clone tree is needed
    '''
    return evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name,
                    O, batch_size, custom_colors, bias_weights, solve_polytomies, estimate_observed_clones=False)

def evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O, batch_size, custom_colors, bias_weights, solve_polytomies, estimate_observed_clones=True):
    
    Ts, pooled_tsv_fns = prep_inputs([tree_fn], [tsv_fn], [run_name], estimate_observed_clones, output_dir)
    assert isinstance(weights.mig, (float, int)), "Weights must be either a float or an int in evaluate mode"
    assert isinstance(weights.seed_site, (float, int)), "Weights must be either a float or an int in evaluate mode"

    T, pooled_tsv_fn = Ts[0], pooled_tsv_fns[0]  
    print(pooled_tsv_fn)  
    primary_sites = dutil.get_primary_sites(pooled_tsv_fn)
    if len(primary_sites) > 1:
        print("Multiple primaries given. Running each as primary")

    for primary_site in primary_sites:
        infer_migration_history(T, pooled_tsv_fn, primary_site, weights, print_config, output_dir, f"{run_name}_{primary_site}", 
                              O=O, batch_size=batch_size, custom_colors=custom_colors, bias_weights=bias_weights, 
                              mode="evaluate", solve_polytomies=solve_polytomies, estimate_observed_clones=estimate_observed_clones)
    if estimate_observed_clones:
        os.remove(pooled_tsv_fn) # cleanup pooled tsv

def calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names,
                               Os, batch_size, custom_colors, bias_weights, solve_polytomies):
    '''
    Observed clone proportions are inputted (in tsv_fns), only labeling of clone tree is needed
    '''
    return calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names,
                    Os, batch_size, custom_colors, bias_weights, solve_polytomies, estimate_observed_clones=False)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names,
              Os, batch_size, custom_colors, bias_weights, solve_polytomies,
              estimate_observed_clones=True):
    '''
    Estimate observed clone proportions and labeling of clone tree for a cohort of patients,
    calibrate parsimony weights to metastasis priors
    '''
    
    Ts, pooled_tsv_fns = prep_inputs(tree_fns, tsv_fns, run_names, estimate_observed_clones, output_dir)

    # Only use maximum parsimony metrics when initially searching for high likelihood trees
    weights = met.Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                          seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, gen_dist=0.0, 
                          organotrop=0.0)
    # Don't spend time making visualizations for calibrated trees
    visualize = print_config.visualize
    input_k = print_config.k_best_trees
    print_config.visualize = False

    calibrate_dir = os.path.join(output_dir, "calibrate")

    print(f"Saving results to {calibrate_dir}")

    if os.path.exists(calibrate_dir):
        shutil.rmtree(calibrate_dir)
        print(f"Overwriting existing directory at {calibrate_dir}")
    
    os.makedirs(calibrate_dir)

    # 1. Go through each patient and get migration history in calibrate mode
    for i in range(len(Ts)):
        print("\n*** Calibrating for patient:", run_names[i], "***")
        O = Os[i] if Os != None else None

        primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        if len(primary_sites) > 1:
            print("Multiple primaries given. Running each as primary")

        for primary_site in primary_sites:
            infer_migration_history(Ts[i], pooled_tsv_fns[i], primary_site, weights, print_config, calibrate_dir, f"{run_names[i]}_{primary_site}", 
                                    O=O, batch_size=batch_size, custom_colors=custom_colors, bias_weights=bias_weights,
                                    mode="calibrate", solve_polytomies=solve_polytomies, estimate_observed_clones=estimate_observed_clones)

    # 2. Find the best theta for this cohort
    best_theta = eutil.get_max_cross_ent_thetas(pickle_file_dirs=[calibrate_dir])
    rounded_best_theta = [round(v,3) for v in best_theta]
    with open(os.path.join(calibrate_dir, "best_theta.json"), 'w') as json_file:
        json.dump(rounded_best_theta, json_file, indent=2)

    ordered_sites = dutil.extract_ordered_sites(pooled_tsv_fns)
        
    # 3. Recalibrate trees using the best thetas
    print_config.visualize = visualize
    print_config.k_best_trees = input_k
    cal_weights = met.Weights(mig=[best_theta[0]*50], comig=best_theta[1]*50, seed_site=[best_theta[2]*50],
                          gen_dist=1.0, organotrop=1.0)
    
    # 4. Use the saved trees to rescore trees, visualize, and re-save 
    for i in range(len(Ts)):
        O = Os[i] if Os != None else None
        primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        for primary_site in primary_sites:
            run_name = f"{run_names[i]}_{primary_site}"
            with gzip.open(os.path.join(calibrate_dir, f"{run_name}.pkl.gz"), 'rb') as f:
                pckl = pickle.load(f)

            saved_U = torch.tensor(pckl[OUT_OBSERVED_CLONES_KEY])
            primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
            primary_idx = ordered_sites[i].index(primary_site)
            p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites[i])).T

            reranked_solutions = rank_solutions(vutil.create_reweighted_solution_set_from_pckl(pckl, O, p, cal_weights),
                                                print_config, needs_pruning=False)
            
            putil.save_best_trees(reranked_solutions, saved_U, O, cal_weights, ordered_sites[i], print_config, 
                                  custom_colors, primary_site, calibrate_dir, run_name)
    
    if estimate_observed_clones:
        for pooled_tsv_fn in pooled_tsv_fns:
            os.remove(pooled_tsv_fn) # cleanup pooled tsv

    return best_theta

def validate_inputs(T, node_idx_to_label, ref, var, primary_site, ordered_sites, weights, O, mode):
    if not (T.shape[0] == T.shape[1]):
        raise ValueError(f"Number of tree nodes should be consistent (T.shape[0] == T.shape[1])")
    if (T.shape[0] != len(node_idx_to_label)):
        raise ValueError(f"Number of node_idx_to_label needs to equal shape of adjacency matrix.")
    if not torch.is_tensor(T):
        raise ValueError("T is not a PyTorch tensor.")
    if ref != None and var != None:
        if ref.shape != var.shape:
            raise ValueError(f"ref and var must have identical shape, got {ref.shape} and {var.shape}")
        if not (ref.shape[1] == var.shape[1] == T.shape[0]):
            raise ValueError(f"Number of mutations/mutation clusters should be consistent (ref.shape[1] == var.shape[1] == T.shape[0])")
        if not (ref.shape[0] == var.shape[0] == len(ordered_sites)):   
            raise ValueError(f"Length of ordered_sites should be equal to ref and var dim 0")
    if not vutil.is_tree(T):
        raise ValueError("Adjacency matrix T is empty or not a tree.")
    if not primary_site in ordered_sites:
        raise ValueError(f"{primary_site} not in ordered_sites: {ordered_sites}")
    if (weights.organotrop == 0.0 and O != None)  and mode == "evaluate":
        print(f"Warning: O matrix was given but organotropism parameter of weights is 0.")
    if (weights.organotrop != 0.0 and O == None):
        raise ValueError(f"O matrix was not given but organotropism parameter of weights is non-zero. Please pass an O matrix.")
    if mode != 'calibrate' and mode != 'evaluate':
        raise ValueError(f"Valid modes are 'evaluate' and 'calibrate'")
    for label in list(node_idx_to_label.values()):
        if ":" in label:
            raise ValueError(f"Unfortunately our visualization code uses pydot, which does not allow colons (:) in node names. Please use a different separator in 'character_label' values.")


def infer_migration_history(T, tsv_fn, primary_site, weights, print_config, output_dir, run_name, estimate_observed_clones=True,
                            O=None, lr=0.05, init_temp=20, final_temp=0.01,batch_size=-1, custom_colors=None, bias_weights=True,
                            mode="evaluate",solve_polytomies=False, needs_pruning=True):
    '''
    Args:
        T: numpy ndarray or torch tensor (shape: num_internal_nodes x num_internal_nodes). Adjacency matrix (directed) of the internal nodes.
        
        tsv_fn: path to tsv with the required columns: 
            ['anatomical_site_index', 'anatomical_site_label', 'cluster_index', 'character_label', 
            'ref', 'var', 'var_read_prob', 'site_category']

        weights: Weight object for how much to penalize each component of the loss
        
        print_config: PrintConfig object with options on how to visualize output
        
        output_dir: path for where to save output trees to

        run_name: e.g. patient name, used for naming output files.

    Optional:
        
        O: numpy ndarray or torch tensor (shape: 1 x  num_anatomical_sites).
        Matrix of organotropism values from primary tumor type to other sites (in order of anatomical
        site indices indicated in tsv_fn).

        bias_weights: whether to initialize weights higher to favor vertex labeling of primary for all internal nodes

        mode: can be "evaluate" or "calibrate"

    Returns:
        Corresponding info on the *best* tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
        (4) dictionary w/ loss values for each component of the loss
        (5) how long (in seconds) the algorithm took to run
    '''

    start_time = datetime.datetime.now()

    # Extract inputs from tsv
    ref, var, omega, ordered_sites, node_idx_to_label, idx_to_observed_sites, G = dutil.extract_matrices_from_tsv(tsv_fn, estimate_observed_clones, T)
    
    # Validate inputs
    validate_inputs(T, node_idx_to_label, ref, var, primary_site, ordered_sites, weights, O, mode)
    print("ordered_sites",  ordered_sites)
    if batch_size == -1:
        batch_size = vutil.calculate_batch_size(T, ordered_sites, solve_polytomies)

    # When calibrating, we want to capture the Pareto front trees and not prune yet
    if mode == 'calibrate':
        print_config.k_best_trees = batch_size

    # Make the root index 0 (if it isn't already) to simplify indexing logic
    # Save the original root index to swap it back later though, since we don't 
    # want to confuse users with different cluster mappings etc.
    original_root_idx = vutil.get_root_index(T)
    T, ref, var, node_idx_to_label, G, idx_to_observed_sites, _, _ = vutil.restructure_matrices_root_index_zero(T, ref, var, node_idx_to_label, G, idx_to_observed_sites)
    assert(vutil.get_root_index(T) == 0)

    primary_idx = ordered_sites.index(primary_site)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites)).T
    num_sites = len(ordered_sites)
    num_internal_nodes = T.shape[0]
    
    primary_site_label = ordered_sites[torch.nonzero(p)[0][0]]

    identical_clone_gen_dist = 0.0
    if G != None:
        identical_clone_gen_dist = torch.min(G[(G != 0)])/2.0

    B = vutil.mutation_matrix_with_normal_cells(T)

    # Keep a copy of input clone tree (T from now on has leaf nodes from U)
    input_T = copy.deepcopy(T)

    config = {
        "init_temp": init_temp,
        "final_temp": final_temp,
        "v_anneal_rate": 0.01,
        "t_anneal_rate": 0.01,
        "lr": lr,
        "first_max_iter": 50,
        "second_max_iter": 100,
        "first_v_interval": 15,
        "second_v_interval": 20,
        "batch_size": batch_size,
        "bias_weights": bias_weights,
        "solve_polytomies": solve_polytomies,
        # the genetic distance between two identical clones is a close to 0 but non-zero value
        "identical_clone_gen_dist": identical_clone_gen_dist,
        "num_v_optimization_runs": 4
    }

    ############ Step 1, optimize U ############

    u_optimizer = uoptim.ObservedClonesSolver(num_sites, num_internal_nodes, ref, var, omega, idx_to_observed_sites,
                                              B, input_T, G, node_idx_to_label, weights, config, estimate_observed_clones)
    u_result = u_optimizer.run()
    U, input_T, T, G, L, node_idx_to_label, num_internal_nodes, idx_to_observed_sites = u_result

    ############ Step 2, optimize V ############

    num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling

    v_optimizer = voptim.VertexLabelingSolver(L, T, p, G, O, weights, config, num_sites, num_nodes_to_label,
                                              node_idx_to_label, input_T, idx_to_observed_sites)
    results = v_optimizer.run()
    
    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")
        
    ############ Step 3, visualize and save outputs ############
    with torch.no_grad():
        
        final_solutions = get_best_final_solutions(results, v_optimizer.G, O, p, weights,
                                                   print_config, node_idx_to_label, num_internal_nodes, needs_pruning)

        print("# final solutions:", len(final_solutions))

        edges, vert_to_site_map, mig_graph_edges, loss_info = putil.save_best_trees(final_solutions, U, O, weights,
                                                                                    ordered_sites,print_config, custom_colors, 
                                                                                    primary_site_label, output_dir, run_name,
                                                                                    original_root_idx=original_root_idx)   

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed

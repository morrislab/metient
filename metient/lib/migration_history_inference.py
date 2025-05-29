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

def prune_histories(solutions):
    """
    Filter solutions to keep only unique migration histories that lie on the Pareto front with respect to parsimony metrics.
    
    Args:
        solutions (list): List of VertexLabelingSolution objects representing different migration histories
        
    Returns:
        list: Subset of solutions that are unique and Pareto-optimal with respect to migration (m), 
              co-migration (c), and seeding (s) parsimony metrics
    """
    # Collect each unique solution's parsimony metrics
    all_pars_metrics = []
    unique_labelings = set()
    final_solutions = []
    for soln in solutions:
        tree = vutil.MigrationHistory(soln.T, soln.V)
        if tree not in unique_labelings:
            final_solutions.append(soln)
            unique_labelings.add(tree)
            all_pars_metrics.append((soln.m, soln.c, soln.s))
    # Keep the pareto front of trees
    pareto_metrics, pruned_solutions = vutil.pareto_front(final_solutions, all_pars_metrics)
    # print("Pareto_metrics:", set(pareto_metrics))

    return pruned_solutions

def rank_solutions(solution_set, print_config):
    """
    Sort solutions by total loss and return the top k solutions.
    
    Args:
        solution_set (list): List of VertexLabelingSolution objects to rank
        print_config (PrintConfig): Configuration object containing k_best_trees parameter
        
    Returns:
        list: Top k solutions sorted by ascending total loss, where k is specified in print_config.
              If fewer solutions exist than k, returns all solutions.
    """
    # 1. Sort the solutions from lowest to highest loss
    final_solutions = sorted(list(solution_set))

    # 2. Return the k best solutions
    k = print_config.k_best_trees if len(final_solutions) >= print_config.k_best_trees else len(final_solutions)
    final_solutions = final_solutions[:k]
    return final_solutions

def get_best_final_solutions(results, G, O, p, weights, print_config, 
                             node_collection, solve_polytomies, 
                             v_solver, num_internal_nodes, keep_pareto_only=True):
    """
    Process optimization results to get the final set of best migration history solutions.
    
    Args:
        results (list): List of optimization results, each containing:
            - best_Vs: Vertex labeling matrices
            - soft_Vs: Soft vertex labeling matrices (probabilities)
            - best_Ts: Tree adjacency matrices
            - poly_res (PolytomyResolver): Object containing polytomy resolver info
            - metrics: Tuple of (m,c,s,g,o,e) parsimony metrics
        G (torch.Tensor): Genetic distance matrix between sites
        O (dict): Organotropism dictionary mapping sites to frequencies
        p (torch.Tensor): One-hot encoding of primary site
        weights (Weights): Object containing weights for different loss components
        print_config (PrintConfig): Configuration for output formatting
        node_collection (NodeCollection): Object tracking tree node information
        solve_polytomies (bool): Whether polytomy resolution was performed
        v_solver (VertexLabelingSolver): Solver object containing optimization state
        num_internal_nodes (int): Number of internal nodes in tree
        keep_pareto_only (bool, optional): If True, only keep Pareto-optimal solutions. Defaults to True.
        
    Returns:
        list: Ranked list of best VertexLabelingSolution objects after:
            1. Collecting unique solutions
            2. Adding back any removed nodes
            3. Attempting to recover primary-only single-source solutions if none exist
            4. Removing unnecessary polytomy resolver nodes
            5. Computing Pareto front (if keep_pareto_only=True)
            6. Ranking by total loss
    """

    has_pss_solution = False
    full_solution_set = []
    all_pars_metrics, all_result_soln_indices = [],[]
    unique_labelings = set()
        
    # Get the indices of unique solutions
    for result_idx, result in enumerate(results):
        best_Vs, _, best_Ts, _, metrics = result

        for soln_idx, (m,c,s,g,o,e) in enumerate(zip(*metrics)):
            tree = vutil.MigrationHistory(best_Ts[soln_idx].clone(), best_Vs[soln_idx].clone())
            # print((int(m), int(c), int(s)))
            # print(tree.tree, "\n", torch.nonzero(tree.labeling))
            if tree not in unique_labelings:
                all_pars_metrics.append((int(m), int(c), int(s)))
                all_result_soln_indices.append((result_idx, soln_idx, (m,c,s,g,o,e)))

            unique_labelings.add(tree)

    fixed_labeling = v_solver.fixed_labeling
    # Create a list of unique VertexLabelingSolutions
    for idx in all_result_soln_indices:
        result_idx, soln_idx, metrics = idx[0], idx[1], idx[2]

        m,c,s,g,o,e = metrics
        loss = vutil.clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=1)
        
        # this solution is primary-only single-source, or we've seen one before
        has_pss_solution = s == 1 or has_pss_solution 
        
        V = results[result_idx][0][soln_idx].clone().cpu()
        soft_V = results[result_idx][1][soln_idx].clone().cpu()
        T = results[result_idx][2][soln_idx].clone().cpu()
        # Add back any removed nodes into V and T using FixedVertexLabeling info
        if fixed_labeling is not None and v_solver.config['collapse_nodes']:
            V = add_back_removed_nodes(V, v_solver, p)
            T = add_back_removed_nodes_to_tree(T, v_solver)
            if G is not None:
                G = v_solver.full_G
        soln = vutil.VertexLabelingSolution(loss, m,c,s,g,o,e, V, soft_V, T, G, node_collection)
        full_solution_set.append(soln)
  
    if not has_pss_solution:
        fixed_indices = [x for x in fixed_labeling.known_indices] if fixed_labeling!=None else []
        recover_prim_ss_solutions(full_solution_set, unique_labelings, 
                                  weights, O, p, solve_polytomies, 
                                  fixed_indices, num_internal_nodes)
        
    #  Remove any extra resolver nodes that don't actually help
    poly_res = results[0][3]
    prutil.remove_extra_resolver_nodes(full_solution_set, poly_res, weights, O, p)

    # Compute the pareto front
    if keep_pareto_only:
        pruned_histories = prune_histories(full_solution_set)  
    else:
        pruned_histories = full_solution_set

    return rank_solutions(pruned_histories, print_config)

def add_back_removed_nodes(V, v_solver, p):
    """Reconstruct full vertex labeling matrix by adding back removed nodes.
    
    Args:
        V (torch.Tensor): Current vertex labeling tensor (num_sites, num_remaining_nodes)
        fixed_labeling (FixedVertexLabeling): Object containing mapping info for removed nodes
        
    Returns:
        torch.Tensor: Expanded vertex labeling tensor including all original nodes
    """
    fixed_labeling = v_solver.fixed_labeling
    if fixed_labeling is None or V.shape[1] == v_solver.full_T.shape[0]:
        return V

    # Get the optimized indices
    optimized_indices_old = [x for x in range(1,v_solver.full_T.shape[0]) if x not in fixed_labeling.known_indices]
    optimized_indices_new = [fixed_labeling.old_to_new[x] for x in optimized_indices_old]

    # Create the full X matrix
    full_V = torch.zeros((V.shape[0], v_solver.full_T.shape[0]), device=V.device)
    full_V[:,0] = p.squeeze()
    # Fill in the full X tensor using the mapped indices
    full_V[:, optimized_indices_old] = V[:, optimized_indices_new]
    # Add in the known labelings (parts of optimal subtrees)
    full_V[:, fixed_labeling.known_indices] = fixed_labeling.known_labelings.to(V.device)

    for x in range(full_V.shape[1]):
        col_sum = int(torch.sum(full_V[:,x]))
        if col_sum != 1:
            print(x, col_sum)

    return full_V

def add_back_removed_nodes_to_tree(T, v_solver):
    """Reconstruct full tree adjacency matrix by adding back removed nodes and their connections.
    
    Args:
        T (torch.sparse.Tensor): Current tree adjacency matrix
        v_solver (VertexLabelingSolver): Object containing info on v optimization
        
    Returns:
        torch.sparse.Tensor: Expanded tree adjacency matrix including all original nodes and connections
    """
    if v_solver.fixed_labeling is None or T.shape[0] == v_solver.full_T.shape[0]:
        return T
        
    # Get dimensions
    num_total_nodes =  v_solver.full_T.shape[0]
    
    # Create mapping from new indices to old indices
    new_to_old = {v: k for k, v in v_solver.fixed_labeling.old_to_new.items()}
    
    # Convert mappings to tensors for vectorized operations
    new_indices = torch.tensor(list(new_to_old.keys()), device=T.device)
    old_indices = torch.tensor(list(new_to_old.values()), device=T.device)
    mapping = torch.zeros(num_total_nodes, dtype=torch.long, device=T.device)
    mapping[new_indices] = old_indices
    
    # Process current T edges
    T = T.coalesce()
    indices = T.indices()
    
    # Map indices using vectorized operations
    mapped_indices = torch.stack([
        mapping[indices[0]],
        mapping[indices[1]]
    ])
    
    # Process full_T edges for removed nodes
    full_T = v_solver.full_T.to_sparse().coalesce().to(T.device)
    full_indices = full_T.indices()
    
    # Create mask for edges involving known nodes
    known_indices = torch.tensor(list(v_solver.fixed_labeling.known_indices), device=T.device)
    known_mask = torch.isin(full_indices[0], known_indices) | torch.isin(full_indices[1], known_indices)
    
    # Combine all indices and values
    all_indices = torch.cat([mapped_indices, full_indices[:, known_mask]], dim=1)
    # Set all values to 1
    all_values = torch.ones(all_indices.shape[1], device=T.device)
    
    # Remove duplicates
    unique_edges, unique_idx = torch.unique(all_indices.T, dim=0, return_inverse=True)
    unique_values = torch.ones_like(unique_edges[:, 0], dtype=torch.float, device=T.device)
    
    # Create final sparse tensor
    new_T = torch.sparse_coo_tensor(
        unique_edges.T, 
        unique_values,
        (num_total_nodes, num_total_nodes)
    ).coalesce()
    
    return new_T

def recover_prim_ss_solutions(solution_set, unique_labelings, 
                              weights, O, p, solve_polytomies, 
                              fixed_indices, num_internal_nodes):
    """
    fixed_indices: indices of nodes that are in optimal subtrees 

    In cases where we are unable to find a primary-only seeding solution, 
    see if we can recover one by post-processing final solutions and removing 
    any met-to-met migration edges, and add these to our final solution set
    """
    print("fixed_indices", fixed_indices)
    all_node_indices = [x for x in range(num_internal_nodes)]
    print("all_node_indices", all_node_indices)
    
    new_solutions = []
    for i, solution in enumerate(solution_set):

        #seeding_nodes = putil.seeding_cluster_parents(solution.V,solution.T)
        all_node_indices = [x for x in range(num_internal_nodes)]
        node_info = solution.node_collection
        leaf_nodes = [x for x in all_node_indices if node_info.get_node(x).is_witness]
        
        # Don't touch the optimal subtrees or leaf nodes
        node_to_label_primary = set(all_node_indices) - (set(fixed_indices).union(set(leaf_nodes)))
        new_V = solution.V.clone()
        for s in node_to_label_primary:
            new_V[:,s] = p.T 
        
        unique_labeled_tree = vutil.MigrationHistory(solution.T, new_V)
        if unique_labeled_tree in unique_labelings:
            continue
        loss, new_values = vutil.clone_tree_labeling_objective(new_V, solution.soft_V, solution.T, 
                                                               solution.G, O, p, weights, update_path_matrix=solve_polytomies or i==0, 
                                                               compute_full_c=True)
        new_solution = vutil.VertexLabelingSolution(loss, *new_values, new_V, solution.soft_V, solution.T, solution.G, node_info)
        
        new_solutions.append(new_solution)
        unique_labelings.add(unique_labeled_tree)

    solution_set.extend(new_solutions)
    return

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

def one_hot_labeling_for_primary(primary_site, ordered_sites):
    """
    Args:
        - primary_site: name of the primary_site (must be a member of ordered_sites)
        - ordered_sites: list of sites (in the order that they appear in all matrix representations)
    Returns:
        one-hot column vector for a node labeled as belonging to the primary site
    """
    primary_idx = ordered_sites.index(primary_site)
    return torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites)).T


def evaluate_label_clone_tree(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O, sample_size, bias_weights, solve_polytomies, num_runs):
    """
    Observed clone proportions are inputted (in tsv_fns), only labeling of clone tree is needed
    """
    return evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name,
                    O, sample_size, bias_weights, solve_polytomies, num_runs, 
                    estimate_observed_clones=False)

def evaluate(tree_fn, tsv_fn, weights, print_config, output_dir, run_name, 
             O, sample_size, bias_weights, solve_polytomies, num_runs, 
             estimate_observed_clones=True):
    """
    Estimate observed clone proportions and labeling of clone tree for a single patient,
    using inputted weights.
    """
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
                              O=O, sample_size=sample_size, bias_weights=bias_weights, 
                              mode="evaluate", solve_polytomies=solve_polytomies, estimate_observed_clones=estimate_observed_clones,
                              num_runs=num_runs)
    if estimate_observed_clones:
        os.remove(pooled_tsv_fn) # cleanup pooled tsv

def patient_calibration_weight_from_soln_info(pt_to_soln_info):
    """
    Calculate the weight to place on this patient's contribution to the 
    cohort-level cross-entropy score. This is based on the tree size (number of edges), the number of possible primaries, 
    and the number of solutions found for each primary, since we don't want to bias towards patients
    with many possible primaries.
    
    Args:
        pt_to_soln_info: dict mapping patient index to list of [num_edges, num_solns] lists,
                        one per primary site
    Returns:
        List of patient weights for calibration
    """
    # Count number of primary sites with multiple solutions per patient
    pt_idx_to_denom = {i:0 for i in range(len(pt_to_soln_info))}
    for pt_idx, info_list in pt_to_soln_info.items():
        for num_edges, num_solns in info_list:
            if num_solns > 1:
                pt_idx_to_denom[pt_idx] += 1
    print('pt_idx_to_denom', pt_idx_to_denom)
    # Calculate weights as num_edges / num_primaries_with_multiple_solns
    weights = []
    for pt_idx, info_list in pt_to_soln_info.items():
        for num_edges, num_solns in info_list:
            if pt_idx_to_denom[pt_idx] > 0 and num_solns > 1:
                weights.append(num_edges / pt_idx_to_denom[pt_idx])
            else:
                weights.append(0)  # No multiple solution primaries
    print('new pt weights', weights)   
    return weights

def get_num_unique_pars_metrics(solns):
    """
    Get the number of unique parsimony metrics for a given set of solutions
    """
    unique_metrics = set()
    for soln in solns:
        unique_metrics.add((soln.m, soln.c, soln.s))
    return len(unique_metrics)

def validate_calibrate(run_names, calibration_type, Os):
    # Validate that run names are unique to avoid overwriting results
    if len(run_names) != len(set(run_names)):
        raise ValueError("Run names must be unique. Found duplicate names in: " + str(run_names))
    
    # Validate calibration type
    if calibration_type not in ["genetic", "organotropism", "both"]:
        raise ValueError("calibrate must be one of: 'genetic', 'organotropism', 'both'. Got: " + str(calibrate))
    
    # Check that Os is provided if using organotropism calibration
    if calibration_type in ["organotropism", "both"] and Os is None:
        raise ValueError("Organotropism frequencies (Os) must be provided when calibrating to organotropism.")

def calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
                               Os, sample_size, bias_weights, solve_polytomies, num_runs):
    """
    Observed clone proportions are inputted (in tsv_fns), only labeling of clone tree is needed
    """
    return calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
                    Os, sample_size, bias_weights, solve_polytomies, num_runs, estimate_observed_clones=False)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names, calibration_type,
              Os, sample_size, bias_weights, solve_polytomies, num_runs,
              estimate_observed_clones=True):
    """
    Estimate observed clone proportions and labeling of clone tree for a cohort of patients,
    calibrate parsimony weights to metastasis priors
    """
    
    validate_calibrate(run_names, calibration_type, Os)
    
    # Set calibration flags based on type
    calibrate_genetic = calibration_type in ["genetic", "both"] 
    calibrate_organotropism = calibration_type in ["organotropism", "both"]

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
    output_files= []
    pt_index_to_soln_info = {i:[] for i in range(len(Ts))}
    full_run_names = [] # includes primary site in the name

    # 1. Go through each patient and get migration history in calibrate mode
    for i in range(len(Ts)):
        print("\n*** Calibrating for patient:", run_names[i], "***")
        O = Os[i] if Os != None else None

        primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        if len(primary_sites) > 1:
            print("Multiple primaries given. Running each as primary")
            # Validate that primary sites are unique to avoid duplicate runs
            if len(primary_sites) != len(set(primary_sites)):
                raise ValueError("Primary site names must be unique. Found duplicate sites in: " + str(primary_sites))

        for primary_site in primary_sites:
            solns = infer_migration_history(Ts[i], pooled_tsv_fns[i], primary_site, weights, print_config, calibrate_dir, f"{run_names[i]}_{primary_site}", 
                                            O=O, sample_size=sample_size, bias_weights=bias_weights,
                                            mode="calibrate", solve_polytomies=solve_polytomies, estimate_observed_clones=estimate_observed_clones,
                                            num_runs=num_runs)

            full_run_name = f"{run_names[i]}_{primary_site}"
            output_files.append(os.path.join(calibrate_dir, f"{full_run_name}.pkl.gz"))
            full_run_names.append(full_run_name)
            pt_index_to_soln_info[i].append([Ts[i].shape[0]-1, get_num_unique_pars_metrics(solns)])
    
    pt_weights = patient_calibration_weight_from_soln_info(pt_index_to_soln_info)
    full_run_name_to_calibration_weight = {k:v for k,v in zip(full_run_names, pt_weights)}
    print("full_run_name_to_calibration_weight",full_run_name_to_calibration_weight)

    # 2. Find the best theta for this cohort
    best_theta = eutil.get_max_cross_ent_thetas(output_files, pt_weights, calibrate_genetic, calibrate_organotropism)
    rounded_best_theta = [round(v,3) for v in best_theta]
    with open(os.path.join(calibrate_dir, "best_theta.json"), 'w') as json_file:
        json.dump(rounded_best_theta, json_file, indent=2)
        
    with open(os.path.join(calibrate_dir, "full_run_name_to_calibration_weight.json"), 'w') as json_file:
        json.dump(full_run_name_to_calibration_weight, json_file, indent=2)

    ordered_sites = dutil.extract_ordered_sites(pooled_tsv_fns)
        
    # 3. Recalibrate trees using the best thetas
    print_config.visualize = visualize
    print_config.k_best_trees = input_k

    # Set genetic distance and organotropism weights based on whether we're using them
    gen_dist_weight = 1.0 if calibrate_genetic else 0.0
    organotrop_weight = 1.0 if calibrate_organotropism else 0.0
    cal_weights = met.Weights(mig=[best_theta[0]*PARS_METRIC_MULTIPLIER], comig=best_theta[1]*PARS_METRIC_MULTIPLIER, seed_site=[best_theta[2]*PARS_METRIC_MULTIPLIER],
                              gen_dist=gen_dist_weight, organotrop=organotrop_weight)
    
    # 4. Use the saved trees to rescore trees, visualize, and re-save 
    for i in range(len(Ts)):
        primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        for primary_site in primary_sites:
            O = Os[i] if Os != None else None
            O = dutil.initialize_organotropism_vector(O, ordered_sites[i], primary_site)
            run_name = f"{run_names[i]}_{primary_site}"
            with gzip.open(os.path.join(calibrate_dir, f"{run_name}.pkl.gz"), 'rb') as f:
                pckl = pickle.load(f)

            saved_U = torch.tensor(pckl[OUT_OBSERVED_CLONES_KEY])
            p = one_hot_labeling_for_primary(primary_site, ordered_sites[i])
            
            reranked_solutions = rank_solutions(vutil.create_reweighted_solution_set_from_pckl(pckl, O, p, cal_weights),
                                                print_config)
            
            putil.save_best_trees(reranked_solutions, saved_U, O, cal_weights, ordered_sites[i], print_config, 
                                  primary_site, calibrate_dir, run_name)
    
    if estimate_observed_clones:
        for pooled_tsv_fn in pooled_tsv_fns:
            os.remove(pooled_tsv_fn) # cleanup pooled tsv

    return best_theta

def to_cpu(tensor):
    """
    Send a tensor to CPU.
    
    Parameters:
        tensor (torch.Tensor or None): The tensor to send to CPU.

    Returns:
        torch.Tensor or None: The tensor on CPU or None if the input was None.
    """
    if tensor is not None:
        return tensor.to('cpu')
    return None

def validate_inputs(T, node_collection, ref, var, primary_site, ordered_sites, weights, O, mode):
    """
    Validate the inputs to Metient.
    """
    if not (T.shape[0] == T.shape[1]):
        raise ValueError(f"Number of tree nodes should be consistent (T.shape[0] == T.shape[1])")
    if (T.shape[0] != len(node_collection.idx_to_label())):
        raise ValueError(f"Number of cluster indices needs to equal shape of adjacency matrix.")
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
        print(f"Warning: O dictionary was given but organotropism parameter of weights is 0.")
    if (weights.organotrop != 0.0 and O == None):
        raise ValueError(f"O dictionary was not given but organotropism parameter of weights is non-zero. Please pass an O matrix.")
    if mode != 'calibrate' and mode != 'evaluate':
        raise ValueError(f"Valid modes are 'evaluate' and 'calibrate'")
    for label in list(node_collection.idx_to_label().values()):
        if ":" in label:
            raise ValueError(f"Unfortunately our visualization code uses pydot, which does not allow colons (:) in node names. Please use a different separator in 'character_label' values.")


def infer_migration_history(T, tsv_fn, primary_site, weights, print_config, output_dir, run_name, estimate_observed_clones=True,
                            O=None, lr=0.05, init_temp=20, final_temp=0.01, sample_size=-1, bias_weights=True,
                            mode="evaluate", solve_polytomies=False, num_runs=1, keep_pareto_only=True):
    """
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
        
        O: a dictionary mapping anatomical site name (as used in tsv_fn) -> frequency of metastasis (these values should be normalized)

        bias_weights: whether to initialize weights higher to favor vertex labeling of primary + the sites that a node's children are detected in 

        mode: can be "evaluate" or "calibrate"

    Returns:
        Corresponding info on the *best* tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
        (4) dictionary w/ loss values for each component of the loss
        (5) how long (in seconds) the algorithm took to run
    """

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Using GPU")

    start_time = datetime.datetime.now()

    # Extract inputs from tsv
    initialize_G = mode == 'calibrate' or weights.gen_dist != 0
    ref, var, omega, ordered_sites, node_collection, idx_to_observed_sites, G, O = dutil.extract_matrices_from_tsv(tsv_fn, estimate_observed_clones, 
                                                                                                                   T, initialize_G, O, primary_site)
    
    print("O:", O)
    print("Tumor samples:", ordered_sites)
    # Validate inputs
    validate_inputs(T, node_collection, ref, var, primary_site, ordered_sites, weights, O, mode)

    if sample_size == -1:
        sample_size = vutil.calculate_sample_size(T.shape[0], len(ordered_sites), solve_polytomies)
    opt_subtree_sample_size = sample_size
    # Total sample size gets split for the individual parsimony models
    sample_size = sample_size // len(ALL_PARSIMONY_MODELS)
    
    # When calibrating, we want to capture the Pareto front trees and not prune yet
    if mode == 'calibrate':
        print_config.k_best_trees = float("inf")

    # Make the root index 0 (if it isn't already) to simplify indexing logic
    # Save the original root index to swap it back later though, since we don't 
    # want to confuse users with different cluster mappings etc.
    original_root_idx = vutil.get_root_index(T)
    T, ref, var, node_collection, G, idx_to_observed_sites, _, _ = vutil.restructure_matrices_root_index_zero(T, ref, var, node_collection, G, idx_to_observed_sites)
    assert(vutil.get_root_index(T) == 0)

    p = one_hot_labeling_for_primary(primary_site, ordered_sites)
    num_sites = len(ordered_sites)
    num_internal_nodes = T.shape[0]
    
    primary_site_label = ordered_sites[torch.nonzero(p)[0][0]]

    identical_clone_gen_dist = 0.0
    if G != None:
        identical_clone_gen_dist = torch.min(G[(G != 0)])/2.0

    # Keep a copy of input clone tree (T from now on has leaf nodes from U)
    input_T = copy.deepcopy(T)
    print("input_T.shape", input_T.shape)
    # TODO: Make this a more sophisticated decision
    use_sparse_T = input_T.shape[0] > 100
    print("Using sparse T:", use_sparse_T)

    config = {
        "init_temp": init_temp,
        "final_temp": final_temp,
        "v_anneal_rate": 0.01,
        "t_anneal_rate": 0.01,
        "lr": lr,
        "first_max_iter": 100,
        "second_max_iter": 150 if solve_polytomies else 100,
        "first_v_interval": 15,
        "second_v_interval": 20,
        "opt_subtree_sample_size": opt_subtree_sample_size,
        "sample_size": sample_size,
        "bias_weights": bias_weights,
        "solve_polytomies": solve_polytomies,
        # the genetic distance between two identical clones is a close to 0 but non-zero value
        "identical_clone_gen_dist": identical_clone_gen_dist,
        "num_runs":num_runs,
        "promote_diversity": True,
        "diversity_weight": 0.2,
        "use_sparse_T": use_sparse_T,
        # If we're using a dense T, collapsing nodes is not worth the compute time,
        # and solving polytomies makes the indexing extremely complicated for collapsing (maybe implement in the future)
        #"collapse_nodes": use_sparse_T and not solve_polytomies
        "collapse_nodes": False
    }

    ############ Step 1, optimize U ############

    u_optimizer = uoptim.ObservedClonesSolver(num_sites, num_internal_nodes, ref, var, omega, idx_to_observed_sites,
                                              input_T, G, node_collection, weights, config, 
                                              estimate_observed_clones, ordered_sites)
    u_result = u_optimizer.run()
    U, input_T, T, G, L, node_collection, num_internal_nodes, idx_to_observed_sites = u_result

    ############ Step 2, optimize V ############

    num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling

    v_optimizer = voptim.VertexLabelingSolver(L, T, p, G, O, weights, config, num_sites, num_nodes_to_label,
                                              node_collection, input_T, idx_to_observed_sites)
    results = v_optimizer.run()
    node_collection = v_optimizer.node_collection

    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")
        
    ############ Step 3, visualize and save outputs ############
    with torch.no_grad():
        # Send tensors to CPU after inference to free up GPU memory
        G = to_cpu(v_optimizer.G)
        O = to_cpu(O)
        p = to_cpu(p)
        
        final_solutions = get_best_final_solutions(results, G, O, p, weights, print_config, 
                                                   node_collection, solve_polytomies,
                                                   v_optimizer, num_internal_nodes, keep_pareto_only=keep_pareto_only)
        print("Number of final solutions:", len(final_solutions))

        putil.save_best_trees(final_solutions, U, O, weights,ordered_sites, print_config,
                              primary_site_label, output_dir, run_name,original_root_idx=original_root_idx) 

    torch.cuda.empty_cache()
 

    return final_solutions

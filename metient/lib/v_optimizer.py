import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import copy
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt

from metient.lib import polytomy_resolver as prutil
from metient.util import vertex_labeling_util as vutil
from metient import metient as met
from metient.util.globals import *
from metient.util import optimal_subtrees as opt_sub

PROGRESS_BAR = 0 # Keeps track of optimization progress using tqdm

class VertexLabelingSolver:
    def __init__(self, L, full_T, p, G, O, weights, config, num_sites, num_nodes_to_label,
                 node_collection, input_T, idx_to_observed_sites):
        """Initialize the VertexLabelingSolver.
        
        Args:
            L: Witness node labels
            full_T: Adjacency matrix of internal nodes + witness nodes
            p: Primary tumor labeling (one-hot vector)
            G: Genetic distance matrix between nodes (or None)
            O: Organotropism vector indicating seeding preferences (or None)
            weights: Weight parameters for different optimization terms
            config: Configuration dictionary for solver parameters
            num_sites: Number of anatomical sites
            num_nodes_to_label: Number of nodes that need labeling
            node_collection: Collection of node information
            input_T: Adjacency matrix of internal nodes only
            idx_to_observed_sites: Mapping from node indices to their observed sites
        """
        # Tree structure matrices
        self.input_T = input_T
        self.full_T = full_T
        self.T = full_T.clone().detach()  # Working copy for optimization
        
        # Node labelings and site information
        self.L = L
        self.p = p
        self.num_sites = num_sites
        self.num_nodes_to_label = num_nodes_to_label
        self.idx_to_observed_sites = idx_to_observed_sites
        
        # Genetic and organotropic information
        self.full_G = G
        self.G = G.clone().detach() if G is not None else None
        self.O = O
        
        # Optimization parameters
        self.weights = weights
        self.config = config
        self.node_collection = node_collection
        
        # State variables (set during optimization)
        self.poly_res = None  # Polytomy resolver
        self.fixed_labeling = None  # Fixed node labelings
    
    def run(self):
        return run_multiple_optimizations(self)
    
        
        
        return valid_sites

        return valid_sites

def optimize_v_t(v_solver, X, poly_res, exploration_weights, max_iter, v_interval, is_second_optimization):
    """
    Perform gradient-based Gumbel-softmax optimization on V and T (if resolving polytomies)
    """
    if v_solver.config['use_sparse_T']:
        v_solver.T = v_solver.T.to(X.device).to_sparse()

    # Unpack config
    lr = v_solver.config['lr']
    init_temp, final_temp = v_solver.config['init_temp'], v_solver.config['final_temp']
    t_anneal_rate, v_anneal_rate = v_solver.config['t_anneal_rate'], v_solver.config['v_anneal_rate']

    v_optimizer = torch.optim.Adam([X], lr=lr)
    v_scheduler = lr_scheduler.LinearLR(v_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)
    
    # Determine device type from X tensor
    device_type = 'cuda' if X.is_cuda else 'cpu'
    scaler = GradScaler()

    solve_polytomies = v_solver.config['solve_polytomies']
    if solve_polytomies:
        v_optimizer = torch.optim.Adam([X, poly_res.latent_var], lr=lr)
    v_temps, t_temps = [], []

    v_temp = init_temp
    t_temp = init_temp
    j = 0
    k = 0
    identical_T = (not solve_polytomies)

    global PROGRESS_BAR


    # print("X\n", X[0])

    for i in range(max_iter):
        update_path = update_path_matrix(i, max_iter, solve_polytomies, is_second_optimization)

        v_optimizer.zero_grad()
        # On the last iteration (before we return results), compute the full comigration number and not an approximation
        compute_full_c = i == (max_iter-1)
        V, losses, soft_V, T, metrics = compute_v_t_loss(X, v_solver, poly_res, exploration_weights, update_path, 
                                                        v_temp, t_temp, compute_full_c, identical_T)
        mean_loss = torch.mean(losses)
        scaler.scale(mean_loss).backward()

        # Step the optimizer, unscale gradients, and update scaler
        scaler.step(v_optimizer)
        scaler.update()
        v_scheduler.step()
        
        if i % v_interval == 0:
            v_temp = np.maximum(v_temp * np.exp(-v_anneal_rate * k), final_temp)
        k += 1

        if solve_polytomies and update_path:
            if i % v_interval == 0:
                t_temp = np.maximum(t_temp * np.exp(-t_anneal_rate * j), final_temp)
            j += 1

        v_temps.append(v_temp)
        t_temps.append(t_temp)

        PROGRESS_BAR.update(1)
    return V, soft_V, T, metrics, poly_res


def initialize_polytomy_resolver(v_solver, solve_polytomies):
    """Initialize the polytomy resolver if needed.
    
    Args:
        v_solver: The vertex labeling solver instance
        solve_polytomies: Boolean indicating whether to solve polytomies
        
    Returns:
        tuple: (PolytomyResolver instance or None, updated solve_polytomies flag)
    """
    if not solve_polytomies:
        return None, False
        
    nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(
        v_solver.input_T, v_solver.T, 
        v_solver.idx_to_observed_sites, 3, True, 2
    )
    
    if len(nodes_w_polys) == 0:
        print("No potential polytomies to solve, not resolving polytomies.")
        return None, False
        
    poly_res = prutil.PolytomyResolver(v_solver, nodes_w_polys, resolver_sites)
    v_solver.num_nodes_to_label += len(poly_res.resolver_indices)
    return poly_res, True
    
def first_v_optimization(v_solver, exploration_weights):
    
    vutil.LAST_P = None
    
    solve_polytomies = v_solver.config['solve_polytomies']
    # mult_run_sample_size = v_solver.config['sample_size']
    # v_solver.config['sample_size'] = v_solver.config['opt_subtree_sample_size']

    # If solving for polytomies, setup T and G appropriately
    if solve_polytomies:
        nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(v_solver.input_T, v_solver.T, 
                                                                           v_solver.idx_to_observed_sites, 3, True, 2)
        if len(nodes_w_polys) == 0:
            print("No potential polytomies to solve, not resolving polytomies.")
            poly_res, solve_polytomies = None, False
        else:
            poly_res = prutil.PolytomyResolver(v_solver, nodes_w_polys, resolver_sites)
            v_solver.num_nodes_to_label += len(poly_res.resolver_indices)
    else:
        poly_res = None

    v_solver.poly_res = poly_res
    v_solver.config['solve_polytomies'] = solve_polytomies
    
    # We're learning X, which is the vertex labeling of the internal nodes
    X = x_weight_initialization(v_solver)
    X.requires_grad = True

    # First optimization
    V, _, T, _, _ = optimize_v_t(v_solver, X, v_solver.poly_res, exploration_weights, v_solver.config['first_max_iter'],
                                 v_solver.config['first_v_interval'], False)
    # Identify optimal subtrees, keep them fixed, and solve for the rest of the tree
    optimal_nodes, optimal_batch_nums = opt_sub.find_optimal_subtrees(T, V, v_solver)
    
    T = T.cpu().detach()
    V = V.cpu().detach()
    torch.cuda.empty_cache()

    #v_solver.config['sample_size'] = mult_run_sample_size
    return optimal_nodes, optimal_batch_nums, T, V

    
def second_v_optimization(v_solver, run_specific_x, run_specific_poly_res, exploration_weights):
    vutil.LAST_P = None

    # Second optimization
    V, soft_V, T, metrics, run_specific_poly_res = optimize_v_t(v_solver, run_specific_x, run_specific_poly_res, exploration_weights,
                                                                v_solver.config['second_max_iter'], v_solver.config['second_v_interval'], True)

    V = V.cpu().detach()
    soft_V = soft_V.cpu().detach()
    T = T.cpu().detach()
    metrics = tuple(metric.cpu().detach() for metric in metrics)

    # Free up GPU memory after inference
    if v_solver.config['solve_polytomies']:
        run_specific_poly_res.latent_var = run_specific_poly_res.latent_var.cpu().detach()
    torch.cuda.empty_cache()

    return V, soft_V, T, run_specific_poly_res, metrics

def full_exploration_weights(weights):
    return met.Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                       seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, data_fit=weights.data_fit, 
                       reg=weights.reg, entropy=weights.entropy, gen_dist=0.0, organotrop=0.0)

def no_metastasis_solution(v_solver):
    """
    In the case where there are no metastases (the only site is the primary), we don't
    need to do any optimization
    """
    vertex_labeling = vutil.add_batch_dim(torch.ones(1, v_solver.num_nodes_to_label))
    V = vutil.stack_vertex_labeling(v_solver.L, vertex_labeling, v_solver.p, None, None)
    metrics = tuple(torch.zeros(size=(1,), device=V.device) for _ in range(6))
    v_solver.T = v_solver.T.to_sparse()
    ret = [(V, torch.zeros(V.shape, device=V.device), vutil.repeat_n(v_solver.T,1), None, metrics)]
    return ret

def run_multiple_optimizations(v_solver):
    """
    Run optimization on V/T on a first pass to find optimal subtrees, fix those subtrees,
    then run second optimization to infer Pareto optimal solutions, using multiple parsimony models
    to promote exploration
    """

    if v_solver.num_sites == 1:
        return no_metastasis_solution(v_solver)

    global PROGRESS_BAR
    PROGRESS_BAR = tqdm(total=v_solver.config['first_max_iter'] + v_solver.config['second_max_iter']*len(ALL_PARSIMONY_MODELS)*v_solver.config['num_runs'], position=0)
    
    results = []
    
    # Only run first optimization once (this finds optimal subtrees)
    first_opt_result = first_v_optimization(v_solver, full_exploration_weights(v_solver.weights))
    optimal_nodes, optimal_batch_nums, T, V = first_opt_result

    # Function to wrap the second optimization 
    def second_optimization_task(v_solver, exploration_weights):
        # Each run needs its own polytomy resolver and X
        run_specific_poly_solver = copy.deepcopy(v_solver.poly_res)
        run_specific_x = x_weight_initialization(v_solver)
        run_specific_x, v_solver = opt_sub.init_optimal_x_polyres(run_specific_x, run_specific_poly_solver, optimal_nodes, 
                                                                  optimal_batch_nums, T, V, v_solver)
        ret = second_v_optimization(v_solver, run_specific_x, run_specific_poly_solver, exploration_weights)
        return ret

    for _ in range(v_solver.config['num_runs']):
        for pars_model in ALL_PARSIMONY_MODELS:
            exploration_weights = met.Weights(mig=pars_model[0], comig=pars_model[1], 
                                              seed_site=pars_model[2], data_fit=v_solver.weights.data_fit, 
                                              reg=v_solver.weights.reg, entropy=v_solver.weights.entropy, gen_dist=0.0, organotrop=0.0)
            ret = second_optimization_task(v_solver, exploration_weights)
            results.append(ret)

    # TODO diagnose why Fitch-Hartigan is failing in some cases (e.g. H103207)
    if not v_solver.config['solve_polytomies']:
        # try:
        vutil.run_fitch_hartigan(v_solver, results)
        # except:
        #     pass

    return results

def sample_gumbel(shape, eps=1e-8):
    G = torch.rand(shape)
    return -torch.log(-torch.log(G + eps) + eps)

def softmax_shifted_3d(X):
    shifted = X - X.max(dim=1, keepdim=True)[0]
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=1, keepdim=True)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return softmax_shifted_3d(y / temperature)

def gumbel_softmax(logits, temperature, hard=True):
    """
    Adapted from https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [sample_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [sample_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes

    """
    shape = logits.size()
    assert len(shape) == 3 # [sample_size, num_sites, num_nodes]
    y_soft = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, k = y_soft.max(1)
        y_hard = torch.zeros(shape, dtype=logits.dtype).scatter_(1, torch.unsqueeze(k, 1), 1.0)

        # This cool bit of code achieves two things:
        # (1) makes the output value exactly one-hot (since we add then subtract y_soft value)
        # (2) makes the gradient equal to y_soft gradient (since we strip all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y, y_soft

def update_t_with_polytomy_resolver(poly_res: prutil.PolytomyResolver, 
                                    t_temp: float, 
                                    v_solver: VertexLabelingSolver) -> torch.Tensor:
    """
    Updates an adjacency matrix T based on children-to-parent assignments 
    resolved using the polytomy resolver (poly_res) and Gumbel-softmax sampling.

    Removes old connections between child nodes and their previous parents, 
    replacing them with the new connections resolved by the polytomy resolver.
    Additionally, removes connections to nodes in `resolver_indices` if they have no children.

    Args:
        poly_res (Any): The polytomy resolver containing:
            - latent_var: A batch_size x n x m matrix, where n is the number of nodes 
              and m is the number of children of polytomies.
            - children_of_polys: The order of child nodes of polytomies in latent_var's 2nd dim.
        t_temp (float): Temperature parameter for Gumbel-softmax sampling.
        v_solver (Any): A solver object with an adjacency matrix T (a sparse tensor).

    Returns:
        torch.Tensor: Updated sparse adjacency matrix T for the batch, 
        with old parent-child connections replaced as per the resolved decisions.
    """
    # Perform Gumbel-softmax to resolve parent assignments for children of polytomies
    # softmax_pol_res is sample_size x number of nodes x children of polytomies
    softmax_pol_res, ss = gumbel_softmax(poly_res.latent_var, t_temp)
    bs = poly_res.latent_var.shape[0]
    
    if not v_solver.T.is_sparse:
        T = vutil.repeat_n(v_solver.T, bs)
        T[:,:,poly_res.children_of_polys] = softmax_pol_res
        return T

    non_zero_indices = torch.nonzero(softmax_pol_res, as_tuple=False).T

    # Extract relevant indices from the non-zero entries
    batch_indices = non_zero_indices[0]
    parent_indices = non_zero_indices[1] 
    child_indices = non_zero_indices[2]
    global_child_indices = torch.tensor(poly_res.children_of_polys, device=softmax_pol_res.device)[child_indices]

    # Create new connections
    new_indices = torch.stack([batch_indices, parent_indices, global_child_indices])
    new_values = torch.ones(new_indices.size(1), dtype=torch.float32, device=softmax_pol_res.device)

    # Repeat and coalesce the adjacency matrix for the batch
    T = vutil.repeat_n(v_solver.T, bs).coalesce()

    # Filter out old connections for the updated children
    existing_indices = T.indices()
    existing_values = T.values()
    
    # Perform comparison using broadcasting
    batch_child_pairs = (
    existing_indices[0] * T.shape[2] + existing_indices[2]
        )  # Hash batch-child pairs
    new_batch_child_pairs = (
        batch_indices * T.shape[2] + global_child_indices
    )  # Hash new batch-child pairs

    mask = ~torch.isin(batch_child_pairs, new_batch_child_pairs)
    filtered_indices = existing_indices[:, mask]  # Mask out parent-child relations
    filtered_values = existing_values[mask]  # Apply the same mask to values
    # Concatenate filtered existing data with new connections
    updated_indices = torch.cat([filtered_indices, new_indices], dim=1)
    updated_values = torch.cat([filtered_values, new_values])

    # Create updated sparse tensor
    updated_T = torch.sparse_coo_tensor(updated_indices, updated_values, T.shape).coalesce()

    return updated_T

def compute_v_t_loss(X, v_solver, poly_res, exploration_weights, update_path_matrix, v_temp, t_temp, compute_full_c, identical_T):
    """
    Args:
        X: latent variable of labelings we are solving for. (sample_size x num_unknown_nodes x num_sites)
            where num_unkown_nodes = len(T) - (len(known_indices)), or len(unknown_indices)
        L: witness node labels derived from U
        T: Full adjacency matrix which includes clone tree nodes as well as witness nodes which were
            added from U > U_CUTOFF (observed in a site)
        p: one-hot vector indicating site of the primary
        G: Matrix of genetic distances between internal nodes (shape:  num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).  

    Returns:
        Loss of the labeling we're trying to learn (X) by computing maximum parsimony loss, organotropism, and
        genetic distance loss (if weights for genetic distance and organotropism != 0)
    """
    softmax_X, softmax_X_soft = gumbel_softmax(X, v_temp)
    V = vutil.stack_vertex_labeling(v_solver.L, softmax_X, v_solver.p, v_solver.poly_res, v_solver.fixed_labeling)

    bs = X.shape[0]
    if poly_res != None:
        T = update_t_with_polytomy_resolver(poly_res, t_temp, v_solver)
    else:
        T = vutil.repeat_n(v_solver.T, bs)
    
    G = v_solver.G
    if G != None:
        G = vutil.repeat_n(G, T.shape[0])
    loss, metrics = vutil.clone_tree_labeling_objective(V, softmax_X_soft, T, v_solver.G, 
                                                        v_solver.O, v_solver.p, exploration_weights, 
                                                        update_path_matrix=update_path_matrix, compute_full_c=compute_full_c,
                                                        identical_T=identical_T)
    # Add diversity loss
    if v_solver.config.get('promote_diversity', True):
        diversity_weight = v_solver.config.get('diversity_weight')
        
        # Calculate pairwise similarities between solutions
        flat_solutions = softmax_X.reshape(bs, -1)
        similarity_matrix = torch.mm(flat_solutions, flat_solutions.t())
        
        # Exclude self-similarity by zeroing diagonal
        similarity_matrix = similarity_matrix * (1 - torch.eye(bs, device=X.device))
        
        # Calculate per-solution similarity scores (sum of similarities to all other solutions)
        per_solution_similarity = similarity_matrix.sum(dim=1) / (bs - 1)  # Normalize by number of other solutions
        
        # Normalize per-solution similarities to [0,1] range
        per_solution_diversity_loss = per_solution_similarity / (per_solution_similarity.max() + 1e-8)
        
        # Normalize main loss to be between 0 and 1
        normalized_main_loss = loss / (loss.max() + 1e-8)

        #print(normalized_main_loss[:5], per_solution_diversity_loss[:5])
        
        # Add weighted normalized losses for each solution individually
        loss = (1 - diversity_weight) * normalized_main_loss + diversity_weight * per_solution_diversity_loss
    return V, loss, softmax_X_soft, T, metrics

def x_weight_initialization(v_solver):
    # Set a random seed for this initialization
    torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
    sample_size = v_solver.config['sample_size']
    X = torch.rand(sample_size, v_solver.num_sites, v_solver.num_nodes_to_label)
    
    if v_solver.config['bias_weights']:
        nodes_w_children, biased_sites = vutil.get_k_or_more_children_nodes(
            v_solver.input_T, v_solver.T, v_solver.idx_to_observed_sites, 1, True, 1, cutoff=False)
        
        # Add more randomness to the bias strength
        eta = 5.0 * (0.5 + torch.rand(1).item())  # Random eta between 2.5 and 5.0
        quart = sample_size // 4
        
        # Add some randomness to the partition sizes
        partition_jitter = torch.randint(-quart//4, quart//4, (1,)).item()
        quart_2 = quart * 2 + partition_jitter
        quart_3 = quart * 3 + partition_jitter
        
        # Create masks for different initialization strategies
        primary_mask = torch.zeros_like(X)
        children_mask = torch.zeros_like(X)
        
        # Primary site bias with random noise
        prim_site_idx = torch.nonzero(v_solver.p)[0][0]
        primary_mask[:quart_2, prim_site_idx, :] = 1.0 + 0.2 * torch.rand_like(X[:quart_2, prim_site_idx, :])
        
        # Children site bias with random noise
        for node_idx, sites in zip(nodes_w_children, biased_sites):
            if node_idx == 0:
                continue
            idx = node_idx - 1
            for site_idx in sites:
                children_mask[quart:quart_3, site_idx, idx] = 1.0 + 0.2 * torch.rand_like(X[quart:quart_3, site_idx, idx])
        
        # Apply biases with additional randomness
        X = X * (1.0 - primary_mask - children_mask) + \
            (torch.rand_like(X) * eta + eta/2) * primary_mask + \
            (torch.rand_like(X) * eta) * children_mask
        
        # Add small random perturbations to all values
        X = X + 0.1 * torch.rand_like(X)
    
    return X

def update_path_matrix(itr, max_iter, solve_polytomies, second_optimization):
    if itr == -1:
        return True
    if not solve_polytomies:
        return False
    return True
    # if second_optimization:
    #     return itr > max_iter*1/4 and itr < max_iter*3/4
    return itr > max_iter*1/4 and itr < max_iter*3/4
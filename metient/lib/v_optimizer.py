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
    def __init__(self, L, T, p, G, O, weights, config, num_sites, num_nodes_to_label,
                 node_collection, input_T, idx_to_observed_sites):
        self.L = L # witness node labels
        self.input_T = input_T # adjacency matrix of internal nodes
        self.T = T # adjacency matrix of internal nodes + witness nodes
        self.p = p # primary tumor labeling
        self.G = G # genetic distance matrix
        self.O = O # organotropism vector
        self.weights = weights
        self.config = config
        self.num_sites = num_sites
        self.num_nodes_to_label = num_nodes_to_label
        self.node_collection = node_collection
        self.idx_to_observed_sites = idx_to_observed_sites
        # This gets set at first optimization time
        self.poly_res = None
        self.fixed_labeling = None
    
    def run(self):
        return run_multiple_optimizations(self)

def optimize_v_t(v_solver, X, poly_res, exploration_weights, max_iter, v_interval, is_second_optimization):
    '''
    Perform gradient-based Gumbel-softmax optimization on V and T (if resolving polytomies)
    '''

    v_solver.T = v_solver.T.to_sparse()

    # Unpack config
    lr = v_solver.config['lr']
    init_temp, final_temp = v_solver.config['init_temp'], v_solver.config['final_temp']
    t_anneal_rate, v_anneal_rate = v_solver.config['t_anneal_rate'], v_solver.config['v_anneal_rate']

    v_optimizer = torch.optim.Adam([X], lr=lr)
    v_scheduler = lr_scheduler.LinearLR(v_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)
    
    solve_polytomies = v_solver.config['solve_polytomies']
    if solve_polytomies:
        v_optimizer = torch.optim.Adam([X, poly_res.latent_var], lr=lr)
    v_temps, t_temps = [], []

    v_temp = init_temp
    t_temp = init_temp
    j = 0
    k = 0

    scaler = GradScaler()

    global PROGRESS_BAR

    for i in range(max_iter):
        update_path = update_path_matrix(i, max_iter, solve_polytomies, is_second_optimization)

        v_optimizer.zero_grad()
        # On the last iteration (before we return results), compute the full comigration number and not an approximation
        compute_full_c = i == (max_iter-1)
        V, losses, soft_V, T, metrics = compute_v_t_loss(X, v_solver, poly_res, exploration_weights, update_path, v_temp, t_temp, compute_full_c)
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
    
    # plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(v_temps))], v_temps, marker='.'); plt.show(); plt.close()
    # plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(t_temps))], t_temps, marker='.'); plt.show(); plt.close()

    return V, soft_V, T, poly_res, metrics

def first_v_optimization(v_solver, exploration_weights):
    
    vutil.LAST_P = None
    
    solve_polytomies = v_solver.config['solve_polytomies']

    # If solving for polytomies, setup T and G appropriately
    if solve_polytomies:
        nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(v_solver.input_T, v_solver.T, 
                                                                           v_solver.idx_to_observed_sites, 3, True, 2)
        if len(nodes_w_polys) == 0:
            print("No potential polytomies to solve, not resolving polytomies.")
            poly_res, solve_polytomies = None, False
        else:
            poly_res = prutil.PolytomyResolver(v_solver, nodes_w_polys, resolver_sites)
            #num_resolver_nodes = poly_res.resolver_indices
            v_solver.num_nodes_to_label += len(poly_res.resolver_indices)
            #T, G, node_idx_to_label = poly_res.T, poly_res.G, poly_res.node_idx_to_label
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
    return optimal_nodes, optimal_batch_nums, T, V

    
def second_v_optimization(v_solver, run_specific_x, run_specific_poly_res, exploration_weights):
    vutil.LAST_P = None

    # Second optimization
    V, soft_V, T, run_specific_poly_res, metrics = optimize_v_t(v_solver, run_specific_x, run_specific_poly_res, exploration_weights,
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


def run_fitch_hartigan(v_solver, results):
    """
    Recover possible ancestral states using Fitch's algorithm and select one optimal solution

    Parameters:
    - adj_matrix: Sparse COO adjacency matrix representing the tree structure (n x n).
    - node_idx_to_observed_sites: Dictionary mapping leaf node indices to their labels.
    - root_label: The known label of the root node.

    Returns:
    - ancestral_matrix: A k x n matrix, where k is the number of labels, and n is the internal node indices.
    """
    adj_matrix = v_solver.input_T
    node_idx_to_observed_sites = v_solver.idx_to_observed_sites
    root_label = torch.argmax(v_solver.p, dim=0).item()
    
    n = adj_matrix.shape[0]
    k = v_solver.num_sites

    # Step 1: Initialize a k x n matrix to store possible labels for each node (True/False for each label)
    ancestral_matrix = torch.zeros((k, n), dtype=torch.float32)

    # Step 2: Fill in the matrix for leaf nodes and root node based on the node_idx_to_observed_sites dictionary
    for node, leaf_labels in node_idx_to_observed_sites.items():
        ancestral_matrix[:, node] = 0
        for leaf_label in leaf_labels:
            ancestral_matrix[leaf_label, node] = 1  # Set the corresponding label to True

    # For leaf nodes without any observed sites, initialize as possibly belonging to all sites
    leaf_nodes = torch.nonzero(adj_matrix.sum(dim=1) == 0).squeeze(dim=1).tolist()

    for leaf in leaf_nodes:
        if leaf not in node_idx_to_observed_sites:
            ancestral_matrix[:, leaf] = 1

    root = vutil.get_root_index(adj_matrix)
    ancestral_matrix[:, root] = 0
    ancestral_matrix[root_label, root] = 1

    # Step 3: Perform the downward pass (Fitch algorithm) for internal nodes
    def fitch_down(node):
        node = int(node)

        # Check if the node has observed sites
        has_observed_sites = node in node_idx_to_observed_sites

        # Find children of the node using the adjacency matrix
        children = torch.nonzero(adj_matrix[node]).squeeze(dim=1)

        # If the node is a leaf, return its known label set
        if len(children) == 0 and has_observed_sites:
            return ancestral_matrix[:, node]

        # Collect label sets from all children
        child_label_sets = [fitch_down(child) for child in children]

        # Perform set intersection or union based on children's labels
        intersection = torch.stack(child_label_sets).all(dim=0)
        if torch.any(intersection):  # If intersection is non-empty, use it as the set
            ancestral_matrix[:, node] = intersection
        else:  # Otherwise, use the union of the children's sets
            union = torch.stack(child_label_sets).any(dim=0)
            ancestral_matrix[:, node] = union

        # If the node has observed sites, enforce them on the computed labels
        if has_observed_sites:
            observed_sites = ancestral_matrix[:, node]
            ancestral_matrix[:, node] = ancestral_matrix[:, node] * observed_sites

        return ancestral_matrix[:, node]
    
    # Step 4: Perform Fitch's downward pass starting from the root
    fitch_down(root)
    # Reinforce root label
    ancestral_matrix[:, root] = 0
    ancestral_matrix[root_label, root] = 1
    single_soln_matrix = ancestral_matrix.clone()

    # Step 5: Perform the upward pass to choose specific states for internal nodes,
    # starting with the root label.
    def fitch_up(node, parent_label):
        children = torch.nonzero(adj_matrix[node]).squeeze(dim=1)

        # If there are multiple possible labels, choose one (use the parent's label if possible)
        possible_labels = torch.where(single_soln_matrix[:, node])[0]
        if parent_label in possible_labels:
            chosen_label = parent_label
        else:
            chosen_label = possible_labels[0]

        # Set the chosen label to True and others to False for the node
        single_soln_matrix[:, node] = 0
        single_soln_matrix[chosen_label, node] = 1

        # Pass the chosen label to the children
        for child in children:
            fitch_up(child, chosen_label)

    # Perform the upward pass starting from the root with the enforced root label
    fitch_up(root, root_label)


    # We needed to include the root labeling for Fitch-Hartigan, but we restack 
    # it using stack_vertex_labeling, so remove it momentarily
    single_soln_matrix = torch.cat((single_soln_matrix[:, :root], single_soln_matrix[:, root+1:]), dim=1)
    V = stack_vertex_labeling(v_solver.L, vutil.add_batch_dim(single_soln_matrix), v_solver.p, None, None)
    T = vutil.repeat_n(v_solver.T,1)
    metrics = vutil.ancestral_labeling_metrics(V, T, v_solver.G, v_solver.O, v_solver.p, 
                                                 update_path_matrix=False, compute_full_c=True, identical_T=True)
    V = V.cpu().detach()
    T = T.cpu().detach()
    metrics = tuple(metric.cpu().detach() for metric in metrics)
    print("Fitch-hartigan result:", metrics)
    results.append((V, torch.zeros(V.shape, device=V.device), T, None, (*metrics,torch.zeros(size=(1,),device=V.device))))


def no_metastasis_solution(v_solver):
    '''
    In the case where there are no metastases (the only site is the primary), we don't
    need to do any optimization
    '''
    vertex_labeling = vutil.add_batch_dim(torch.ones(1, v_solver.num_nodes_to_label))
    V = stack_vertex_labeling(v_solver.L, vertex_labeling, v_solver.p, None, None)
    metrics = tuple(torch.zeros(size=(1,), device=V.device) for _ in range(6))
    v_solver.T = v_solver.T.to_sparse()
    ret = [(V, torch.zeros(V.shape, device=V.device), vutil.repeat_n(v_solver.T,1), None, metrics)]
    return ret

def run_multiple_optimizations(v_solver):
    '''
    Run optimization on V/T on a first pass to find optimal subtrees, fix those subtrees,
    then run second optimization to infer Pareto optimal solutions, using multiple parsimony models
    to promote exploration
    '''

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

    if not v_solver.config['solve_polytomies']:
        run_fitch_hartigan(v_solver, results)

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
    '''
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

    '''
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

def stack_vertex_labeling(L, X, p, poly_res, fixed_labeling):
    '''
    Use leaf labeling L and X (both of size sample_size x num_sites X num_internal_nodes)
    to get the anatomical sites of the leaf nodes and the internal nodes (respectively). 
    Stack the root labeling to get the full vertex labeling V. 
    '''
    # Expand leaf node labeling L to be repeated sample_size times
    bs = X.shape[0]
    L = vutil.repeat_n(L, bs)

    if fixed_labeling != None:
        full_X = torch.zeros((bs, X.shape[1], len(fixed_labeling.known_indices)+len(fixed_labeling.unknown_indices)))
        known_labelings = vutil.repeat_n(fixed_labeling.known_labelings, bs)
        full_X[:,:,fixed_labeling.unknown_indices] = X
        full_X[:,:,fixed_labeling.known_indices] = known_labelings
    else:
        full_X = X

    full_vert_labeling = torch.cat((full_X, L), dim=2)
    p = vutil.repeat_n(p, bs)
    # Concatenate the left part, new column, and right part along the second dimension
    full_vert_labeling = torch.cat((p, full_vert_labeling), dim=2)

    # print(full_vert_labeling[0,:,14])
    # print(full_vert_labeling[-1,:,14])

    return full_vert_labeling

def update_t_with_polytomy_resolver(poly_res: prutil.PolytomyResolver, 
                                    t_temp: float, 
                                    v_solver: VertexLabelingSolver) -> torch.Tensor:
    """
    Updates a sparse adjacency matrix T based on children-to-parent assignments 
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
    # print("ss\n", ss)
    non_zero_indices = torch.nonzero(softmax_pol_res, as_tuple=False).T
    # print("non_zero_indices\n", non_zero_indices)

    # Extract relevant indices from the non-zero entries
    batch_indices = non_zero_indices[0]
    parent_indices = non_zero_indices[1] 
    child_indices = non_zero_indices[2]
    global_child_indices = torch.tensor(poly_res.children_of_polys, device=softmax_pol_res.device)[child_indices]
    # print("poly_res.children_of_polys", poly_res.children_of_polys)
    # print('parent_indices\n', parent_indices )
    # print('parent_indices\n', parent_indices )
    # print('global_child_indices\n', global_child_indices)
    # Create new connections
    new_indices = torch.stack([batch_indices, parent_indices, global_child_indices])
    new_values = torch.ones(new_indices.size(1), dtype=torch.float32, device=softmax_pol_res.device)

    # Repeat and coalesce the adjacency matrix for the batch
    bs = poly_res.latent_var.shape[0]
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

def compute_v_t_loss(X, v_solver, poly_res, exploration_weights, update_path_matrix, v_temp, t_temp, compute_full_c):
    '''
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
    '''
    softmax_X, softmax_X_soft = gumbel_softmax(X, v_temp)
    V = stack_vertex_labeling(v_solver.L, softmax_X, v_solver.p, v_solver.poly_res, v_solver.fixed_labeling)

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
                                                        update_path_matrix, compute_full_c=compute_full_c)
    return V, loss, softmax_X_soft, T, metrics

def x_weight_initialization(v_solver):

    nodes_w_children, biased_sites = vutil.get_k_or_more_children_nodes(v_solver.input_T, v_solver.T, v_solver.idx_to_observed_sites, 1, True, 1, cutoff=False)
    sample_size = v_solver.config['sample_size']
    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(sample_size, v_solver.num_sites, v_solver.num_nodes_to_label)
    if v_solver.config['bias_weights']:
        eta = 3
        # Make 4 partitions: (1) biased towards the primary, (2) biased towards the primary + sites of children/grandchildren
        # (3) biased towards sites of children/grandchildren, (4) no bias
        quart = sample_size // 4
        
        # Bias for partitions [1-2]
        prim_site_idx = torch.nonzero(v_solver.p)[0][0]
        # This is really important to prevent large trees from getting stuck in local optima
        X[:quart*2,prim_site_idx,:] = eta / 2

        # Bias for partitions [2-3]
        # For each node, find which sites to bias labeling towards
        # by using the sites it and its children are detected in
        for node_idx, sites in zip(nodes_w_children, biased_sites):
            if node_idx == 0:
                continue # we know the root labeling
            idx = node_idx - 1
            for site_idx in sites:
                X[quart:quart*3,site_idx,idx] = eta

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

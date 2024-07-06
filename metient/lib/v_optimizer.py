import torch
import torch.optim.lr_scheduler as lr_scheduler
from metient.util import vertex_labeling_util as vutil
import numpy as np
import copy
from metient.lib import polytomy_resolver as prutil
from metient import metient as met
from metient.util.globals import *
import matplotlib.pyplot as plt
from tqdm import tqdm

PROGRESS_BAR = 0 # Keeps track of optimization progress using tqdm

class VertexLabelingSolver:
    def __init__(self, L, T, p, G, O, weights, config, num_sites, num_nodes_to_label,
                 node_idx_to_label, input_T, idx_to_observed_sites):
        self.L = L
        self.T = T
        self.p = p
        self.G = G
        self.O = O
        self.weights = weights
        self.config = config
        self.num_sites = num_sites
        self.num_nodes_to_label = num_nodes_to_label
        self.node_idx_to_label = node_idx_to_label
        self.input_T = input_T
        self.idx_to_observed_sites = idx_to_observed_sites
        # This gets set at first optimization time
        self.poly_res = None
        self.fixed_labeling = None
    
    def run(self):
        return run_multiple_optimizations(self)

def optimize_v(v_solver, X, poly_res, exploration_weights, max_iter, v_interval, is_second_optimization):
    
    # Unpack config
    lr = v_solver.config['lr']
    init_temp, final_temp = v_solver.config['init_temp'], v_solver.config['final_temp']
    t_anneal_rate, v_anneal_rate = v_solver.config['t_anneal_rate'], v_solver.config['v_anneal_rate']

    v_optimizer = torch.optim.Adam([X], lr=lr)
    v_scheduler = lr_scheduler.LinearLR(v_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)
    
    solve_polytomies = v_solver.config['solve_polytomies']
    if solve_polytomies:
        poly_optimizer = torch.optim.Adam([poly_res.latent_var], lr=lr)
        poly_scheduler = lr_scheduler.LinearLR(poly_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)
    v_temps, t_temps = [], []

    v_temp = init_temp
    t_temp = init_temp
    j = 0
    k = 0

    global PROGRESS_BAR
    for i in range(max_iter):
        update_path = update_path_matrix(i, max_iter, solve_polytomies, is_second_optimization)
        if solve_polytomies and update_path:
            poly_optimizer.zero_grad()

        v_optimizer.zero_grad()
        V, v_losses, soft_V, T = compute_v_loss(X, v_solver, poly_res, exploration_weights, update_path, v_temp, t_temp)
        mean_loss = torch.mean(v_losses)
        mean_loss.backward()

        if solve_polytomies and update_path:
            poly_optimizer.step()
            poly_scheduler.step()
            if i % 5 == 0:
                t_temp = np.maximum(t_temp * np.exp(-t_anneal_rate * j), final_temp)
            j += 1

        else:
            v_optimizer.step()
            v_scheduler.step()

            if i % v_interval == 0:
                v_temp = np.maximum(v_temp * np.exp(-v_anneal_rate * k), final_temp)
            k += 1

        v_temps.append(v_temp)
        t_temps.append(t_temp)

        PROGRESS_BAR.update(1)
    
    # plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(v_temps))], v_temps, marker='.'); plt.show(); plt.close()
    # plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(t_temps))], t_temps, marker='.'); plt.show(); plt.close()

    return V, soft_V, T, poly_res

def first_v_optimization(v_solver, exploration_weights):
    # We're learning X, which is the vertex labeling of the internal nodes
    X = x_weight_initialization(v_solver)
    X.requires_grad = True
    
    vutil.LAST_P = None
    
    solve_polytomies = v_solver.config['solve_polytomies']

    # If solving for polytomies, setup T and G appropriately
    if solve_polytomies:
        nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(v_solver.input_T, v_solver.T, 
                                                                           v_solver.idx_to_observed_sites, 3, True, 2)
        # print("nodes_w_polys",nodes_w_polys, "resolver_sites", resolver_sites)
        if len(nodes_w_polys) == 0:
            print("No potential polytomies to solve, not resolving polytomies.")
            poly_res, solve_polytomies = None, False
        else:
            poly_res = prutil.PolytomyResolver(v_solver, nodes_w_polys, resolver_sites)
            #T, G, node_idx_to_label = poly_res.T, poly_res.G, poly_res.node_idx_to_label
    else:
        poly_res = None

    v_solver.poly_res = poly_res
    v_solver.config['solve_polytomies'] = solve_polytomies

    # First optimization
    optimized_Vs, _, optimized_Ts, _ = optimize_v(v_solver, X, v_solver.poly_res, exploration_weights, v_solver.config['first_max_iter'],
                                                  v_solver.config['first_v_interval'], False)
    
    # Identify optimal subtrees, keep them fixed, and solve for the rest of the tree
    optimal_nodes, optimal_batch_nums = find_optimal_subtrees(optimized_Ts, optimized_Vs, v_solver)

    return optimal_nodes, optimal_batch_nums, optimized_Ts, optimized_Vs

    
def second_v_optimization(v_solver, run_specific_x, run_specific_poly_res, exploration_weights):
    vutil.LAST_P = None

    # Second optimization
    optimzed_Vs, optimzed_soft_Vs, optimzed_Ts, run_specific_poly_res = optimize_v(v_solver, run_specific_x, run_specific_poly_res, exploration_weights,
                                                            v_solver.config['second_max_iter'], v_solver.config['second_v_interval'], True)


    return optimzed_Vs, optimzed_soft_Vs, optimzed_Ts, run_specific_poly_res

def full_exploration_weights(weights):
    return met.Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                       seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, data_fit=weights.data_fit, 
                       reg=weights.reg, entropy=weights.entropy, gen_dist=0.0, organotrop=0.0)


def run_multiple_optimizations(v_solver):
    global PROGRESS_BAR
    PROGRESS_BAR = tqdm(total=v_solver.config['first_max_iter'] + v_solver.config['second_max_iter']*v_solver.config['num_v_optimization_runs'], position=0)

    results = []
    
    # Only run first optimization once (this finds optimal subtrees)

    first_opt_result = first_v_optimization(v_solver, full_exploration_weights(v_solver.weights))
    optimal_nodes, optimal_batch_nums, optimized_Ts, optimized_Vs = first_opt_result

    # Function to wrap the second optimization 
    def second_optimization_task(v_solver, exploration_weights):
        # Each run needs its own polytomy resolver and X
        run_specific_poly_solver = copy.deepcopy(v_solver.poly_res)
        run_specific_x = x_weight_initialization(v_solver)
        run_specific_x, v_solver = initialize_optimal_x_polyres(run_specific_x, run_specific_poly_solver, optimal_nodes, optimal_batch_nums, optimized_Ts, optimized_Vs, v_solver)
        ret = second_v_optimization(v_solver, run_specific_x, run_specific_poly_solver, exploration_weights)
        return ret
    mig_weights = [1000.0,1.0,1.0]*100
    comig_weights = [1.0,1000.0,1.0]*100
    seed_weights = [1.0, 1.0,1000.0]*100
    order = [0,1,2]*100
    for r in range(v_solver.config['num_v_optimization_runs']):
        weight_idx = order[r]
        exploration_weights = met.Weights(mig=mig_weights[weight_idx], comig=comig_weights[weight_idx], 
                                         seed_site=seed_weights[weight_idx], data_fit=v_solver.weights.data_fit, 
                                         reg=v_solver.weights.reg, entropy=v_solver.weights.entropy, gen_dist=0.0, organotrop=0.0)
        
        # print(exploration_weights.mig,exploration_weights.comig, exploration_weights.seed_site)
        results.append(second_optimization_task(v_solver, exploration_weights))
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
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes

    '''
    shape = logits.size()
    assert len(shape) == 3 # [batch_size, num_sites, num_nodes]
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
    Use leaf labeling L and X (both of size batch_size x num_sites X num_internal_nodes)
    to get the anatomical sites of the leaf nodes and the internal nodes (respectively). 
    Stack the root labeling to get the full vertex labeling V. 
    '''
    # Expand leaf node labeling L to be repeated batch_size times
    bs = X.shape[0]
    L = vutil.repeat_n(L, bs)

    if fixed_labeling != None:
        full_X = torch.zeros((bs, X.shape[1], len(fixed_labeling.known_indices)+len(fixed_labeling.unknown_indices)))
        known_labelings = vutil.repeat_n(fixed_labeling.known_labelings, bs)
        full_X[:,:,fixed_labeling.unknown_indices] = X
        full_X[:,:,fixed_labeling.known_indices] = known_labelings
    else:
        full_X = X

    if poly_res != None:
        # Order is: internal nodes, new poly nodes, leaf nodes from U
        full_vert_labeling = torch.cat((full_X, vutil.repeat_n(poly_res.resolver_labeling, bs), L), dim=2)
    else:
        full_vert_labeling = torch.cat((full_X, L), dim=2)

    p = vutil.repeat_n(p, bs)
    # Concatenate the left part, new column, and right part along the second dimension
    return torch.cat((p, full_vert_labeling), dim=2)

def compute_v_loss(X, v_solver, poly_res, exploration_weights, update_path_matrix, v_temp, t_temp):
    '''
    Args:
        X: latent variable of labelings we are solving for. (batch_size x num_unknown_nodes x num_sites)
            where num_unkown_nodes = len(T) - (len(known_indices)), or len(unknown_indices)
        L: leaf node labels derived from U
        T: Full adjacency matrix which includes clone tree nodes as well as leaf nodes which were
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
        softmax_pol_res, _ = gumbel_softmax(poly_res.latent_var, t_temp)
        T = vutil.repeat_n(v_solver.T, bs)
        T[:,:,poly_res.children_of_polys] = softmax_pol_res
    else:
        T = vutil.repeat_n(v_solver.T, bs)

    G = v_solver.G
    if G != None:
        G = vutil.repeat_n(G, T.shape[0])
    loss, _ = vutil.clone_tree_labeling_objective(V, softmax_X_soft, T, v_solver.G, 
                                                  v_solver.O, v_solver.p, exploration_weights, update_path_matrix)
    return V, loss, softmax_X_soft, T

def x_weight_initialization(v_solver):

    nodes_w_children, biased_sites = vutil.get_k_or_more_children_nodes(v_solver.input_T, v_solver.T, v_solver.idx_to_observed_sites, 1, True, 1, cutoff=False)
    batch_size = v_solver.config['batch_size']
    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(batch_size, v_solver.num_sites, v_solver.num_nodes_to_label)
    if v_solver.config['bias_weights']:
        eta = 3
        # Make 4 partitions: (1) biased towards the primary, (2) biased towards the primary + sites of children/grandchildren
        # (3) biased towards sites of children/grandchildren, (4) no bias
        quart = batch_size // 4
        
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
    if second_optimization:
        return itr > max_iter*1/3 and itr < max_iter*2/3
    return itr > max_iter*1/3 and itr < max_iter*2/3

#################################################################
################### FIX OPTIMAL SUBTREES ########################
#################################################################

def find_optimal_subtree_nodes(optimized_Ts, optimized_Vs, num_internal_nodes):
    '''
    Args:
        - optimized_Ts: all possible solutions for possible adjacency matrices
        - optimized_Vs: all possible solutions for possible vertex labeling
    Returns:
        A list of node indices and their descendatns which belong to optimal subtrees (i.e.)
        all nodes in the subtree have the same color/label, and a list of the batch numbers
        that these optimal subtrees were found 
    '''

    P = vutil.path_matrix(optimized_Ts, remove_self_loops=False)
    VT = torch.transpose(optimized_Vs, 2, 1)
    # i,j = 1 if node i and node j have the same label 
    X = VT @ optimized_Vs
    same_color_subtrees = torch.logical_not(torch.logical_and(1 - X, P))
    # Get the indices of rows where all elements are True (all nodes have the same label)
    cand_optimal_subtree_indices = torch.nonzero(torch.all(same_color_subtrees, dim=2))
    # Tells us how many descendants each node has
    row_sums = torch.sum(P, dim=2)
    indexed_row_sums = torch.tensor([row_sums[idx[0]][idx[1]] for idx in cand_optimal_subtree_indices]).unsqueeze(1)
    cand_optimal_subtree_indices = torch.cat((cand_optimal_subtree_indices, indexed_row_sums), dim=1)
    # 2. Sort the optimal_subtrees by the number of children they have,
    # so that when we are solving for polytomies, we get the largest optimal subtrees possible
    cand_optimal_subtree_indices = cand_optimal_subtree_indices[cand_optimal_subtree_indices[:, 2].argsort(descending=True)]
    
    ## 3. Only keep optimal subtree roots that have a leaf node
    # nodes_w_leaves = vutil.nodes_w_leaf_nodes(optimized_Ts, num_internal_nodes)
    # cand_optimal_subtree_indices = [x for x in cand_optimal_subtree_indices if nodes_w_leaves[x[0], x[1]]]
    seen_nodes = set()
    optimal_batch_nums, optimal_subtree_nodes = [],[]
    for cand in cand_optimal_subtree_indices:
        batch_num = int(cand[0])
        optimal_subtree_root = int(cand[1])
        if optimal_subtree_root not in optimal_subtree_nodes:
            # Add the optimal_subtree_root and all its descendants
            descendants = [t.item() for t in torch.nonzero(P[batch_num,optimal_subtree_root])]
            
            # Don't fix clone's leaf nodes (num. descendants == 1), since we already know their labeling, 
            # and if they are under an optimal polytomy branch, they would be getting added by an optimal
            # subtree rooted by an ancestor
            if len(descendants) == 1: 
                continue
            # Don't fix a subtree where there are no leaf nodes 
            leaf_node_in_optimal_subtree = False
            for descendant in descendants:
                if descendant >= num_internal_nodes:
                    leaf_node_in_optimal_subtree = True
            
            if not leaf_node_in_optimal_subtree:
                continue
            # Don't fix when it's just a node and its leaf (the labeling of the node could be
            # multiple possibilities and still be optimal)
            if len(descendants) == 2:
                continue
            #print(optimal_subtree_root, descendants, leaf_node_in_optimal_subtree, batch_num)
            current_node_set = []
            for descendant in descendants:
                if descendant not in seen_nodes:
                    current_node_set.append(descendant)
                    seen_nodes.add(descendant)
            if len(current_node_set) > 0:
                optimal_batch_nums.append(batch_num)
                optimal_subtree_nodes.append(current_node_set)
    return optimal_subtree_nodes, optimal_batch_nums

def initialize_optimal_x_polyres(X, poly_res, optimal_subtree_nodes,optimal_batch_nums, optimized_Ts, optimized_Vs, v_solver):
    
    poly_resolver_to_optimal_children = {}
    known_indices = []
    known_labelings = []
    # TODO: re-initialize after first optimization?
    # if poly_res != None:
    #     # Anywhere the resolver is not -inf, reset the starting values to 1s
    #     poly_res.latent_var[poly_res.latent_var != float('-inf')] = 1

    # Fix node labels and node edges
    for optimal_subtree_set,optimal_batch_num in zip(optimal_subtree_nodes,optimal_batch_nums):
        for node_idx in optimal_subtree_set:
            # If this is a witness node from U or the root index, we already know its vertex labeling
            if node_idx <= X.shape[2] and node_idx != 0:
                optimal_site = int(optimized_Vs[optimal_batch_num,:,node_idx].nonzero(as_tuple=False))
                idx = node_idx - 1 # X doesn't include root node
                known_indices.append(idx)
                known_labelings.append(torch.eye(v_solver.num_sites)[optimal_site].T)
                X[:,optimal_site,idx] = 1
                non_optimal_sites = [i for i in range(v_solver.num_sites) if i != optimal_site]
                X[:,non_optimal_sites,idx] = float("-inf")

            # If this node is the child of a polytomy resolver node, fix its location
            # if the parent (the polytomy resolver node) belongs to the same optimal subtree
            if poly_res != None and node_idx in poly_res.children_of_polys:
                poly_idx = poly_res.children_of_polys.index(node_idx)
                parent_idx = int(optimized_Ts[optimal_batch_num,:,node_idx].nonzero(as_tuple=False))
                if parent_idx not in optimal_subtree_set:
                    continue
                poly_res.latent_var[:,parent_idx, poly_idx] = 1
                non_parents = [i for i in range(optimized_Ts.shape[1]) if i != parent_idx]
                poly_res.latent_var[:,non_parents,poly_idx] = float("-inf")
                poly_res.latent_var[:,non_parents,poly_idx] = float("-inf")

                # Don't let any other non-optimal children of this polytomy resolver move around
                if parent_idx in poly_res.resolver_indices:
                    if parent_idx not in poly_resolver_to_optimal_children:
                        optimal_children = vutil.get_child_indices(optimized_Ts[optimal_batch_num,:,:], [parent_idx])
                        poly_resolver_to_optimal_children[parent_idx] = optimal_children

    
    if poly_res != None:
        # Fix all other polytomy children s.t. they cannot move to be a child of the fixed node_idx
        # for parent_idx in poly_resolver_to_optimal_children:
            
        #     optimal_children = poly_resolver_to_optimal_children[parent_idx]
        #     optimal_children_poly_indices = [poly_res.children_of_polys.index(i) for i in optimal_children]
        #     other_children = [i for i in range(poly_res.latent_var.shape[2]) if i not in optimal_children_poly_indices]
        #     poly_res.latent_var[:,parent_idx,other_children] = float("-inf")
        poly_res.latent_var.requires_grad = True
    
    fixed_labeling = None
    if len(known_indices) != 0:
        unknown_indices = [x for x in range(v_solver.num_nodes_to_label) if x not in known_indices]
        known_labelings = torch.stack(known_labelings, dim=1)
        X = X[:,:,unknown_indices] # only include the unknown indices for inference
        fixed_labeling = vutil.FixedVertexLabeling(known_indices, unknown_indices, known_labelings)

    v_solver.fixed_labeling = fixed_labeling
    X.requires_grad = True

    return X, v_solver

def find_optimal_subtrees(optimized_Ts, optimized_Vs, v_solver):
    '''
    After the first round of optimization, there are optimal subtrees (subtrees where
    the labelings of *all* nodes is the same), which we can keep fixed, since there
    are no other more optimal labelings rooted at this branch.

    Two things we can fix: the labeling of the nodes in optimal subtrees,
    and the edges of the subtrees if polytomy resolution is being used. Search all
    samples to find optimal subtrees, since there might not be one solution with all 
    optimal subtrees.
    '''
    
    num_internal_nodes = v_solver.num_nodes_to_label + 1 # root node
    
    # Re-initialize optimized_Ts with the tree with the best subtree structure
    poly_res = v_solver.poly_res
    if poly_res != None:
        poly_res.latent_var.requires_grad = False
        num_internal_nodes += len(poly_res.resolver_indices)
        # print("poly_res.children_of_polys", poly_res.children_of_polys)
        # print("poly_res.resolver_indices", poly_res.resolver_indices)
        # print("poly_res.resolver_index_to_parent_idx",poly_res.resolver_index_to_parent_idx)
    
    # 1. Find samples with optimal subtrees
    optimal_subtree_nodes, optimal_batch_nums = find_optimal_subtree_nodes(optimized_Ts, optimized_Vs, num_internal_nodes)

    # # 3. Re-initialize X and polytomy resolver with optimal subtrees (labelings and structure) fixed. 
    # X = x_weight_initialization(v_solver)
    # X, v_solver = initalize_optimal_x_polyres(X, optimal_subtree_nodes, optimal_batch_nums, optimized_Ts, optimized_Vs, v_solver)
            
    return optimal_subtree_nodes, optimal_batch_nums
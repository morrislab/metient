import torch
import numpy as np
import datetime
from tqdm import tqdm
from torch.distributions.binomial import Binomial
import os
import copy
import shutil
import pickle
import gzip
import json
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from metient import metient as met
from metient.util import vertex_labeling_util as vutil
from metient.util import data_extraction_util as dutil
from metient.util import eval_util as eutil
from metient.util import plotting_util as putil
from metient.util import polytomy_resolution_util as prutil
from metient.util.globals import *

torch.set_printoptions(precision=2)

PROGRESS_BAR = 0 # Keeps track of optimization progress using tqdm

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def get_repeating_weight_vector(bs, weight_list):
    # Calculate the number of times each weight should be repeated
    total_weights = len(weight_list)
    repeats = bs // total_weights
    remaining_elements = bs % total_weights

    # Create a list where each weight is repeated 'repeats' times
    repeated_list = [weight for weight in weight_list for _ in range(repeats)]

    # Add remaining elements to match the batch size exactly
    if remaining_elements > 0:
        additional_weights = weight_list[:remaining_elements]
        additional_repeated = [weight for weight in additional_weights]
        repeated_list += additional_repeated

    # Convert the list to a tensor
    weights_vec = torch.tensor(repeated_list)

    return weights_vec

def get_mig_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.mig)

def get_seed_site_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.seed_site)

def extract_metrics_from_loss_dict(loss_dict):
    m, c, s = loss_dict[MIG_KEY].item(), loss_dict[COMIG_KEY].item(), loss_dict[SEEDING_KEY].item()
    g = loss_dict[GEN_DIST_KEY].item()
    o = loss_dict[ORGANOTROP_KEY].item()
    e = loss_dict[ENTROPY_KEY].item()
    return m, c, s, g, o, e

def clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=1):

    # Combine all 5 components with their weights
    # Explore different weightings
    if isinstance(weights.mig, list) and isinstance(weights.seed_site, list):
        mig_weights_vec = get_mig_weight_vector(bs, weights)
        seeding_sites_weights_vec = get_seed_site_weight_vector(bs, weights)
        mig_loss = torch.mul(mig_weights_vec, m)
        seeding_loss = torch.mul(seeding_sites_weights_vec, s)
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
        
    else:
        mig_loss = weights.mig*m
        seeding_loss = weights.seed_site*s
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
    return labeling_loss

def clone_tree_labeling_objective(V, soft_V, A, G, O, p, weights, update_path_matrix):
    '''
    Args:
        V: Vertex labeling of the full tree (batch_size x num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        G: Matrix of genetic distances between internal nodes (shape:  num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        p: one-hot vector indicating site of the primary
        weights: Weights object

    Returns:
        Loss to score the ancestral vertex labeling of the given tree. This combines (1) migration number, (2) seeding site
        number, (3) comigration number, and optionally (4) genetic distance and (5) organotropism.
    '''
    
    m, c, s, g, o = vutil.ancestral_labeling_metrics(V, A, G, O, p, update_path_matrix)
    # Entropy
    e = vutil.calc_entropy(V, soft_V)

    labeling_loss = clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=V.shape[0])

    loss_dict = {MIG_KEY: m, COMIG_KEY:c, SEEDING_KEY: s, ORGANOTROP_KEY: o, GEN_DIST_KEY: g, ENTROPY_KEY: e}

    return labeling_loss, loss_dict

# Adapted from PairTree
def calc_llh(F_hat, R, V, omega_v):
    '''
    Args:
        F_hat: estimated subclonal frequency matrix (num_nodes x num_mutation_clusters)
        R: Reference allele count matrix (num_samples x num_mutation_clusters)
        V: Variant allele count matrix (num_samples x num_mutation_clusters)
    Returns:
        Data fit using the Binomial likelihood (p(x|F_hat)). See PairTree (Wintersinger et. al.)
        supplement section 2.2 for details.
    '''

    N = R + V
    S, K = F_hat.shape

    for matrix in V, N, omega_v:
        assert(matrix.shape == (S, K-1))

    P = torch.mul(omega_v, F_hat[:,1:])

    bin_dist = Binomial(N, P)
    F_llh = bin_dist.log_prob(V) / np.log(2)
    assert(not torch.any(F_llh.isnan()))
    assert(not torch.any(F_llh.isinf()))

    llh_per_sample = -torch.sum(F_llh, axis=1) / S
    nlglh = torch.sum(llh_per_sample) / (K-1)
    return nlglh

def compute_u_loss(psi, ref, var, omega, B, weights):
    '''
    Args:
        psi: raw values we are estimating of matrix U (num_sites x num_internal_nodes)
        ref: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        omega: VAF to subclonal frequency correction 
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        weights: Weights object

    Returns:
        Loss to score the estimated proportions of each clone in each site
    '''
    
    # Using the softmax enforces that the row sums are 1, since the proprtions of
    # clones in a given site should sum to 1
    U = torch.softmax(psi, dim=1)
    # print("psi", psi)
    #print("U", U)

    # 1. Data fit
    F_hat = (U @ B)
    nlglh = calc_llh(F_hat, ref, var, omega)
    # 2. Regularization to make some values of U -> 0
    reg = torch.sum(psi) # l1 norm 
    # print("weights.reg", weights.reg, "reg", reg)
    # print("weights.data_fit", weights.data_fit, "nlglh", nlglh)
    clone_proportion_loss = (weights.data_fit*nlglh + weights.reg*reg)
    loss_components = {DATA_FIT_KEY: round(nlglh.item(), 3), REG_KEY: reg.item()}
    return U, clone_proportion_loss, loss_components

def no_cna_omega(shape):
    '''
    Returns omega values assuming no copy number alterations (0.5)
    Shape is (num_anatomical_sites x num_mutation_clusters)
    '''
    return torch.ones(shape) * 0.5

def sample_gumbel(shape, eps=1e-20):
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

def compute_v_loss(X, L, T, p, G, O, v_temp, t_temp, weights, update_path_matrix, 
                   fixed_labeling, poly_res=None):
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
    V = stack_vertex_labeling(L, softmax_X, p, poly_res, fixed_labeling)

    bs = X.shape[0]
    if poly_res != None:
        softmax_pol_res, _ = gumbel_softmax(poly_res.latent_var, t_temp)
        T = vutil.repeat_n(T, bs)
        T[:,:,poly_res.children_of_polys] = softmax_pol_res
    else:
        T = vutil.repeat_n(T, bs)

    if G != None:
        G = vutil.repeat_n(G, T.shape[0])
    loss, loss_dicts = clone_tree_labeling_objective(V, softmax_X_soft, T, G, O, p, weights, update_path_matrix)
    return V, loss, loss_dicts, softmax_X_soft, T
    
def get_avg_loss_components(loss_components):
    '''
    Calculate the averages of each loss component (e.g. "nll" 
    (negative log likelihood), "mig" (migration number), etc.)
    '''
    d = {}
    for key in loss_components:
        if isinstance(loss_components[key], float) or isinstance(loss_components[key], int):
            d[key] = loss_components[key]
        else:
            d[key] = torch.mean(loss_components[key])
    return d

def expand_solutions(best_Vs, best_soft_Vs, best_Ts, p):
    '''
    In hard (i.e. usually large input) cases where we are unable to find a 
    primary-only seeding solution, see if we can recover one by post-processing
    final solutions and removing any met-to-met migration edges, and add these
    to our final solution set
    '''
    expanded_best_Ts, expanded_best_soft_Vs, expanded_best_Vs = [],[],[]
    for T,soft_V,V in zip(best_Ts, best_soft_Vs, best_Vs):
        expanded_best_Ts.append(T)
        expanded_best_soft_Vs.append(soft_V)
        expanded_best_Vs.append(V)
        clusters = putil.seeding_clusters(V,T)
        new_V = copy.deepcopy(V)
        for s in clusters:
            new_V[:,s] = p.T 
        expanded_best_Ts.append(T)
        expanded_best_soft_Vs.append(soft_V)
        expanded_best_Vs.append(new_V)
    
    return expanded_best_Vs, expanded_best_soft_Vs, expanded_best_Ts

def prune_histories(solutions, O, p):
    '''
    Only keep the Pareto front of trees
    '''

    # Collect each solution's parsimony metrics
    all_pars_metrics = []
    for soln in solutions:
        V, T, G = soln.V, soln.T, soln.G
        m, c, s, _, _ = vutil.ancestral_labeling_metrics(vutil.add_batch_dim(V), T, G, O, p, True)
        all_pars_metrics.append((int(m), int(c), int(s)))
    #print("all pars metrics", set(all_pars_metrics))
    pareto_metrics, pruned_solutions = vutil.pareto_front(solutions, all_pars_metrics)
    print("pareto_metrics", set(pareto_metrics))
    return pruned_solutions

def _get_best_final_solutions(best_Vs, best_soft_Vs, Ts, Gs, O, p, weights, print_config, 
                              solve_polys, idx_label_dicts, needs_pruning=True):
    '''
    Return the sorted, top k solutions
    '''

    # 1. Calculate loss for each solution
    solutions = []
    # If we're not doing polytomy resolution, we can calculate ancestral labeling objectives in parallel (faster)
    # since all Ts and Gs are the same size
    if solve_polys == False:
        losses, _ = clone_tree_labeling_objective(vutil.to_tensor(best_Vs), vutil.to_tensor(best_soft_Vs), vutil.to_tensor(Ts), torch.stack(Gs), O, p, weights, True)
        for i,loss in enumerate(losses):    
            solutions.append(vutil.VertexLabelingSolution(loss, best_Vs[i], best_soft_Vs[i], Ts[i], Gs[i], idx_label_dicts[i], i))

    else:
        for i, (V, soft_V, T, G, idx_label_dict) in enumerate(zip(best_Vs, best_soft_Vs, Ts, Gs, idx_label_dicts)):
            reshaped_V = vutil.add_batch_dim(V)
            reshaped_soft_V = vutil.add_batch_dim(soft_V)        
            loss, _ = clone_tree_labeling_objective(reshaped_V, reshaped_soft_V, T, G, O, p, weights, True)
            solutions.append(vutil.VertexLabelingSolution(loss, V, soft_V, T, G, idx_label_dict, i))
        
    if needs_pruning:
        # 2. Prune bad histories (anything not in the pareto front)
        solutions = prune_histories(solutions, O, p)
        # 3. Make the solutions unique
        unique_solutions = set()
        final_solutions = []
        for soln in solutions:
            tree = vutil.LabeledTree(soln.T, soln.V)
            if tree not in unique_solutions:
                final_solutions.append(soln)
                unique_solutions.add(tree)
        
    else:
        final_solutions = solutions

    # 4. Sort the solutions from lowest to highest loss
    final_solutions = sorted(list(final_solutions))

    # 5. Return the k best solutions
    k = print_config.k_best_trees if len(final_solutions) >= print_config.k_best_trees else len(final_solutions)
    final_solutions = final_solutions[:k]
    return final_solutions

def get_best_final_solutions(best_Vs, best_soft_Vs, best_Ts, G, O, p, weights, print_config, 
                             poly_res, node_idx_to_label, optimal_subtree_nodes, needs_pruning=True):
    '''
    Prune unecessary poly nodes (if they weren't used) and return the top k solutions
    '''
    best_Vs, best_soft_Vs, best_Ts = expand_solutions(best_Vs, best_soft_Vs, best_Ts, p)
    # 1. Make solutions unique before we do all this remove node work which is time intensive
    unique_solutions = set()
    unique_best_Vs, unique_best_soft_Vs, unique_best_Ts = [],[], []
    for V,soft_V, T in zip(best_Vs, best_soft_Vs, best_Ts):
        tree = vutil.LabeledTree(T,V)
        if tree not in unique_solutions:
            unique_best_Vs.append(V)
            unique_best_soft_Vs.append(soft_V)
            unique_best_Ts.append(T)
            unique_solutions.add(tree)
    print("unique solutions", len(unique_solutions))
    
    # 2. Remove any extra resolver nodes that don't actually help
    best_Vs, best_Ts, Gs, idx_label_dicts = prutil.remove_extra_resolver_nodes(unique_best_Vs, unique_best_Ts, node_idx_to_label, G, poly_res, p, optimal_subtree_nodes)
    solve_polys = True if poly_res != None else False
    return _get_best_final_solutions(best_Vs, unique_best_soft_Vs, best_Ts, Gs, O, p, weights, print_config, 
                                     solve_polys, idx_label_dicts, needs_pruning)

def init_X(batch_size, num_sites, num_nodes_to_label, bias_weights, p, input_T, T, idx_to_observed_sites):

    nodes_w_children, biased_sites = vutil.get_k_or_more_children_nodes(input_T, T, idx_to_observed_sites, 1, True, 1, cutoff=False)
    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(batch_size, num_sites, num_nodes_to_label)
    if bias_weights:
        eta = 3 # TODO: how do we set this parameter?
        # Make 4 partitions: (1) biased towards the primary, (2) biased towards the primary + sites of children/grandchildren
        # (3) biased towards sites of children/grandchildren, (4) no bias
        quart = batch_size // 4
        
        # Bias for partitions [1-2]
        prim_site_idx = torch.nonzero(p)[0][0]
        # This is really important to prevent large trees from getting stuck in local optima
        X[:quart*2,prim_site_idx,:] = eta / 2
        #X[:quart*2,prim_site_idx,:] = eta - 1

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
    # TODO: should we optimize polytomies a second time?
    if second_optimization:
        return itr > max_iter*2/5 and itr < max_iter*3/5
    # #     return False
    #     
    return itr > max_iter*1/3 and itr < max_iter*2/3

def remove_leaf_nodes_idx_to_label_dicts(dicts):
    '''
    After running migration history inference, the leaf nodes 
    (U nodes) get added to the idx to label dicts when plotting 
    and saving to pickle files, so we don't want to do that twice
    when in calibrate mode
    '''
    new_dicts = []
    for i,dct in enumerate(dicts):
        new_dicts.append(copy.deepcopy(dct))
        for key in dct:
            if dct[key][1] == True:
                del new_dicts[i][key]
            else:
                new_dicts[i][key] = new_dicts[i][key][0]
    return new_dicts

def _convert_list_of_numpys_to_tensors(lst):
        return [torch.tensor(x) for x in lst]

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
        get_migration_history(T, pooled_tsv_fn, primary_site, weights, print_config, output_dir, f"{run_name}_{primary_site}", 
                              O=O, batch_size=batch_size, custom_colors=custom_colors, bias_weights=bias_weights, 
                              mode="evaluate", solve_polytomies=solve_polytomies, estimate_observed_clones=estimate_observed_clones)
    if estimate_observed_clones:
        os.remove(pooled_tsv_fn) # cleanup pooled tsv

# TODO: should this take input more similar to the calibrate input
def calibrate_label_clone_tree(tree_fns, tsv_fns, print_config, output_dir, run_names,
                               Os, batch_size, custom_colors, bias_weights, solve_polytomies):
    '''
    Observed clone proportions are inputted (in tsv_fns), only labeling of clone tree is needed
    '''
    return calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names,
                    Os, batch_size, custom_colors, bias_weights, solve_polytomies, estimate_observed_clones=False)

def calibrate(tree_fns, tsv_fns, print_config, output_dir, run_names,
              Os, batch_size, custom_colors, bias_weights, solve_polytomies, estimate_observed_clones=True):
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
            get_migration_history(Ts[i], pooled_tsv_fns[i], primary_site, weights, print_config, calibrate_dir, f"{run_names[i]}_{primary_site}", 
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
    weights = met.Weights(mig=[best_theta[0]*50], comig=best_theta[1]*50, seed_site=[best_theta[2]*50],
                          gen_dist=1.0, organotrop=1.0)
    
    # 4. Use the saved trees to rescore trees, visualize, and re-save 
    for i in range(len(Ts)):
        O = Os[i] if Os != None else None
        primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        for primary_site in primary_sites:
            run_name = f"{run_names[i]}_{primary_site}"
            with gzip.open(os.path.join(calibrate_dir, f"{run_name}.pkl.gz"), 'rb') as f:
                pckl = pickle.load(f)
            saved_Ts = _convert_list_of_numpys_to_tensors(pckl[OUT_ADJ_KEY])
            saved_Vs = _convert_list_of_numpys_to_tensors(pckl[OUT_LABElING_KEY])
            saved_soft_Vs = _convert_list_of_numpys_to_tensors(pckl[OUT_SOFTV_KEY])
            saved_U = torch.tensor(pckl[OUT_OBSERVED_CLONES_KEY])
            saved_Gs = _convert_list_of_numpys_to_tensors(pckl[OUT_GEN_DIST_KEY])
            saved_idx_to_label_dicts = remove_leaf_nodes_idx_to_label_dicts(pckl[OUT_IDX_LABEL_KEY])
            primary_sites = dutil.get_primary_sites(pooled_tsv_fns[i])
        
            primary_idx = ordered_sites[i].index(primary_site)
            p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites[i])).T
            final_solutions = _get_best_final_solutions(saved_Vs, saved_soft_Vs, saved_Ts, saved_Gs, O, p, weights,
                                                        print_config, solve_polytomies, saved_idx_to_label_dicts,
                                                        needs_pruning=False)
            
            putil.save_best_trees(final_solutions, saved_U, O,
                                  weights, ordered_sites[i], print_config, 
                                  custom_colors, primary_site, calibrate_dir, 
                                  run_name)
    
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

def initalize_optimal_X_polyres(optimal_subtree_nodes,optimal_batch_nums, optimized_Ts, optimized_Vs, 
                                num_sites, num_nodes_to_label, bias_weights, p, 
                                input_T, T_w_leaves, idx_to_observed_sites, poly_res):
    
    bs = optimized_Vs.shape[0]
    X = init_X(bs, num_sites, num_nodes_to_label, bias_weights, p, input_T, T_w_leaves, idx_to_observed_sites)
    poly_resolver_to_optimal_children = {}
    known_indices = []
    known_labelings = []
    #print("optimal subtree sets", optimal_subtree_nodes)
    # Fix node labels and node edges
    for optimal_subtree_set,optimal_batch_num in zip(optimal_subtree_nodes,optimal_batch_nums):
        for node_idx in optimal_subtree_set:
            # If this is a witness node from U or the root index, we already know its vertex labeling
            if node_idx <= X.shape[2] and node_idx != 0:
                optimal_site = int(optimized_Vs[optimal_batch_num,:,node_idx].nonzero(as_tuple=False))
                idx = node_idx - 1 # X doesn't include root node
                known_indices.append(idx)
                known_labelings.append(torch.eye(num_sites)[optimal_site].T)
                X[:,optimal_site,idx] = 1
                non_optimal_sites = [i for i in range(num_sites) if i != optimal_site]
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
    #print("poly_resolver_to_optimal_children", poly_resolver_to_optimal_children)
    # Fix all other polytomy children s.t. they cannot move to be a child of the fixed node_idx,
    if poly_res != None:
        for parent_idx in poly_resolver_to_optimal_children:
            
            optimal_children = poly_resolver_to_optimal_children[parent_idx]
            optimal_children_poly_indices = [poly_res.children_of_polys.index(i) for i in optimal_children]
            other_children = [i for i in range(poly_res.latent_var.shape[2]) if i not in optimal_children_poly_indices]

            poly_res.latent_var[:,parent_idx,other_children] = float("-inf")
        poly_res.latent_var.requires_grad = True
    
    fixed_labeling = None
    if len(known_indices) != 0:
        unknown_indices = [x for x in range(num_nodes_to_label) if x not in known_indices]
        known_labelings = torch.stack(known_labelings, dim=1)
        X = X[:,:,unknown_indices] # only include the unknown indices for inference
        fixed_labeling = vutil.FixedVertexLabeling(known_indices, unknown_indices, known_labelings)
        #print("known_indices", known_indices)
        #print("unknown_indices", unknown_indices)
    X.requires_grad = True
    return X, poly_res, fixed_labeling

def fix_optimal_subtrees(optimized_Ts, optimized_Vs, num_sites, num_nodes_to_label, bias_weights, p, 
                         input_T, T_w_leaves, idx_to_observed_sites, poly_res):
    '''
    After the first round of optimization, there are optimal subtrees (subtrees where
    the labelings of *all* nodes is the same), which we can keep fixed, since there
    are no other more optimal labelings rooted at this branch.

    Two things we can fix: the labeling of the nodes in optimal subtrees,
    and the edges of the subtrees if polytomy resolution is being used. Search all
    samples to find optimal subtrees, since there might not be one solution with all 
    optimal subtrees.
    '''
    
    num_internal_nodes = num_nodes_to_label + 1 # root node
    
    # Re-initialize optimized_Ts with the tree with the best subtree structure
    if poly_res != None:
        poly_res.latent_var.requires_grad = False
        num_internal_nodes += len(poly_res.resolver_indices)
        # print("poly_res.children_of_polys", poly_res.children_of_polys)
        # print("poly_res.resolver_indices", poly_res.resolver_indices)
        # print("poly_res.resolver_index_to_parent_idx",poly_res.resolver_index_to_parent_idx)
    
    # 1. Find samples with optimal subtrees
    optimal_subtree_nodes, optimal_batch_nums = find_optimal_subtree_nodes(optimized_Ts, optimized_Vs, num_internal_nodes)

    # 3. Re-initialize X and polytomy resolver with optimal subtrees (labelings and structure) fixed. 
    X, poly_res, fixed_labeling = initalize_optimal_X_polyres(optimal_subtree_nodes, optimal_batch_nums, optimized_Ts, optimized_Vs, 
                                                              num_sites, num_nodes_to_label, bias_weights, p, 
                                                              input_T, T_w_leaves, idx_to_observed_sites, poly_res)
            
    return X, poly_res, fixed_labeling, optimal_subtree_nodes

def optimize_umap(num_sites, num_internal_nodes, ref, var, omega, B, T, G, weights, lr, identical_clone_gen_dist):
    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and anatomical locations of the extant clones (U > U_CUTOFF)
    #psi = -1 * torch.rand(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi = torch.ones(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi.requires_grad = True 
    u_optimizer = torch.optim.Adam([psi], lr=lr)

    i = 0
    u_prev = psi
    u_diff = 1e9
    losses = []
   
    while u_diff > 1e-6 and i < 300:
        u_optimizer.zero_grad()
        U, u_loss, loss_dict = compute_u_loss(psi, ref, var, omega, B, weights)
        u_loss.backward()
        u_optimizer.step()
        u_diff = torch.abs(torch.norm(u_prev - U))
        u_prev = U
        i += 1
        losses.append(u_loss.detach().numpy())
    
    T, G, L, idx_to_observed_sites = vutil.full_adj_matrix_using_inferred_observed_clones(U, T, G, num_sites, identical_clone_gen_dist)

    return U, T, G, L, idx_to_observed_sites

def optimize_V(X, L, input_T, p, G, O, init_temp, final_temp, v_anneal_rate, 
               t_anneal_rate, v_interval, lr, weights, max_iter, solve_polytomies, 
               poly_res, print_config, second_optimization, fixed_labeling):
    
    v_optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = lr_scheduler.LinearLR(v_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)
    if solve_polytomies:
        poly_optimizer = torch.optim.Adam([poly_res.latent_var], lr=lr)
    v_temps, t_temps = [], []
    v_loss_components = []

    v_temp = init_temp
    t_temp = init_temp
    j = 0
    k = 0
    global PROGRESS_BAR
    for i in range(max_iter):
        update_path = update_path_matrix(i, max_iter, solve_polytomies, second_optimization)
        if solve_polytomies and update_path:
            poly_optimizer.zero_grad()
        v_optimizer.zero_grad()
        V, v_losses, loss_comps, soft_V, T = compute_v_loss(X, L, input_T, p, G, O, v_temp, t_temp, weights, 
                                                            update_path, fixed_labeling, poly_res)
        mean_loss = torch.mean(v_losses)
        mean_loss.backward()

        if solve_polytomies and update_path:
            poly_optimizer.step()
            if i % 5 == 0:
                t_temp = np.maximum(t_temp * np.exp(-t_anneal_rate * j), final_temp)
            j += 1

        else:
            v_optimizer.step()
            scheduler.step()

            if i % v_interval == 0:
                v_temp = np.maximum(v_temp * np.exp(-v_anneal_rate * k), final_temp)
            k += 1

        if print_config.visualize:
            v_loss_components.append(get_avg_loss_components(loss_comps))

        v_temps.append(v_temp)
        t_temps.append(t_temp)

        PROGRESS_BAR.update(1)
    
    # fig = plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(v_temps))], v_temps, marker='.'); plt.show(); plt.close()
    # fig = plt.figure(figsize=(1,1),dpi=100)
    # plt.plot([x for x in range(len(t_temps))], t_temps, marker='.'); plt.show(); plt.close()

    return V, soft_V, T, loss_comps
    
def get_migration_history(T, tsv_fn, primary_site, weights, print_config, output_dir, run_name, estimate_observed_clones=True,
                          O=None, max_iter=100, lr=0.05, init_temp=20, final_temp=0.01,
                          batch_size=-1, custom_colors=None, bias_weights=True, mode="evaluate",
                          solve_polytomies=False, needs_pruning=True):
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
    num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling
    primary_site_label = ordered_sites[torch.nonzero(p)[0][0]]

    # Initialize global, cached path matrix as None (if module is not reloaded in between patients, this can cause problems)
    vutil.LAST_P = None

    identical_clone_gen_dist = 0.0
    if G != None:
        identical_clone_gen_dist = torch.min(G[(G != 0)])/2.0

    B = vutil.mutation_matrix(T)
    # Add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # Add a column of ones to indicate that every clone is a descendent of the non-cancerous root node
    B = torch.hstack([torch.ones(B.shape[0]).reshape(-1,1), B])

    start_time = datetime.datetime.now()

    # Keep a copy of input clone tree (T from now on has leaf nodes from U)
    input_T = copy.deepcopy(T)

    ############ Step 1, find a MAP estimate of U ############
    if estimate_observed_clones:
        U, T, G, L, idx_to_observed_sites = optimize_umap(num_sites, num_internal_nodes, ref, var, omega, B, T, G, weights, lr, identical_clone_gen_dist)
        #vutil.print_U(U, B, node_idx_to_label, ordered_sites, ref, var)
        #print("idx_to_observed_sites", idx_to_observed_sites)
    else:
        # Add inputted observed clones as leaf nodes to T
        U = None
        T, G, L = vutil.full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, G, idx_to_observed_sites, num_sites, identical_clone_gen_dist)
    num_leaves = L.shape[1]

    # If solving for polytomies, setup T and G s.t. 
    if solve_polytomies:
        nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(input_T, T, idx_to_observed_sites, 3, True, 2)
        if len(nodes_w_polys) == 0:
            print("No potential polytomies to solve, not resolving polytomies.")
            poly_res, solve_polytomies = None, False
        else:
            poly_res = prutil.PolytomyResolver(T, G, num_sites, num_leaves, batch_size, node_idx_to_label, nodes_w_polys, resolver_sites, identical_clone_gen_dist)
            T, G, node_idx_to_label = poly_res.T, poly_res.G, poly_res.node_idx_to_label
    else:
        poly_res = None
    
    ############ Step 2, sample from V to estimate q(V) ############

    # We're learning X, which is the vertex labeling of the internal nodes
    X = init_X(batch_size, num_sites, num_nodes_to_label, bias_weights, p, input_T, T, idx_to_observed_sites)
    X.requires_grad = True

    # Temperature and annealing
    v_anneal_rate = 0.01
    t_anneal_rate = 0.01
    
    # Calculate V only using maximum parsimony metrics
    if mode == 'calibrate':
        # Make a copy of inputs for later
        G_copy = copy.deepcopy(G)
        O_copy = copy.deepcopy(O)
        G, O = None, None

    input_weights = copy.deepcopy(weights)
    # Only use maximum parsimony metrics when initially searching for high likelihood trees
    exploration_weights = met.Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                                      seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, data_fit=weights.data_fit, 
                                      reg=weights.reg, entropy=weights.entropy, gen_dist=0.0, organotrop=0.0)
    global PROGRESS_BAR
    first_max_iter, second_max_iter = 66, max_iter
    PROGRESS_BAR = tqdm(total=first_max_iter+second_max_iter, position=0)
    optimzed_Vs, optimzed_soft_Vs, optimzed_Ts, _ = optimize_V(X, L, T, p, G, O, init_temp, final_temp, v_anneal_rate, 
                                                               t_anneal_rate, 15, lr, exploration_weights, first_max_iter, solve_polytomies, 
                                                               poly_res, print_config, False, None)
    
    # Identify optimal subtrees, keep them fixed, and solve for the rest of the tree
    # Also fix T (hooked to poly_res) if we're solving polytomies
    X, poly_res, fixed_labeling, optimal_subtree_nodes = fix_optimal_subtrees(optimzed_Ts, optimzed_Vs, num_sites, num_nodes_to_label, 
                                                                     bias_weights, p, input_T, T, idx_to_observed_sites, poly_res) 
    optimzed_Vs, optimzed_soft_Vs, optimzed_Ts, _ = optimize_V(X, L, T, p, G, O, init_temp, final_temp, v_anneal_rate, 
                                                               t_anneal_rate, 20, lr, exploration_weights, second_max_iter, False, 
                                                               poly_res, print_config, True, fixed_labeling)
    
    PROGRESS_BAR.close()
    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")
        
    ############ Step 3, visualize and save outputs ############
    with torch.no_grad():
        if mode == "calibrate":
            # Reload inputs
            G, O = G_copy, O_copy
        weights = input_weights
        
        final_solutions = get_best_final_solutions(optimzed_Vs, optimzed_soft_Vs, optimzed_Ts, G, O, p, weights,
                                                   print_config, poly_res, node_idx_to_label, optimal_subtree_nodes, needs_pruning=needs_pruning)

        print("# final solutions:", len(final_solutions))

        # if print_config.visualize:
        #     putil.plot_loss_components(v_loss_dict, weights)
            
        edges, vert_to_site_map, mig_graph_edges, loss_info = putil.save_best_trees(final_solutions, U, O, weights,
                                                                                    ordered_sites,print_config, custom_colors, 
                                                                                    primary_site_label, output_dir, run_name,
                                                                                    original_root_idx=original_root_idx)   

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed

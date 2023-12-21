import torch
import sys
import numpy as np
import datetime
from tqdm import tqdm
import heapq
import torchopt

from metient import metient as met
from metient.util import vertex_labeling_util as vutil
from metient.util import eval_util as eutil
from metient.util import plotting_util as plot_util
import torch.optim.lr_scheduler as lr_scheduler
from metient.util.globals import *

from torch.distributions.binomial import Binomial

import os
import csv
import copy
import shutil
import pickle
import gzip
import json

G_IDENTICAL_CLONE_VALUE = None

# gumbel noise is initialized non-randomly for latent variables
RANDOM_VALS = None
LAST_P = None
SOLVE_POLYS = False

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def add_batch_dim(x):
    return x.reshape(1, x.shape[0], x.shape[1])

def get_ancestral_labeling_metrics(V, A, G, O, p, i=-1, max_iter=-1):
    single_A = A
    bs = V.shape[0]
    num_sites = V.shape[1]
    num_nodes = V.shape[2]
    A = A if len(A.shape) == 3 else vutil.repeat_n(single_A, bs) 

    # 1. Migration number
    VA = V @ A
    VT = torch.transpose(V, 2, 1)
    site_adj = VA @ VT
    site_adj_trace = torch.diagonal(site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    m = torch.sum(site_adj, dim=(1, 2)) - site_adj_trace

    # 2. Seeding site number
    # remove the same site transitions from the site adj matrix
    site_adj_no_diag = torch.mul(site_adj, vutil.repeat_n(1-torch.eye(num_sites, num_sites), bs))
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=2)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    
    binarized_row_sums_site_adj = torch.sigmoid(BINARY_ALPHA * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj, dim=(1))

    # 3. Comigration number
    VAT = torch.transpose(VA, 2, 1)
    W = VAT @ VA # 1 if two nodes' parents are the same color
    X = VT @ V # 1 if two nodes are the same color
    Y = torch.sum(torch.mul(VAT, 1-VT), axis=2) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X) # 1 if two nodes' parents are same color AND nodes are same color
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    global LAST_P # this is expensive to compute, so hash it if we don't need to update it
    if LAST_P != None and not update_adj_matrix(i, max_iter):
        P = LAST_P
    else:
        P = vutil.get_path_matrix_tensor(A)
        LAST_P = P

    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=2)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y), axis=1)
    binarized_site_adj = torch.sigmoid(BINARY_ALPHA * (2 * site_adj - 1))
    bin_site_trace = torch.diagonal(binarized_site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    c = torch.sum(binarized_site_adj, dim=(1,2)) - bin_site_trace + repeated_temporal_migrations

    # 4. Genetic distance
    g = 0
    if G != None:
        #print("G\n", G[0,:,:])
        # calculate if 2 nodes are in diff sites and there's an edge between them (i.e there is a migration edge)
        R = torch.mul(A, (1-X))
        adjusted_G = -torch.log(G+0.01)
        R = torch.mul(R, adjusted_G)
        # if bs == 1:
        #     nonzero_indices =  torch.nonzero(R[0])
        #     print("gen dist")
        #     print("R\n", R)
        #     print(nonzero_indices)
        #     print(R[0][nonzero_indices])

        # TODO: get the average of genetic distance scores or normalize by max?
        g = torch.sum(R, dim=(1,2))/(m)

    # 5. Organotropism
    o = 0
    if O != None:
        # the organotropism frequencies can only be used on the first 
        # row, which is for the migrations from primary cancer site to
        # other metastatic sites (we don't have frequencies for every 
        # site to site migration)
        prim_site_idx = torch.nonzero(p)[0][0]
        O = O.repeat(bs,1).reshape(bs, O.shape[0])
        adjusted_freqs = -torch.log(O+0.01)
        organ_penalty = torch.mul(site_adj_no_diag[:,prim_site_idx,:], adjusted_freqs)
        #print(organ_penalty[0])
        o = torch.sum(organ_penalty, dim=(1))/(num_sites-1)

    return m, c, s, g, o

def get_entropy(V, soft_V):
    return -torch.sum(torch.mul(soft_V, torch.log2(soft_V)), dim=(1, 2)) / V.shape[2]

def get_repeating_weight_vector(bs, weight_list):
    repeat_times = bs // len(weight_list)

    # Repeat each number in the list the calculated number of times
    weights_vec = torch.tensor(weight_list * repeat_times)

    # If n is not a multiple of the length of the list, repeat the elements as necessary
    remaining_elements = bs % len(weight_list)
    if remaining_elements > 0:
        weights_vec = torch.cat([weights_vec, torch.tensor(weight_list[:remaining_elements])])

    return weights_vec

def get_mig_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.mig)

def get_seed_site_weight_vector(bs, weights):
    return get_repeating_weight_vector(bs, weights.seed_site)

def ancestral_labeling_objective(V, soft_V, A, G, O, p, weights, i=-1, max_iter=-1):
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
    bs = V.shape[0]
    
    #assert(A.shape[0]==A.shape[1]==V.shape[2])
    m, c, s, g, o = get_ancestral_labeling_metrics(V, A, G, O, p, i, max_iter)

    # Entropy
    e = get_entropy(V, soft_V)

    num_sites, num_nodes = V.shape[1], V.shape[2]

    # TODO: should we normalize??
    #max_comigrations = num_sites**2
    # normalized_migrations = m/(num_nodes - 1)
    # normalized_comigrations = c/max_comigrations
    # normalized_seeding_sites = s/(num_sites)

    # normalized_migrations = m
    # normalized_comigrations = c
    # normalized_seeding_sites = s

    # normed_averaged_migrations = (weights.mig_delta*normalized_migrations) + ((1-weights.mig_delta)*normalized_comigrations)

    # averaged_migrations = (weights.mig_delta*m) + ((1-weights.mig_delta)*c)

    # Combine all 5 components with their weights
    # Explore different weightings
    if isinstance(weights.mig, list) and isinstance(weights.seed_site, list):
        mig_weights_vec = get_mig_weight_vector(bs, weights)
        seeding_sites_weights_vec = get_seed_site_weight_vector(bs, weights)
        mig_loss = torch.mul(mig_weights_vec, m)
        seeding_loss = torch.mul(seeding_sites_weights_vec, s)
        vertex_labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
        
    else:
        mig_loss = weights.mig*m
        seeding_loss = weights.seed_site*s
        vertex_labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)

    loss_components = {MIG_KEY: m, COMIG_KEY:c, SEEDING_KEY: s, ORGANOTROP_KEY: o, GEN_DIST_KEY: g, ENTROPY_KEY: e}

    return vertex_labeling_loss, loss_components

# Adapted from PairTree
def calc_llh(F_hat, R, V, omega_v, epsilon=1e-5):
    '''
    Args:
        F_hat: estimated subclonal frequency matrix (num_nodes x num_mutation_clusters)
        R: Refernce matrix (num_samples x num_mutation_clusters)
        V: Variant matrix (num_samples x num_mutation_clusters)
    Returns:
        Data fit using the Binomial likelihood (p(x|F_hat)). See PairTree
        supplement section 2.2 for details.
    '''

    N = R + V
    S, K = F_hat.shape

    for matrix in V, N, omega_v:
        assert(matrix.shape == (S, K-1))

    # TODO: how do we make sure the non-cancerous root clone subclonal frequencies here are 1
    #assert(np.allclose(1, phi[0]))
    P = torch.mul(omega_v, F_hat[:,1:])

    # TODO: why these cutoffs?
    #P = torch.maximum(P, epsilon)
    #P = torch.minimum(P, 1 - epsilon)

    bin_dist = Binomial(N, P)
    F_llh = bin_dist.log_prob(V)
    #phi_llh = stats.binom.logpmf(V, N, P) / np.log(2)
    assert(not torch.any(F_llh.isnan()))
    assert(not torch.any(F_llh.isinf()))

    # TODO: why the division by K-1 and S?
    llh_per_sample = -torch.sum(F_llh, axis=1) / (K-1)
    nlglh = torch.sum(llh_per_sample) / S
    return (F_llh, llh_per_sample, nlglh)

def subclonal_presence_objective(ref, var, omega_v, U, B, weights):
    '''
    Args:
        ref: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        U: Mixture matrix (num_sites x num_internal_nodes)
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        weights: Weights object

    Returns:
        Loss to score the ancestral vertex labeling of the given tree. This combines (1) migration number, (2) seeding site
        number, (3) comigration number, and optionally (4) genetic distance and (5) organotropism.
    '''
    # 1. Data fit
    F_hat = (U @ B)
    F_llh, llh_per_sample, nlglh = calc_llh(F_hat, ref, var, omega_v)

    # 2. Regularization to make some values of U -> 0
    reg = torch.sum(U)

    subclonal_presence_loss = (weights.data_fit*nlglh + weights.reg*reg)
    loss_components = {DATA_FIT_KEY: round(nlglh.item(), 3), REG_KEY: reg.item()}
    return subclonal_presence_loss, loss_components

def no_cna_omega(shape):
    '''
    Returns omega values assuming no copy number alterations (0.5)
    Shape is (num_anatomical_sites x num_mutation_clusters)
    '''
    # TODO: don't hardcode omega here (omegas are all 1/2 since we're assuming there are no CNAs)
    return torch.ones(shape) * 0.5

def get_random_vals_fixed_seeds(shape):
    global RANDOM_VALS
    if RANDOM_VALS != None and shape in RANDOM_VALS:
        return RANDOM_VALS[shape]

    if RANDOM_VALS == None:
        RANDOM_VALS = dict()

    rands = torch.zeros(shape)
    for i in range(shape[0]):
        torch.manual_seed(i)
        rands[i] = torch.rand(shape[1:])
    RANDOM_VALS[shape] = rands
    return RANDOM_VALS[shape]

def sample_gumbel(shape, eps=1e-20):
    G = get_random_vals_fixed_seeds(shape)
    return -torch.log(-torch.log(G + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature, hard=False):
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

def full_adj_matrix(U, T, G):
    '''
    All non-zero values of U represent extant clones (leaf nodes of the full tree).
    For each of these non-zero values, we add an edge from parent clone to extant clone.
    '''
    U = U[:,1:] # don't include column for normal cells
    num_leaves = (U > U_CUTOFF).nonzero().shape[0]
    num_internal_nodes = T.shape[0]
    full_adj = torch.nn.functional.pad(input=T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
    leaf_idx = num_internal_nodes
    # Also add branch lengths (genetic distances) for the edges we're adding.
    # Since all of these edges we're adding represent genetically identical clones,
    # we are going to add a very small but non-zero value.
    full_G = torch.nn.functional.pad(input=G, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0) if G is not None else None

    # Iterate through the internal nodes that have nonzero values
    for internal_node_idx in (U > U_CUTOFF).nonzero()[:,1]:
        full_adj[internal_node_idx, leaf_idx] = 1
        if G is not None:
            full_G[internal_node_idx, leaf_idx] = G_IDENTICAL_CLONE_VALUE
        leaf_idx += 1
    return full_adj, full_G, num_leaves
    
def compute_u_loss(psi, ref, var, B, weights):
    '''
    Computes loss for latent variable U (dim: num_internal_nodes x num_sites)
    '''
    # Using the softmax enforces that the row sums are 1, since the proprtions of
    # subclones in a given site should sum to 1
    U = torch.softmax(psi, dim=1)
    batch_size = U.shape[0]
    omega_v = no_cna_omega(ref.shape)
    u_loss, loss_dict_b = subclonal_presence_objective(ref, var, omega_v, U, B, weights)
    return U, u_loss

def get_full_G(T, G, poly_res, i=-1, max_iter=-1):
    '''
    If resolving polytomies, dynamically calculate branch lengths between
    polytomy nodes and the "resolver" nodes as the average of the branch 
    lengths of the resolver nodes' children
    '''
    if G == None:
        return None

    if poly_res == None:
        return vutil.repeat_n(G, T.shape[0])

    full_G = vutil.repeat_n(G, T.shape[0])
    # !!TODO: What should the genetic distance be??
    # resolver_indices = poly_res.resolver_indices
    # for batch_idx in range(T.shape[0]):
    #     for res_idx in resolver_indices:
    #         parent_idx = poly_res.resolver_index_to_parent_idx[res_idx]
    #         res_children = vutil.get_child_indices(T[batch_idx], [res_idx])
    #         # no children for this resolver node, so keep the original branch length
    #         if len(res_children) == 0:
    #             avg = G[parent_idx, res_idx]
    #         else:
    #             avg = torch.mean(G[parent_idx, res_children])
    #         full_G[batch_idx, parent_idx, res_idx] = avg
    # LAST_G = full_G
    return full_G

def compute_v_loss(U, X, T, ref, var, p, G, O, v_temp, t_temp, hard, weights, i, max_iter, poly_res=None):
    '''
    Computes loss for X (dim: batch_size x num_internal_nodes x num_sites)
    '''

    def stack_vertex_labeling(U, X, p, T):
        '''
        Use U and X (both of size batch_size x num_internal_nodes x num_sites)
        to get the anatomical sites of the leaf nodes and the internal nodes (respectively). 
        Stack the root labeling to get the full vertex labeling V. 
        '''
        U = U[:,1:] # don't include column for normal cells
        num_sites = U.shape[0]
        L = torch.nn.functional.one_hot((U > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
        # Expand leaf node labeling L to be repeated batch_size times
        bs = X.shape[0]
        L = vutil.repeat_n(L, bs)
        
        if poly_res != None:
            # order is: given internal nodes, new poly nodes, leaf nodes from U)
            full_vert_labeling = torch.cat((X, vutil.repeat_n(poly_res.resolver_labeling, bs), L), dim=2)
        else:
            full_vert_labeling = torch.cat((X, L), dim=2)

        p = vutil.repeat_n(p, bs)
        # Concatenate the left part, new column, and right part along the second dimension
        return torch.cat((p, full_vert_labeling), dim=2)

    softmax_X, softmax_X_soft = gumbel_softmax(X, v_temp, hard)
    V = stack_vertex_labeling(U, softmax_X, p, T)

    bs = X.shape[0]
    if poly_res != None:
        softmax_pol_res, _ = gumbel_softmax(poly_res.latent_var, t_temp, hard)
        T = vutil.repeat_n(T, bs)
        T[:,:,poly_res.children_of_polys] = softmax_pol_res
    else:
        T = vutil.repeat_n(T, bs)

    if G != None:
        G = get_full_G(T, G, poly_res, i, max_iter)
    loss, loss_components = ancestral_labeling_objective(V, softmax_X_soft, T, G, O, p, weights, i, max_iter)
    return V, loss, loss_components, softmax_X_soft, T
    
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

class VertexLabelingSolution:
    def __init__(self, loss, V, soft_V, T, G, node_idx_to_label, mig_weight, comig_weight, seed_weight, i):
        self.loss = loss
        self.V = V
        self.soft_V = soft_V
        self.T = T
        self.G = G
        self.node_idx_to_label = node_idx_to_label
        self.mig_weight = mig_weight
        self.seed_weight = seed_weight
        self.comig_weight = comig_weight
        self.i = i

   # override the comparison operator
    def __lt__(self, other):
        return self.loss < other.loss


def get_weights_with_best_eval_loss(solutions, weights, O, p):
    '''
    Which parsimony weights give the best gen dist/organotrop
    '''    
    weights_to_loss = {}
    
    out = []
    for i, soln in enumerate(solutions):
        V, soft_V, T, G, mig_weight, seed_weight = soln.V, soln.soft_V, soln.T, soln.G, soln.mig_weight, soln.seed_weight
        reshaped_V = add_batch_dim(V)
        reshaped_soft_V = add_batch_dim(soft_V)
        eval_loss, loss_components = ancestral_labeling_objective(reshaped_V, reshaped_soft_V, T, G, O, p, weights)
        weights_at_i = (float(mig_weight), float(seed_weight))
        if weights_at_i not in weights_to_loss:
            weights_to_loss[weights_at_i] = []
        weights_to_loss[weights_at_i].append(float(eval_loss))
        out.append([float(mig_weight), float(seed_weight), float(eval_loss)])

    #print("weights_to_loss", weights_to_loss)
    softmaxed_scores = torch.softmax(torch.tensor([item[2] for item in out]), dim=0)
    weights_to_softmax_loss = dict()
    for score, item in zip(softmaxed_scores, out):
        weights_at_i = (item[0],item[1])
        if weights_at_i not in weights_to_softmax_loss:
            weights_to_softmax_loss[weights_at_i] = []
        weights_to_softmax_loss[weights_at_i].append(score)

    #print("weights_to_softmax_loss", weights_to_softmax_loss)
    weights_to_avg_loss = {k:(sum(weights_to_loss[k])/len(weights_to_loss[k])) for k in weights_to_loss}
    print("weights_to_avg_loss", weights_to_avg_loss)

    best_weights = None
    lowest_loss = min(list(weights_to_avg_loss.values()))

    best_weights = [k for k,v in weights_to_avg_loss.items() if v == lowest_loss][0]
    
    out.insert(0, ["mig weight", "seed weight", "gen dist loss"])
    return best_weights, out

def get_best_solutions(prev_v_losses, v_losses, Vs, soft_Vs, Ts, best_Vs, best_soft_Vs, best_Ts):
    '''
    Only update solutions where the loss has decreased
    '''
    # First iteration
    if best_Vs == None:
        return Vs, soft_Vs, Ts
    indices_to_update = torch.nonzero(v_losses < prev_v_losses).squeeze()
    best_Vs[indices_to_update] = Vs[indices_to_update]
    best_soft_Vs[indices_to_update] = soft_Vs[indices_to_update]
    best_Ts[indices_to_update] = Ts[indices_to_update]
    return best_Vs, best_soft_Vs, best_Ts

from collections import OrderedDict
import math

class PolytomyResolver():

    def __init__(self, input_T, T, G, U, num_leaves, bs, node_idx_to_label, nodes_w_polys, resolver_sites):
        '''
        This is post U matrix estimation, so T already has leaf nodes.
        '''
        
        # 1. nodes_w_polys are the nodes have polytomies
        #print("nodes_w_polys", nodes_w_polys, "resolver_sites", resolver_sites)
        # 2. Pad the adjacency matrix so that there's room for the new resolver nodes
        # (we place them in this order: given internal nodes, new resolver nodes, leaf nodes from U)
        num_new_nodes = 0
        for r in resolver_sites:
            num_new_nodes += len(r)
       # print("num_new_nodes", num_new_nodes)
        num_internal_nodes = T.shape[0]-num_leaves
        T = torch.nn.functional.pad(T, pad=(0, num_new_nodes, 0, num_new_nodes), mode='constant', value=0)
        # 3. Shift T and G to make room for the new indices (so the order is input internal nodes, new poly nodes, leaves)
        idx1 = num_internal_nodes
        idx2 = num_internal_nodes+num_leaves
        T = torch.cat((T[:,:idx1], T[:,idx2:], T[:,idx1:idx2]), dim=1)
        if G != None:
            G = torch.nn.functional.pad(G, pad=(0, num_new_nodes, 0, num_new_nodes), mode='constant', value=0)
            G = torch.cat((G[:,:idx1], G[:,idx2:], G[:,idx1:idx2]), dim=1)

        # 3. Get each polytomy's children (these are the positions we have to relearn)
        children_of_polys = vutil.get_child_indices(T, nodes_w_polys)
        #print("children_of_polys", children_of_polys)

        # 4. Initialize a matrix to learn the polytomy structure
        num_nodes_full_tree = T.shape[0]
        poly_adj_matrix = vutil.repeat_n(torch.zeros((num_nodes_full_tree, len(children_of_polys)), dtype=torch.float32), bs)
        resolver_indices = [x for x in range(num_internal_nodes, num_internal_nodes+num_new_nodes)]
        #print("resolver_indices", resolver_indices)

        nodes_w_polys_to_resolver_indices = OrderedDict()
        start_new_node_idx = resolver_indices[0]
        for parent_idx, r in zip(nodes_w_polys, resolver_sites):
            num_new_nodes_for_poly = len(r)
            if parent_idx not in nodes_w_polys_to_resolver_indices:
                nodes_w_polys_to_resolver_indices[parent_idx] = []

            for i in range(start_new_node_idx, start_new_node_idx+num_new_nodes_for_poly):
                nodes_w_polys_to_resolver_indices[parent_idx].append(i)
            start_new_node_idx += num_new_nodes_for_poly
        #print("nodes_w_polys_to_resolver_indices", nodes_w_polys_to_resolver_indices)

        resolver_labeling = torch.zeros(U.shape[0], len(resolver_indices))
        t = 0
        for sites in resolver_sites:
            for site in sites:
                resolver_labeling[site, t] = 1
                t += 1
        #print("resolver_labeling", resolver_labeling)

        # print("resolver_indices", resolver_indices)
        offset = 0
        for parent_idx in nodes_w_polys:
            child_indices = vutil.get_child_indices(T, [parent_idx])
            # make the children of polytomies start out as children of their og parent
            # with the option to "switch" to being the child of the new poly node
            poly_adj_matrix[:,parent_idx,offset:(offset+len(child_indices))] = 1.0
            # we only want to let these children choose between being the child
            # of their original parent or the child of this new poly node, which
            # we can do by setting all other indices to -inf
            mask = torch.ones(num_nodes_full_tree, dtype=torch.bool)
            new_nodes = nodes_w_polys_to_resolver_indices[parent_idx]
            mask_indices = new_nodes + [parent_idx]
            #print("parent_idx", parent_idx, "mask_indices", mask_indices)
            mask[[mask_indices]] = 0
            poly_adj_matrix[:,mask,offset:(offset+len(child_indices))] = float("-inf")
            offset += len(child_indices)

        poly_adj_matrix.requires_grad = True
        
        # 5. Initialize potential new nodes as children of the polytomy nodes
        for i in nodes_w_polys:
            for j in nodes_w_polys_to_resolver_indices[i]:
                T[i,j] = 1.0
                node_idx_to_label[j] = f"{i}pol{j}"
                if G != None:
                    G[i,j] = G_IDENTICAL_CLONE_VALUE

        # 6. The genetic distance between a new node and its potential
        # new children which "switch" is the same distance between the new
        # node's parent and the child
        resolver_index_to_parent_idx = {}
        for poly_node in nodes_w_polys_to_resolver_indices:
            new_nodes = nodes_w_polys_to_resolver_indices[poly_node]
            for new_node_idx in new_nodes:
                resolver_index_to_parent_idx[new_node_idx] = poly_node
        #print("resolver_index_to_parent_idx", resolver_index_to_parent_idx)

        if G != None:
            for new_node_idx in resolver_indices:
                parent_idx = resolver_index_to_parent_idx[new_node_idx]
                potential_child_indices = vutil.get_child_indices(T, [parent_idx])
                for child_idx in potential_child_indices:
                    G[new_node_idx, child_idx] = G[parent_idx, child_idx]

        self.latent_var = poly_adj_matrix
        self.nodes_w_polys = nodes_w_polys
        self.children_of_polys = children_of_polys
        self.resolver_indices = resolver_indices
        self.T = T
        self.G = G
        self.node_idx_to_label = node_idx_to_label
        self.resolver_index_to_parent_idx = resolver_index_to_parent_idx
        self.resolver_labeling = resolver_labeling

def is_same_mig_hist_with_node_removed(T, V, remove_idx, poly_res, p):
    '''
    Returns True if migration #, comigration # and seeding # are
    the same after removing node at index remove_idx
    '''
    prev_m, prev_c, prev_s, _, _ = get_ancestral_labeling_metrics(add_batch_dim(V), T, None, None, p)
    # Attach all the children of the candidate removal node to
    # its parent, and then check if that changes the migration history or not

    candidate_T = T.clone().detach()
    candidate_V = V.clone().detach()
    parent_idx = np.where(T[:,remove_idx] > 0)[0][0]
    child_indices = vutil.get_child_indices(T, [remove_idx])
    for child_idx in child_indices:
        candidate_T[parent_idx,child_idx] = 1.0
    candidate_T = np.delete(candidate_T, remove_idx, 0)
    candidate_T = np.delete(candidate_T, remove_idx, 1)
    candidate_V = np.delete(candidate_V, remove_idx, 1)
    new_m, new_c, new_s, _, _ = get_ancestral_labeling_metrics(add_batch_dim(candidate_V), candidate_T, None, None, p)
    
    return ((prev_m == new_m) and (prev_c == new_c) and (prev_s == new_s))

def remove_nodes(removal_indices, V, T, G, node_idx_to_label):
    '''
    '''
    # TODO: test if we need this, pytorch acting weird about no grad mode
    T = T.clone().detach()
    V = V.clone().detach()
    
    # Attach children of the node to remove to their original parent
    for remove_idx in removal_indices:
        parent_idx = np.where(T[:,remove_idx] > 0)[0][0]
        child_indices = vutil.get_child_indices(T, [remove_idx])
        for child_idx in child_indices:
            T[parent_idx,child_idx] = 1.0
    # Remove indices from T, V, soft V and G
    T = np.delete(T, removal_indices, 0)
    T = np.delete(T, removal_indices, 1)
    V = np.delete(V, removal_indices, 1)
    if G != None: 
        G = G.clone().detach()
        G = np.delete(G, removal_indices, 0)
        G = np.delete(G, removal_indices, 1)

    # Reindex the idx to label dict
    copy_node_idx_to_label = copy.deepcopy(node_idx_to_label)
    for idx in removal_indices:
        del copy_node_idx_to_label[idx]
    new_node_idx_to_label = dict()
    for i,key in enumerate(sorted(list(copy_node_idx_to_label.keys()))):
        new_node_idx_to_label[i] = copy_node_idx_to_label[key]
    return V, T, G, new_node_idx_to_label

def remove_extra_resolver_nodes(best_Vs, Ts, node_idx_to_label, G, poly_res, p):
    '''
    If there are any resolver nodes that were added to resolve polytomies but they 
    weren't used (i.e. 1. they have no children or 2. they don't change the 
    migration history), remove them
    '''

    if poly_res == None:
        return best_Vs, Ts, [G for _ in range(len(best_Vs))], [node_idx_to_label for _ in range(len(best_Vs))]

    out_Vs, out_Ts, out_Gs, out_node_idx_to_labels = [],[],[],[]
    for V, T in zip(best_Vs, Ts):
        nodes_to_remove = []
        for new_node_idx in poly_res.resolver_indices:
            children_of_new_node = vutil.get_child_indices(T, [new_node_idx])
            if len(children_of_new_node) == 0:
                nodes_to_remove.append(new_node_idx)
            elif is_same_mig_hist_with_node_removed(T, V, new_node_idx, poly_res, p):
                nodes_to_remove.append(new_node_idx)

        # Genetic distance matrix must get dynamically updated based on the resolved polytomies
        if G != None:
            resolved_G = get_full_G(add_batch_dim(T), G, poly_res)[0]
        else:
            resolved_G = None
        new_V, new_T, new_G, new_node_idx_to_label = remove_nodes(nodes_to_remove, V, T, resolved_G, node_idx_to_label)
        out_Vs.append(new_V)
        out_Ts.append(new_T)
        out_Gs.append(new_G)
        out_node_idx_to_labels.append(new_node_idx_to_label)
    return out_Vs, out_Ts, out_Gs, out_node_idx_to_labels


# def is_less_parsimonious(pars_metrics, best_pars_metrics):
    
#     for best_pars_metric in best_pars_metrics:
#         if not (pars_metrics[0] <= best_pars_metric[0] and pars_metrics[1] <= best_pars_metric[1] and pars_metrics[2] <= best_pars_metric[2]):
#             return True
#     return False

def made_mistake(solution, U):
    V = solution.V
    A = solution.T
    VA = V @ A
    W = VA.T @ VA # 1 if two nodes' parents are the same color
    X = V.T @ V # 1 if two nodes are the same color
    Y = torch.sum(torch.mul(VA.T, 1-V.T), axis=1) # Y has a 1 for every node where its parent has a diff color
    nonzero_indices = torch.nonzero(Y).squeeze()
    U = U[:,1:] # don't include column for normal cells
    num_sites = U.shape[0]
    L = torch.nn.functional.one_hot((U > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
    num_internal_nodes = A.shape[0] - L.shape[1]
    if nonzero_indices.dim() == 0:
        return False
    # print(nonzero_indices)
    for mig_node in nonzero_indices:
        # it's a leaf node itself!
        if mig_node > (num_internal_nodes-1):
            continue
        if not vutil.has_leaf_node(A, int(mig_node), num_internal_nodes):
            return True
    return False
    

def prune_bad_histories(solutions, O, p, weights, mode, U):
    '''
    Detect mistakes and prune out those histories (this is specifically)
    in datasets where there are branches of the tree with no U leaf nodes
    Removes any solutions that are > parsimony_eps parsimony metrics
    away from the best solution
    '''
    parsimony_eps = 1

    def keep_tree(cand_metric, best_sum, best_pars_metrics):
        if abs(best_sum - sum(cand_metric)) > parsimony_eps:
            return False
        if sum(cand_metric) == best_sum:
            return True
        # All these trees are +/- parsimony_eps, and they need to 
        # improve at least one of the metrics compared to the best_pars_metrics
        for best_metric in best_pars_metrics:
            if cand_metric[0] < best_metric[0] or cand_metric[1] < best_metric[1] or cand_metric[2] < best_metric[2]:
                return True
        return False



    # Collect each solution's parsimony metrics
    all_pars_metrics = []
    for soln in solutions:
        V, T, G = soln.V, soln.T, soln.G
        m, c, s, _, _ = get_ancestral_labeling_metrics(add_batch_dim(V), T, G, O, p)
        all_pars_metrics.append((int(m), int(c), int(s)))

    # Find the best parsimony sum
    best_sum = float("inf")
    for i, pars_metrics in enumerate(all_pars_metrics):
        pars_sum = sum(pars_metrics)
        if pars_sum < best_sum:
            best_sum = pars_sum

    # Find all pars metrics combinations that match the best sum
    best_pars_metrics = set()
    for i, pars_metrics in enumerate(all_pars_metrics):
        pars_sum = sum(pars_metrics)
        if pars_sum == best_sum:
            best_pars_metrics.add(pars_metrics)

    pruned_solutions = []
    
    # Go through and prune any solutions that are worse than the best sum
    # or made any mistakes
    for soln, pars_metrics in zip(solutions, all_pars_metrics):
        if keep_tree(pars_metrics, best_sum, best_pars_metrics) and not made_mistake(soln, U):
            pruned_solutions.append(soln)

    if len(pruned_solutions) == 0: 
        print("no solutions without mistakes detected")
        # ideally this doesn't happen, but remove mistake detection so 
        # that we return some results
        for soln, pars_metrics in zip(solutions, all_pars_metrics):
            if keep_tree(pars_metrics, best_sum, best_pars_metrics):
                pruned_solutions.append(soln)

    return pruned_solutions

def _get_best_final_solutions(mode, best_Vs, best_soft_Vs, Ts, Gs, O, p, weights, print_config, 
                              poly_res, idx_label_dicts, output_dir, run_name, U, needs_pruning=True):
    '''
    Weigh solutions differently based on evaluate or calibrate mode. 
    Return the top k solutions
    '''

    # 1. Initialize weights for loss calculations based on what mode we're in
    input_weights = weights
    if mode == "evaluate":
        weights = weights
    elif mode == 'calibrate': 
        # Evaluate most parsimonious trees via gen. dist. and organotropism
        weights = met.Weights(mig=[0.0], comig=0.0, seed_site=[0.0], gen_dist=weights.gen_dist, organotrop=weights.organotrop, data_fit=0.0, reg=0.0, entropy=0.0)

    # 2. Calculate loss for each solution
    bs = len(best_Vs)
    mig_weights = get_mig_weight_vector(bs, input_weights)
    seed_weights = get_seed_site_weight_vector(bs, input_weights)
    solutions = []
    for i, (V, soft_V, T, G, idx_label_dict) in enumerate(zip(best_Vs, best_soft_Vs, Ts, Gs, idx_label_dicts)):
        reshaped_V = add_batch_dim(V)
        reshaped_soft_V = add_batch_dim(soft_V)        
        loss, loss_components = ancestral_labeling_objective(reshaped_V, reshaped_soft_V, T, G, O, p, weights)
        # TODO: take mig_delta as input?
        solutions.append(VertexLabelingSolution(loss, V, soft_V, T, G, idx_label_dict, mig_weights[i], input_weights.comig, seed_weights[i], i))
    
    # 3. Prune bad histories (anything less parsimonious than the best solution we find)
    if needs_pruning:
        pruned_solutions = prune_bad_histories(solutions, O, p, input_weights, mode, U)
    else:
        pruned_solutions = solutions

    # 4. Make the solutions unique
    unique_solutions = set()
    final_solutions = []
    seeding_outputs = [["Evaluation loss", "Seeding pattern"]]
    for soln in pruned_solutions:
        tree = vutil.LabeledTree(soln.T, soln.V)
        if tree not in unique_solutions:
            final_solutions.append(soln)
            unique_solutions.add(tree)
        seeding_outputs.append([float(soln.loss), plot_util.get_seeding_pattern(soln.V, soln.T)])

    # 5. Sort the solutions from lowest to highest loss
    final_solutions = sorted(list(final_solutions))

    # 6. Return the k best solutions
    k = print_config.k_best_trees if len(final_solutions) >= print_config.k_best_trees else len(final_solutions)
    final_solutions = final_solutions[:k]
    return final_solutions, seeding_outputs

def get_best_final_solutions(mode, best_Vs, best_soft_Vs, Ts, G, O, p, weights, print_config, 
                             poly_res, node_idx_to_label, output_dir, run_name, U):
    '''
    Prune unecessary poly nodes (if they weren't used), then weigh solutions  
    differently based on evaluate or calibrate mode. 
    Return the top k solutions
    '''
    # 1. Remove any extra resolver nodes that don't actually help
    best_Vs, Ts, Gs, idx_label_dicts = remove_extra_resolver_nodes(best_Vs, Ts, node_idx_to_label, G, poly_res, p)

    return _get_best_final_solutions(mode, best_Vs, best_soft_Vs, Ts, Gs, O, p, weights, print_config, 
                                     poly_res, idx_label_dicts, output_dir, run_name, U)

def init_X(batch_size, num_sites, num_nodes_to_label, weight_init_primary, p, input_T, T, U):
    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(batch_size, num_sites, num_nodes_to_label)
    #print("X", X.shape)
    eta = 2 # TODO: how do we set this parameter?
    if weight_init_primary:
        # for each node, find which sites to bias labeling towards using
        # the sites it and its children are detected in
        nodes_w_children, sites = vutil.get_k_or_more_children_nodes(input_T, T, U, 1, 1, cutoff=False)
        #print("nodes_w_children", nodes_w_children, "sites", sites)
        prim_site_idx = torch.nonzero(p)[0][0]
        #print("p", p, "prim_site_idx", prim_site_idx)
        X[:,prim_site_idx,:] = eta

        for node_idx, sites in zip(nodes_w_children, sites):
            if node_idx == 0:
                continue # we know the root labeling
            idx = node_idx - 1
            for site_idx in sites:
                X[:,site_idx,idx] = eta
        #print("X", X[0])
    X.requires_grad = True
    return X

def update_adj_matrix(itr, max_iter):
    if itr == -1:
        return True
    global SOLVE_POLYS
    if not SOLVE_POLYS:
        return False
    return itr > max_iter*1/3 and itr < max_iter*2/3
    # return itr > max_iter*1/2

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

# TODO: should we not take weights as input for this?
def calibrate(Ts, ref_matrices, var_matrices, ordered_sites, primary_sites, 
              node_idx_to_labels, weights, print_config, output_dir, run_names,
              Gs=None, Os=None, max_iter=100, lr=0.1, init_temp=40, final_temp=0.01,
              batch_size=64, custom_colors=None, weight_init_primary=True,  solve_polytomies=False):
    if not (len(Ts) == len(ref_matrices) == len(var_matrices) == len(ordered_sites) == len(primary_sites) == len(node_idx_to_labels) == len(run_names)):
        raise ValueError("Inputs ref_matrices, var_matrices, ordered_sites, primary_sites, node_idx_to_labels and run_names must have equal length (length = to number of patients in cohort")
    if (Gs == None and Os == None):
        raise ValueError("In calibrate mode, either Gs or Os should be input")
    if (Gs != None and len(Gs) != len(Ts)):
        raise ValueError("Length of Ts and Gs must be equal")
    if (Os != None and len(Os) != len(Ts)):
        raise ValueError("Length of Ts and Os must be equal")


    def _convert_list_of_numpys_to_tensors(lst):
        return [torch.tensor(x) for x in lst]

    input_weights = copy.deepcopy(weights)
    organotrop_weight = 0.0 if Os == None else (weights.organotrop if weights != None else 0.1)
    gen_dist_weight = 0.0 if Gs == None else (weights.gen_dist if weights != None else 0.1)

    weights = met.Weights(mig=DEFAULT_CALIBRATE_MIG_WEIGHTS, comig=DEFAULT_CALIBRATE_COMIG_WEIGHTS, 
                          seed_site=DEFAULT_CALIBRATE_SEED_WEIGHTS, gen_dist=gen_dist_weight, 
                          organotrop=organotrop_weight)
    # Don't spend time making visualizations for calibrated trees
    visualize = print_config.visualize
    input_k = print_config.k_best_trees
    print_config.visualize = False
    print_config.k_best_trees = batch_size

    calibrate_dir = os.path.join(output_dir, "calibrate")

    print(f"Saving results to {calibrate_dir}")

    if os.path.exists(calibrate_dir):
        shutil.rmtree(calibrate_dir)
        print(f"Overwriting existing directory at {calibrate_dir}")
    
    os.makedirs(calibrate_dir)

    # 1. Go through each patient and get migration history in calibrate mode
    for i in range(len(Ts)):
        print("Calibrating for patient:", run_names[i])
        G = Gs[i] if Gs != None else None
        O = Os[i] if Os != None else None
        get_migration_history(Ts[i], ref_matrices[i], var_matrices[i], ordered_sites[i], primary_sites[i],
                              node_idx_to_labels[i], weights, print_config, calibrate_dir, f"{run_names[i]}", 
                              G=G, O=O, max_iter=max_iter, lr=lr, init_temp=init_temp, final_temp=final_temp,
                              batch_size=batch_size, custom_colors=custom_colors, weight_init_primary=weight_init_primary, 
                              mode="calibrate", solve_polytomies=solve_polytomies)

    # 2. Find the best theta for this cohort
    best_theta = eutil.get_max_cross_ent_thetas(pickle_file_dirs=[calibrate_dir])
    rounded_best_theta = [round(v,3) for v in best_theta]
    with open(os.path.join(calibrate_dir, "best_theta.json"), 'w') as json_file:
        json.dump(rounded_best_theta, json_file, indent=2)
        
    # 3. Recalibrate trees using the best thetas
    print_config.visualize = visualize
    print_config.k_best_trees = input_k
    weights = met.Weights(mig=[best_theta[0]], comig=best_theta[1], seed_site=[best_theta[2]],
                          gen_dist=input_weights.gen_dist if Gs != None else 0.0, organotrop=input_weights.organotrop if Os != None else 0.0)
    
    for i in range(len(Ts)):
        O = Os[i] if Os != None else None
        with gzip.open(os.path.join(calibrate_dir, f"{run_names[i]}.pkl.gz"), 'rb') as f:
            pckl = pickle.load(f)
        saved_Ts = _convert_list_of_numpys_to_tensors(pckl[OUT_ADJ_KEY])
        saved_Vs = _convert_list_of_numpys_to_tensors(pckl[OUT_LABElING_KEY])
        saved_soft_Vs = _convert_list_of_numpys_to_tensors(pckl[OUT_SOFTV_KEY])
        saved_U = torch.tensor(pckl[OUT_SUB_PRES_KEY])
        saved_Gs = _convert_list_of_numpys_to_tensors(pckl[OUT_GEN_DIST_KEY])
        saved_idx_to_label_dicts = remove_leaf_nodes_idx_to_label_dicts(pckl[OUT_IDX_LABEL_KEY])
        primary_idx = ordered_sites[i].index(primary_sites[i])
        p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites[i])).T
        
        final_solutions, _ = _get_best_final_solutions('evaluate', saved_Vs, saved_soft_Vs, saved_Ts, saved_Gs, O, p, weights,
                                                       print_config, None, saved_idx_to_label_dicts, calibrate_dir, run_names[i], 
                                                       saved_U, needs_pruning=False)
        
        plot_util.print_best_trees(final_solutions, saved_U, ref_matrices[i], var_matrices[i], O,
                                  weights, ordered_sites[i], print_config, 
                                  custom_colors, primary_sites[i], 
                                  calibrate_dir, run_names[i])

    return best_theta


def get_migration_history(T, ref, var, ordered_sites, primary_site, node_idx_to_label,
                          weights, print_config, output_dir, run_name, 
                          G=None, O=None, max_iter=100, lr=0.1, init_temp=40, final_temp=0.01,
                          batch_size=64, custom_colors=None, weight_init_primary=True, mode="evaluate",
                          solve_polytomies=False):
    '''
    Args:
        T: numpy ndarray or torch tensor (shape: num_internal_nodes x num_internal_nodes). Adjacency matrix (directed) of the internal nodes.
        
        ref: numpy ndarray or torch tensor (shape: num_anatomical_sites x num_mutation_clusters). Reference matrix, i.e., num. reads that map to reference allele
        
        var: numpy ndarray or torch tensor (shape:  num_anatomical_sites x num_mutation_clusters). Variant matrix, i.e., num. reads that map to variant allele
        
        ordered_sites: list of the anatomical site names (e.g. ["breast", "lung_met"]) with length =  num_anatomical_sites) and 
        the order matches the order of sites in the ref and var
        
        primary_site: name of the primary site (must be an element of ordered_sites)

        node_idx_to_label: dictionary mapping vertex indices (corresponding to their index in T) to custom labels
        for plotting

        weights: Weight object for how much to penalize each component of the loss
        
        print_config: PrintConfig object with options on how to visualize output
        
        output_dir: path for where to save output trees to

        run_name: e.g. patient name, used for naming output files.

    Optional:
        G: numpy ndarray or torch tensor (shape: num_internal_nodes x num_internal_nodes).
        Matrix of genetic distances between internal nodes.
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        
        O: numpy ndarray or torch tensor (shape: num_anatomical_sites x  num_anatomical_sites).
        Matrix of organotropism values between sites.

        weight_init_primary: whether to initialize weights higher to favor vertex labeling of primary for all internal nodes

        mode: can be "evaluate" or "calibrate"

    Returns:
        # TODO: return info for k best trees
        Corresponding info on the *best* tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
        (4) dictionary w/ loss values for each component of the loss
        (5) how long (in seconds) the algorithm took to run
    '''
    if not (T.shape[0] == T.shape[1]):
        raise ValueError(f"Number of tree nodes should be consistent (T.shape[0] == T.shape[1])")
    if (T.shape[0] != len(node_idx_to_label)):
        raise ValueError(f"Number of node_idx_to_label needs to equal shape of adjacency matrix.")
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
    if (weights.gen_dist == 0.0 and G != None):
        raise ValueError(f"G matrix was given but genetic distance parameter of weights is 0. Please set genetic distance weight to > 0.")
    if (weights.gen_dist != 0.0 and G == None):
        raise ValueError(f"G matrix was not given but genetic distance parameter of weights is non-zero. Please pass a G matrix.")
    if (weights.organotrop == 0.0 and O != None):
        raise ValueError(f"O matrix was given but organotropism parameter of weights is 0. Please set organotropism weight to > 0.")
    if (weights.organotrop != 0.0 and O == None):
        raise ValueError(f"O matrix was not given but organotropism parameter of weights is non-zero. Please pass an O matrix.")
    if mode != 'calibrate' and mode != 'evaluate':
        raise ValueError(f"Valid modes are 'evaluate' and 'calibrate'")
    if mode == 'calibrate':
        if not ((weights.organotrop != 0.0 and O != None) or (weights.gen_dist != 0.0 and G != None)):
            raise ValueError(f"In calibrate mode, either organotropism or genetic distance matrices and weights must be set.")
    for label in list(node_idx_to_label.values()):
        if ":" in label:
            raise ValueError(f"Unfortunately our visualization code uses pydot, which does not allow colons (:) in node names. Please use a different separator in the values of node_idx_to_label")

    # Don't alter the inputs in place
    T = copy.deepcopy(T)
    G = copy.deepcopy(G)
    ref = copy.deepcopy(ref)
    var = copy.deepcopy(var)
    ordered_sites = copy.deepcopy(ordered_sites)
    primary_site = copy.deepcopy(primary_site)
    node_idx_to_label = copy.deepcopy(node_idx_to_label)

    root_idx = vutil.get_root_index(T)
    if root_idx != 0:
        print(f"Restructuring adjacency matrix for {run_name} since root node is not at index 0")
        T, ref, var, node_idx_to_label, G = vutil.restructure_matrices(T, ref, var, node_idx_to_label, G)
    assert(vutil.get_root_index(T) == 0)

    # TODO: convert to torch float 32 if its not already
    if not torch.is_tensor(T):
        T = torch.tensor(T, dtype=torch.float32)
    if not torch.is_tensor(ref):
        ref = torch.tensor(ref, dtype=torch.float32)
    if not torch.is_tensor(var):
        var = torch.tensor(var, dtype=torch.float32)

    primary_idx = ordered_sites.index(primary_site)
    p = torch.nn.functional.one_hot(torch.tensor([primary_idx]), num_classes=len(ordered_sites)).T
    num_sites = ref.shape[0]
    num_internal_nodes = T.shape[0]

    # If we don't know the anatomical site of the primary tumor, we need to learn it
    num_nodes_to_label = -1
   
    assert(p.shape[1] == 1)
    assert(p.shape[0] == ref.shape[0]) # num_anatomical_sites
    num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling
    prim_site_idx = torch.nonzero(p)[0][0]
    primary_site_label = ordered_sites[prim_site_idx]

    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and anatomical locations of the extant clones (U > U_CUTOFF)
    psi = -1 * torch.rand(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi.requires_grad = True 
    u_optimizer = torch.optim.Adam([psi], lr=lr)

    B = vutil.get_mutation_matrix_tensor(T)
    # add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # add a column of ones to indicate that every subclone has the non-cancerous mutations
    B = torch.hstack([torch.ones(B.shape[0]).reshape(-1,1), B])

    # reinitialize globals as None (if module is not reloaded in between runs, this can cause problems)
    global RANDOM_VALS, LAST_P, G_IDENTICAL_CLONE_VALUE, SOLVE_POLYS
    RANDOM_VALS = None
    LAST_P = None
    SOLVE_POLYS = solve_polytomies

    if G != None:
        G_IDENTICAL_CLONE_VALUE = torch.min(G[(G != 0)])/2.0

    start_time = datetime.datetime.now()

    input_T = copy.deepcopy(T)
    i = 0
    u_prev = psi
    u_diff = 1e9
    ############ Step 1, find a MAP estimate of U ############
    while u_diff > 1e-4 and i < 200:
        u_optimizer.zero_grad()
        U, u_loss = compute_u_loss(psi, ref, var, B, weights)
        u_loss.backward()
        u_optimizer.step()
        u_diff = torch.abs(torch.norm(u_prev - U))
        u_prev = U
        i += 1
    
    T, G, num_leaves = full_adj_matrix(U, T, G)
    if solve_polytomies:
        nodes_w_polys, resolver_sites = vutil.get_k_or_more_children_nodes(input_T, T, U, 3, 2)
        if len(nodes_w_polys) == 0:
            print("No potential polytomies to solve, not resolving polytomies.")
            poly_res = None
            solve_polytomies = False
        else:
            poly_res = PolytomyResolver(input_T, T, G, U, num_leaves, batch_size, node_idx_to_label, nodes_w_polys, resolver_sites)
            T = poly_res.T
            print("T", T.shape)
            G = poly_res.G
            node_idx_to_label = poly_res.node_idx_to_label
            #num_nodes_to_label += len(poly_res.resolver_indices)

            poly_optimizer = torch.optim.Adam([poly_res.latent_var], lr=lr)
            print("num_nodes_to_label", num_nodes_to_label)
    else:
        poly_res = None

    # We're learning X, which is the vertex labeling of the internal nodes
    X = init_X(batch_size, num_sites, num_nodes_to_label, weight_init_primary, p, input_T, T, U)
    
    v_optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = lr_scheduler.LinearLR(v_optimizer, start_factor=1.0, end_factor=0.5, total_iters=max_iter)

    ############ Step 2, sample from V to estimate q(V) ############

    # Temperature and annealing
    v_temp = init_temp
    t_temp = init_temp
    hard = True
    v_anneal_rate = 0.002
    t_anneal_rate = 0.1
    intermediate_data = []
    temps = []
    v_loss_components = []

    # Calculate V only using maximum parsimony metrics
    if mode == 'calibrate':
        # Make a copy of inputs for later
        weights_copy = copy.deepcopy(weights)
        G_copy = copy.deepcopy(G)
        O_copy = copy.deepcopy(O)
        weights = met.Weights(mig=weights.mig, comig=weights.comig, seed_site=weights.seed_site, gen_dist=0.0, organotrop=0.0, data_fit=weights.data_fit, reg=weights.reg, entropy=weights.entropy)
        G, O = None, None
    
    # Keep track of the best trees and losses
    best_Vs, best_soft_Vs, best_Ts, = None, None, None
    prev_v_losses = None
    j = 0
    k = 0
    for i in tqdm(range(max_iter)):

        if solve_polytomies: 
            poly_optimizer.zero_grad()
        v_optimizer.zero_grad()
        Vs, v_losses, loss_comps, soft_Vs, Ts = compute_v_loss(U, X, T, ref, var, p, G, O, v_temp, t_temp, hard, weights, i, max_iter, poly_res)
        mean_loss = torch.sum(v_losses)
        mean_loss.backward()

        if solve_polytomies and update_adj_matrix(i, max_iter):
            poly_optimizer.step()
            if i % 2 == 0:
                t_temp = np.maximum(t_temp * np.exp(-t_anneal_rate * j), final_temp)
            j += 1

        else:
            v_optimizer.step()
            scheduler.step()
            if i % 5 == 0:
                v_temp = np.maximum(v_temp * np.exp(-v_anneal_rate * k), final_temp)
            k += 1

        if print_config.visualize:
            v_loss_components.append(get_avg_loss_components(loss_comps))

        with torch.no_grad():
            best_Vs, best_soft_Vs, best_Ts = get_best_solutions(prev_v_losses, v_losses, Vs, soft_Vs, Ts, best_Vs, best_soft_Vs, best_Ts)
            
        temps.append(v_temp)

        # print("i", i, "v temp", v_temp, "t_temp", t_temp)

        prev_v_losses = v_losses


    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")

    with torch.no_grad():
        if mode == "calibrate":
            # Reload
            G, O = G_copy, O_copy
            weights = weights_copy
        
        final_solutions, seeding_outputs = get_best_final_solutions(mode, best_Vs, best_soft_Vs, best_Ts, G, O, p, weights,
                                                                    print_config, poly_res, node_idx_to_label, output_dir, 
                                                                    run_name, U)

        print("# final solutions:", len(final_solutions))

        if mode == 'calibrate':
            with open(os.path.join(output_dir, f"{run_name}_eval_loss_by_seeding_pattern.txt"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(seeding_outputs)


        if print_config.visualize:
            plot_util.plot_loss_components(v_loss_components, weights)
            #plot_util.plot_temps(temps)

        edges, vert_to_site_map, mig_graph_edges, loss_info = plot_util.print_best_trees(final_solutions, U, ref, var, O, 
                                                                                         weights, ordered_sites,print_config, 
                                                                                         custom_colors, primary_site_label, 
                                                                                         output_dir, run_name)

        # avg_tree = plot_util.print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors,
        #                                     ordered_sites, print_config)

       

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed

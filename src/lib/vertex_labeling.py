
import torch
import sys
import numpy as np
import datetime
from tqdm import tqdm

import torchopt

from src.util import vertex_labeling_util as vutil
from src.util import plotting_util as plot_util

from torch.distributions.binomial import Binomial
from src.util.globals import *

from pprint import pprint
import os

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def repeat_n(x, n):
    '''
    Repeats tensor x 'n' times along the first axis, returning a tensor
    w/ dim (n, x.shape[0], x.shape[1])
    '''
    return x.repeat(n,1).reshape(n, x.shape[0], x.shape[1])

def get_parsimony_metrics(V, A):

    single_A = A
    bs = V.shape[0]
    num_sites = V.shape[1]
    num_nodes = V.shape[2]
    A = repeat_n(single_A, bs)

    # 1. Migration number
    VA = V @ A
    VT = torch.transpose(V, 2, 1)
    site_adj = VA @ VT
    site_adj_trace = torch.diagonal(site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    m = torch.sum(site_adj, dim=(1, 2)) - site_adj_trace

    # 2. Seeding site number
    # remove the same site transitions from the site adj matrix
    site_adj_no_diag = torch.mul(site_adj, repeat_n(1-torch.eye(num_sites, num_sites), bs))
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=2)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    alpha = 100.0
    binarized_row_sums_site_adj = torch.sigmoid(alpha * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj, dim=(1))

    # # 3. Comigration number
    VAT = torch.transpose(VA, 2, 1)
    W = VAT @ VA # 1 if two nodes' parents are the same color
    X = VT @ V # 1 if two nodes are the same color
    Y = torch.sum(torch.mul(VAT, 1-VT), axis=2) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X) # 1 if two nodes' parents are same color AND nodes are same color
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    P = repeat_n(vutil.get_path_matrix_tensor(single_A.cpu().numpy()), bs)
    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=2)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y), axis=1)
    binarized_site_adj = torch.sigmoid(alpha * (2 * site_adj - 1))
    bin_site_trace = torch.diagonal(binarized_site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    c = torch.sum(binarized_site_adj, dim=(1,2)) - bin_site_trace + repeated_temporal_migrations

    return m, c, s, X, site_adj_no_diag

def _ancestral_labeling_objective(V, soft_V, A, G, O, p, weights):
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
    
    assert(A.shape[0]==A.shape[1]==V.shape[2])

    m, c, s, X, site_adj_no_diag = get_parsimony_metrics(V, A)

    # Entropy
    entropy = -torch.sum(torch.mul(soft_V, torch.log2(soft_V)), dim=(1, 2)) / V.shape[2]

    # 4. Genetic distance
    g = 0
    if G != None and weights.gen_dist != 0:
        # calculate if 2 nodes are in diff sites and there's an edge between them (i.e there is a migration edge)
        G = repeat_n(G, bs)
        R = torch.mul(repeat_n(A, bs), (1-X))
        adjusted_G = torch.exp(GENETIC_ALPHA*G)
        R = torch.mul(R, adjusted_G)
        g = torch.sum(R, dim=(1,2))

    # 5. Organotropism
    o = 0
    if O != None and weights.organotrop != 0:
        # the organotropism frequencies can only be used on the first 
        # row, which is for the migrations from primary cancer site to
        # other metastatic sites (we don't have frequencies for every 
        # site to site migration)
        prim_site_idx = torch.nonzero(p)[0][0]
        O = O.repeat(bs,1).reshape(bs, O.shape[0])
        adjusted_freqs = torch.exp(ORGANOTROP_ALPHA*O)
        organ_penalty = torch.mul(site_adj_no_diag[:,prim_site_idx,:], adjusted_freqs)
        o = torch.sum(organ_penalty, dim=(1))
    # Combine all 5 components with their weights
    vertex_labeling_loss = (weights.mig*m + weights.seed_site*s + weights.comig*c + weights.gen_dist*g + weights.organotrop*o+ entropy)
    loss_components = {MIG_KEY: m, COMIG_KEY: c, SEEDING_KEY: s, ORGANOTROP_KEY: o, GEN_DIST_KEY: g, ENTROPY_KEY: entropy}

    return vertex_labeling_loss, loss_components

# Adapted from PairTree
def _calc_llh(F_hat, R, V, omega_v, epsilon=1e-5):
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

def _subclonal_presence_objective(ref, var, omega_v, U, B, weights):
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
    F_llh, llh_per_sample, nlglh = _calc_llh(F_hat, ref, var, omega_v)

    # 2. Regularization to make some values of U -> 0
    # TODO: this is not a normal matrix norm, but works very well here...
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

def evaluate(V, soft_V, A, ref, var, U, B, G, O, p, weights):
    '''
    Args:
        V: Vertex labeling of the **full** tree (includes leaf nodes) (num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        ref: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        U: Mixture matrix (num_sites x num_internal_nodes)
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        weights: Weights object
    Returns:
        Loss to score this tree and labeling combo.
    '''

    ancestral_labeling_loss, loss_dict_a = _ancestral_labeling_objective(V, soft_V, A, G, O, p, weights)
    
    omega_v = no_cna_omega(ref.shape)
    subclonal_presence_loss, loss_dict_b = _subclonal_presence_objective(ref, var, omega_v, U, B, weights)
    
    loss = subclonal_presence_loss + ancestral_labeling_loss
    loss_components = {**loss_dict_a, **loss_dict_b, **{FULL_LOSS_KEY: round(torch.mean(loss).item(), 3)}}

    return loss, loss_components

def sample_gumbel(shape, eps=1e-20):
    G = torch.rand(shape)
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

    return full_adj, full_G
    

def compute_u_loss(psi, ref, var, B, weights):
    '''
    Computes loss for latent variable U (dim: num_internal_nodes x num_sites)
    '''
    # Using the softmax enforces that the row sums are 1, since the proprtions of
    # subclones in a given site should sum to 1
    U = torch.softmax(psi, dim=1)
    batch_size = U.shape[0]
    omega_v = no_cna_omega(ref.shape)
    u_loss, loss_dict_b = _subclonal_presence_objective(ref, var, omega_v, U, B, weights)
    return U, u_loss


def compute_v_loss(U, X, T, ref, var, p, G, O, temp, hard, weights):
    '''
    Computes loss for X (dim: batch_size x num_internal_nodes x num_sites)
    '''

    def stack_vertex_labeling(U, X, p, T):
        '''
        Use U and X (both of size batch_size x num_internal_nodes x num_sites)
        to get the anatomical sites of the leaf nodes and the internal nodes (respectively). If the root labeling
        p is known, then stack the root labeling to get the full vertex labeling V. If the root labeling is not known,
        X has an extra column (the first column) used to learn the site of the primary.
        '''
        U = U[:,1:] # don't include column for normal cells
        num_sites = U.shape[0]
        L = torch.nn.functional.one_hot((U > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
        # Expand leaf node labeling L to be repeated batch_size times
        bs = X.shape[0]
        L = repeat_n(L, bs)

        if p is None:
            return torch.cat((X, L), dim=2)
        # p needs to be placed at the index of the root node
        full_vert_labeling = torch.cat((X, L), dim=2)
        # Index where vector p should be inserted
        root_idx = plot_util.get_root_index(T)
        # Split the tensor into two parts at the specified index
        left_part = full_vert_labeling[:,:,:root_idx]
        right_part = full_vert_labeling[:,:,root_idx:]

        p = repeat_n(p, bs)
        # Concatenate the left part, new column, and right part along the second dimension
        return torch.cat((left_part, p, right_part), dim=2)


    softmax_X, softmax_X_soft = gumbel_softmax(X, temp, hard)
    V = stack_vertex_labeling(U, softmax_X, p, T)
    loss, loss_components = _ancestral_labeling_objective(V, softmax_X_soft, T, G, O, p, weights)
    return V, loss, loss_components, softmax_X_soft
    
def get_avg_loss_components(loss_components):
    '''
    Calculate the averages of each loss component (e.g. "nll" 
    (negative log likelihood), "mig" (migration number), etc.)
    '''
    d = {}
    for key in loss_components:
        if isinstance(loss_components[key], int):
            d[key] = loss_components[key]
        else:
            d[key] = torch.mean(loss_components[key])
    return d

def get_migration_history(T, ref, var, ordered_sites, primary_site, node_idx_to_label,
                          weights, print_config, output_dir, run_name, 
                          G=None, O=None, max_iter=200, lr=0.1, init_temp=40, final_temp=0.01,
                          batch_size=64, custom_colors=None, weight_init_primary=True, lr_sched="step"):
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
    if p is None:
        num_nodes_to_label = num_internal_nodes
    else:
        assert(p.shape[1] == 1)
        assert(p.shape[0] == ref.shape[0]) # num_anatomical_sites
        num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling
        prim_site_idx = torch.nonzero(p)[0][0]
        primary_site_label = ordered_sites[prim_site_idx]

    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(batch_size, num_sites, num_nodes_to_label)
    if weight_init_primary:
        if p is None: raise ValueError(f"Cannot use weight_init_primary flag without inputting p vector")
        prim_site_idx = torch.nonzero(p)[0][0]
        X[:batch_size//2,prim_site_idx,:] = 2 # TODO: how do we set this parameter?

    X.requires_grad = True
    v_optimizer = torch.optim.Adam([X], lr=lr)

    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and antomical locations of the extant clones (U > U_CUTOFF)
    psi = -1 * torch.rand(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi.requires_grad = True 
    u_optimizer = torch.optim.Adam([psi], lr=lr)

    B = vutil.get_mutation_matrix_tensor(T)
    # add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # add a column of ones to indicate that every subclone has the non-cancerous mutations
    B = torch.hstack ([torch.ones(B.shape[0]).reshape(-1,1), B])

    start_time = datetime.datetime.now()

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

    T, G = full_adj_matrix(U, T, G)
    ############ Step 2, sample from V to estimate q(V) ############

    # Temperature and annealing
    temp = init_temp
    hard = True
    anneal_rate = 0.002
    intermediate_data = []
    temps = []
    v_loss_components = []

    for i in tqdm(range(max_iter)):

        v_optimizer.zero_grad()
        V, v_loss, loss_comps, soft_V = compute_v_loss(U, X, T, ref, var, p, G, O, temp, hard, weights)
        mean_loss = torch.mean(v_loss)
        mean_loss.backward()
        v_optimizer.step()

        with torch.no_grad():
            if i % 20 == 0:
                intermediate_data.append([v_loss, V, soft_V])

        v_loss_components.append(get_avg_loss_components(loss_comps))

        if i % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * i), final_temp)
        temps.append(temp)

    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")

    with torch.no_grad():
        full_loss, loss_dict = evaluate(V, soft_V, T, ref, var, U, B, G, O, p, weights)

        if print_config.visualize:
            plot_util.plot_loss_components(v_loss_components, weights)

        edges, vert_to_site_map, mig_graph_edges, loss_info = plot_util.print_best_trees(full_loss, V, soft_V, U, ref, var, B, O, G, T, 
                                                                                         weights,node_idx_to_label, ordered_sites,
                                                                                         print_config, intermediate_data, 
                                                                                         custom_colors, primary_site_label, 
                                                                                         output_dir, run_name)

        # avg_tree = plot_util.print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors,
        #                                     ordered_sites, print_config)

       

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed
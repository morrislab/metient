
import torch
import sys
import numpy as np
import datetime
from tqdm import tqdm

import torchopt

from src.util import vertex_labeling_util as vert_util
from src.util import plotting_util as plot_util

from torch.distributions.binomial import Binomial
from src.util.globals import *

from pprint import pprint
import os

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Weights:
    def __init__(self, data_fit=1.0, mig=1.0, comig=1.0, seed_site=1.0, reg=1.0, gen_dist=0.0, organotrop=0.0):
        self.data_fit = data_fit
        self.mig = mig
        self.comig = comig
        self.seed_site = seed_site
        self.reg = reg
        self.gen_dist = gen_dist
        self.organotrop = organotrop

def _ancestral_labeling_objective(V, A, G, O, weights):
    '''
    Args:
        V: Vertex labeling of the full tree (num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        weights: Weights object

    Returns:
        Loss to score the ancestral vertex labeling of the given tree. This combines (1) migration number, (2) seeding site
        number, (3) comigration number, and optionally (4) genetic distance and (5) organotropism.
    '''

    # 1. Migration number
    VA = V @ A
    site_adj = VA @ V.T
    m = torch.sum(site_adj) - torch.trace(site_adj)

    # 2. Seeding site number
    # remove the same site transitions from the site adj matrix
    site_adj_no_diag = torch.mul(site_adj, 1-torch.eye(site_adj.shape[0], site_adj.shape[1]))
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=1)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    alpha = 100.0
    binarized_row_sums_site_adj = torch.sigmoid(alpha * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj)

    # 3. Comigration number
    W = VA.T @ VA # W tells us if two nodes' parents are the same color
    X = V.T @ V # X tells us if two nodes are the same color
    Y = torch.sum(torch.mul(VA.T, 1-V.T), axis=1) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X)
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    # TODO: this is computationally expensive, maybe we could cache path matrices we've calculated before?
    P = vert_util.get_path_matrix_tensor(A.cpu().numpy())
    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=1)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y))
    binarized_site_adj = torch.sigmoid(alpha * (2 * site_adj - 1))
    c = torch.sum(binarized_site_adj) - torch.trace(binarized_site_adj) + repeated_temporal_migrations

    # 4. Genetic distance
    g = 0
    if G != None and weights.gen_dist != 0:
        # calculate if 2 nodes are in diff sites and there's an edge between them (i.e there is a migration edge)
        R = torch.mul(A, (1-X))
        R = -1.0*torch.mul(R, G)
        g = torch.sum(R)

    # 5. Organotropism
    o = 0
    if O != None and weights.organotrop != 0:
        # the organotropism frequencies can only be used on the first 
        # row, which is for the migrations from primary cancer site to
        # other metastatic sites (we don't have frequencies for every 
        # site to site migration)
        organ_penalty = -1.0*torch.mul(O, site_adj_no_diag[0,:])
        o = torch.sum(organ_penalty)


    # Combine all 5 components with their weights
    vertex_labeling_loss = (weights.mig*m + weights.seed_site*s + weights.comig*c + weights.gen_dist*g + weights.organotrop*o)
    loss_components = {"mig": m.item(), "comig": c.item(), "seeding": s.item(), 
                       "organo": 0 if o == 0 else round(o.item(), 3), "gen": 0 if g == 0 else round(g.item(), 3)}

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
    # TODO: why divide by np.log(2)
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
    loss_components = {"nll": round(nlglh.item(), 3), "reg": reg.item()}
    return subclonal_presence_loss, loss_components

def no_cna_omega(shape):
    '''
    Returns omega values assuming no copy number alterations (0.5)
    Shape is (num_anatomical_sites x num_mutation_clusters)
    '''
    # TODO: don't hardcode omega here (omegas are all 1/2 since we're assuming there are no CNAs)
    return torch.ones(shape) * 0.5

def get_lambdas(epoch, max_iter, lr_sched):
    '''
    Returns weights for subclonal presence loss and ancestral labeling loss,
    and bools indicating if we need to calculate that loss at this iteration
    (for perf optimization)
    '''
    calc_subclonal, calc_ancestral = True, True
    lam1, lam2 = 1.0, 1.0
    if epoch != -1: # to evaluate loss values after training is done
        if lr_sched == "step":   
            if epoch < max_iter/2:
                lam1, lam2 = 1.0, 0.0
                calc_ancestral = False
            else:
                lam1, lam2 = 0.0, 1.0
                calc_subclonal = False
        elif lr_sched == "em": 
            if ((epoch//20) % 2 == 0):
                lam1, lam2 = 1.0, 0.0
                calc_ancestral = False
            else:
                lam1, lam2 = 0.0, 1.0
                calc_subclonal = False
        elif lr_sched == "linear":
            l = (epoch+1)*(1.0/max_iter)
            lam1 = 1.0 - l
            lam2 = l

    return lam1, lam2, calc_subclonal, calc_ancestral

def objective(V, A, ref, var, U, B, G, O, weights, epoch, max_iter, lr_sched):
    '''
    Args:
        V: Vertex labeling of the full tree (num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        ref: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        U: Mixture matrix (num_sites x num_internal_nodes)
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        weights: Weights object
        epoch: number of current epoch, used to determine weighting in different learning rate schemes
        max_iter: the maximum number of iterations to be run
        lr_sched: how to weight the two tasks, see 'get_migration_history(...)' for documentation

    Returns:
        Loss to score this tree and labeling combo.
    '''

    lam1, lam2, calc_subclonal, calc_ancestral = get_lambdas(epoch, max_iter, lr_sched)

    if calc_ancestral:
        ancestral_labeling_loss, loss_dict_a = _ancestral_labeling_objective(V, A, G, O, weights)
    else:
        ancestral_labeling_loss, loss_dict_a = 0.0, {}
   
    if calc_subclonal:
        omega_v = no_cna_omega(ref.shape)
        subclonal_presence_loss, loss_dict_b = _subclonal_presence_objective(ref, var, omega_v, U, B, weights)
    else:
        subclonal_presence_loss, loss_dict_b = 0.0, {}

    loss = (lam1*subclonal_presence_loss) + (lam2*ancestral_labeling_loss)

    loss_components = {**loss_dict_a, **loss_dict_b, **{"loss": round(loss.item(), 3)}}

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


def compute_losses(psi, X, T, ref, var, B, p, G, O, temp, hard, weights, epoch, max_iter, lr_sched):
    '''
    Takes latent variables U and X (both of size batch_size x num_internal_nodes x num_sites)
    and computes loss for each gumbel-softmax estimated training example.
    '''
    # Using the softmax enforces that the row sums are 1, since the proprtions of
    # subclones in a given site should sum to 1
    U = torch.softmax(psi, dim=2)

    assert(U.shape[0] == X.shape[0])

    batch_size = U.shape[0]

    # TODO test indexing in this function
    def vertex_labeling(i, U, X, p, T):
        '''
        Get the ith example of the inputs U and X (both of size batch_size x num_internal_nodes x num_sites)
        to get the anatomical sites of the leaf nodes and the internal nodes (respectively). If the root labeling
        p is known, then stack the root labeling to get the full vertex labeling V. If the root labeling is not known,
        X has an extra column (the first column) used to learn the site of the primary.
        '''
        # TODO: make helper functions for the U stuff
        U_i = U[i,:,:][:,1:] # don't include column for normal cells
        num_sites = U.shape[1]
        L = torch.nn.functional.one_hot((U_i > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
        # TODO: is this indexing the way to handle two latent vars?? probs not
        X_i = X[i,:,:] # internal labeling
        if p is None:
            return torch.hstack((X_i, L))
        # p needs to be placed at the index of the root node
        full_vert_labeling = torch.hstack((X_i, L))
        # Index where vector p should be inserted
        root_idx = plot_util.get_root_index(T)
        # Split the tensor into two parts at the specified index
        left_part = full_vert_labeling[:, :root_idx]
        right_part = full_vert_labeling[:, root_idx:]

        # Concatenate the left part, new column, and right part along the second dimension
        return torch.cat((left_part, p, right_part), dim=1)

    def full_adj_matrix(i, T, U, G):
        '''
        All non-zero values of U represent extant clones (leaf nodes of the full tree).
        For each of these non-zero values, we add an edge from parent clone to extant clone.
        See MACHINA Supp. Fig. 23 for a visual.
        '''
        U_i = U[i,:,:][:,1:] # don't include column for normal cells
        num_leaves = (U_i > U_CUTOFF).nonzero().shape[0]
        num_internal_nodes = T.shape[0]
        full_adj = torch.nn.functional.pad(input=T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
        leaf_idx = num_internal_nodes
        # Also add branch lengths (genetic distances) for the edges we're adding.
        # Since all of these edges we're adding represent genetically identical clones,
        # we are going to add a very small but non-zero value.
        full_G = torch.nn.functional.pad(input=G, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0) if G is not None else None

        # Iterate through the internal nodes that have nonzero values
        for internal_node_idx in (U_i > U_CUTOFF).nonzero()[:,1]:
            full_adj[internal_node_idx, leaf_idx] = 1
            if G is not None:
                full_G[internal_node_idx, leaf_idx] = G_IDENTICAL_CLONE_VALUE
            leaf_idx += 1

        return full_adj, full_G

    # For each i in batch_size, collect the loss, vertex labeling, full adj matrix, and full branch length matrix
    losses_list = []
    V_list = []
    full_trees_list = []
    full_branch_lengths_list = []
    loss_components_list = []
    softmax_X, softmax_X_soft = gumbel_softmax(X, temp, hard)

    # TODO: performance wise it is probably faster not to iterate like this but to
    # do the matrix operations together - although not sure if we can due to dimensionality issues?
    for idx in range(batch_size):
        V = vertex_labeling(idx, U, softmax_X, p, T)
        full_T, full_G = full_adj_matrix(idx, T, U, G)
        loss, loss_components = objective(V, full_T, ref, var, U[idx,:,:], B, full_G, O, weights, epoch, max_iter, lr_sched)
        losses_list.append(loss)
        loss_components_list.append(loss_components)
        V_list.append(V)
        full_trees_list.append(full_T)
        full_branch_lengths_list.append(full_G)

    return torch.stack(losses_list), V_list, full_trees_list, full_branch_lengths_list, softmax_X_soft, loss_components_list
    
class UNet(torch.nn.Module):
    def __init__(self, psi):
        super().__init__()
        self.psi = torch.nn.Parameter(psi, requires_grad=True)

    def forward(self, X, T, ref, var, B, p, G, O, temp, hard, weights, i, max_iter, lr_sched):
        omega_v = no_cna_omega(ref.shape)
        losses, V, trees, branch_lengths, softmax_Xs, loss_comps = compute_losses(self.psi, X, T, ref, var, 
                                                                                  B, p, G, O, temp, hard, weights, i, 
                                                                                  max_iter, lr_sched)
        return losses, V, trees, branch_lengths, softmax_Xs, loss_comps

def get_avg_loss_components(loss_components):
    '''
    Calculate the averages of each loss component (e.g. "nll" 
    (negative log likelihood), "mig" (migration number), etc.)
    '''
    d = {}
    for key in loss_components[0]:
        if key not in d:
            d[key] = 0
        for e in loss_components:
            d[key] += e[key]
    d = {key:d[key]/len(loss_components) for key in d} # calc averages across batch
    return d

def get_migration_history(T, ref, var, ordered_sites, p, node_idx_to_label,
                          weights, print_config, output_dir, run_name, 
                          G=None, O=None, max_iter=200, lr=0.1, init_temp=40, final_temp=0.01,
                          batch_size=64, custom_colors=None, weight_init_primary=False, lr_sched="step"):
    '''
    Args:
        T: Adjacency matrix (directed) of the internal nodes (shape: num_internal_nodes x num_internal_nodes)
        
        ref: Reference matrix (num_anatomical_sites x num_mutation_clusters), i.e., num. reads that map to reference allele
        
        var: Variant matrix (num_anatomical_sites x num_mutation_clusters), i.e., num. reads that map to variant allele
        
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung_met"]) with length =  num_anatomical_sites) and 
        the order matches the order of sites in the ref and var
        
        weights: Weight object for how much to penalize each component of the loss
        
        print_config: PrintConfig object with options on how to visualize output
        
        node_idx_to_label: dictionary mapping vertex indices (corresponding to their index in T) to custom labels
        for plotting

        p: one-hot vector (shape: num_anatomical_sites x 1) indicating the location
        of the primary tumor (root vertex must be labeled with the primary)
        
        output_dir: path for where to save output trees to

        run_name: e.g. patient name

    Optional:
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        
        O: Matrix of organotropism values between sites (shape: num_anatomical_sites x  num_anatomical_sites).

        weight_init_primary: whether to initialize weights higher to favor vertex labeling of primary for all internal nodes
        
        lr_sched: how to weight the tasks of (1) leaf node inference and (2) internal vertex labeling. options:
            "bi-level": outer objective is to learn vertex labeling, inner objective is to learn leaf nodes (subclonal presence)
            "constant": default, (1) and (2) weighted equally at each epoch
            "step": (1) has weight=1 for first half of epochs and (2) has weight=0, and then we flip the weights for the last half of training
            "em": (1) has weight=1 and (2) has weight=0 for 20 epochs, and we flip every 20 epochs (kind of like E-M)
            "linear": (1) decreases linearly while (2) increases linearly at the same rate (1/max_iter)
    Returns:
        # TODO: return info for k best trees
        Corresponding info on the *best* tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
        (4) dictionary w/ loss values for each component of the loss
        (5) how long (in seconds) the algorithm took to run
    '''

    # TODO: test these assertions
    if not (T.shape[0] == T.shape[1]):
        raise ValueError(f"Number of tree nodes should be consistent (T.shape[0] == T.shape[1])")
    if ref.shape != var.shape:
        raise ValueError(f"ref and var must have identical shape, got {ref.shape} and {var.shape}")
    if not (ref.shape[1] == var.shape[1] == T.shape[0]):
        raise ValueError(f"Number of mutations/mutation clusters should be consistent (ref.shape[1] == var.shape[1] == T.shape[0])")
    if not (ref.shape[0] == var.shape[0] == len(ordered_sites)):   
        raise ValueError(f"Length of ordered_sites should be equal to ref and var dim 0")
    if not vert_util.is_tree(T):
        raise ValueError("Adjacency matrix T is empty or not a tree.")
    if not os.path.isdir(output_dir):
        raise ValueError(f"{output_dir} does not exist.")

    B = vert_util.get_mutation_matrix_tensor(T)
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
        if p is None: raise ValueError(f"Cannot use weight_init_primary flag without inputting p vector ")
        prim_site_idx = torch.nonzero(p)[0][0]
        X[:batch_size//2,prim_site_idx,:] = 5
        # TODO: applying softmax on iterations where we're not learning vertex labeling
        # lessens the effect of this weight initialization

    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and antomical locations of the extant clones (U > U_CUTOFF)
    psi = -1 * torch.rand(batch_size, num_sites, num_internal_nodes + 1) # an extra column for normal cells
    if lr_sched == 'bi-level':
        X = torch.nn.Parameter(X, requires_grad=True)
        unet = UNet(psi)
        meta_optim = torchopt.MetaAdam(unet, lr=lr)
    else:
        psi.requires_grad = True 
        X.requires_grad = True
        optimizer = torch.optim.Adam([psi, X], lr=lr)

    # add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # add a column of ones to indicate that every subclone has the non-cancerous mutations
    B = torch.hstack ([torch.ones(B.shape[0]).reshape(-1,1), B])

    # Temperature and annealing
    temp = init_temp
    decay = (init_temp - final_temp) / max_iter
    hard = True

    max_patience_epochs = 20
    early_stopping_ctr = 0
    eps = 1e-2
    anneal_rate = 0.002
    intermediate_data = []
    temps = []
    start_time = datetime.datetime.now()
    all_loss_components = []

    for i in tqdm(range(max_iter)):

        if lr_sched == 'bi-level':
            inner_losses, V, trees, branch_lengths, Xs, loss_comps = unet(X, T, ref, var, B, p, G, O, 
                                                                          temp, hard, weights, i, max_iter, lr_sched)
            inner_loss = torch.mean(inner_losses)
            
            meta_optim.step(inner_loss)
            with torch.no_grad():
                if i % 20 == 0:
                    intermediate_data.append([inner_losses, trees, V, torch.softmax(psi, dim=2), branch_lengths, Xs])

        else:

            optimizer.zero_grad()
            losses_tensor, V, trees, branch_lengths, Xs, loss_comps = compute_losses(psi, X, T, ref, var, 
                                                                                     B, p, G, O, temp, hard, weights, i, 
                                                                                     max_iter, lr_sched)
            loss = torch.mean(losses_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if i % 20 == 0:
                    intermediate_data.append([losses_tensor, trees, V, torch.softmax(psi, dim=2), branch_lengths, Xs])

        all_loss_components.append(get_avg_loss_components(loss_comps))

        if i % 10 == 0:
            if lr_sched != "step" or (lr_sched == "step" and i > max_iter/2):
                temp = np.maximum(temp * np.exp(-anneal_rate * i), final_temp)
        temps.append(temp)

    if lr_sched == 'bi-level':
        outer_losses, V, trees, branch_lengths, Xs, loss_comps = unet(X, T, ref, var, B, p, G, O, 
                                                                      temp, hard, weights, i, max_iter, lr_sched)
        outer_loss = torch.mean(outer_losses)
        outer_loss.backward()


    if print_config.visualize:
        plot_util.plot_loss_components(all_loss_components, weights)
        plot_util.plot_temps(temps)

    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")

    with torch.no_grad():
        if lr_sched == 'bi-level': psi = unet.psi
        losses, V, trees, branch_lengths, _, _ = compute_losses(psi, X, T, ref, var, B, p, G, O, temp, hard, weights, -1, max_iter, lr_sched)
        _, best_tree_idx = torch.topk(losses, 1, largest=False, sorted=True)
        edges, vert_to_site_map, mig_graph_edges, loss_info = plot_util.print_best_trees(losses, V, torch.softmax(psi, dim=2), trees, 
                                                                                         branch_lengths, ref, var, B, O, G, weights, 
                                                                                         node_idx_to_label, ordered_sites,
                                                                                         print_config, intermediate_data, 
                                                                                         custom_colors, primary_site_label, 
                                                                                         max_iter, output_dir, run_name)

        # avg_tree = plot_util.print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors,
        #                                     ordered_sites, print_config)

       

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed

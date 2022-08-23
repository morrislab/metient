
import torch
import logging
import sys
from scipy import stats
import numpy as np

from src.util import vertex_labeling_util
from torch.distributions.binomial import Binomial

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

logger = logging.getLogger('SGD')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n\r%(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

U_CUTOFF = 0.05

class LabeledTree:
    def __init__(self, tree, labeling, U, loss):
        self.tree = tree
        self.labeling = labeling
        self.U = U
        self.loss = loss

    def __eq__(self, other):
        return isinstance(other, LabeledTree) and self.tree == other.tree and self.labeling == other.labeling

    def __hash__(self):
        return hash((self.tree, self.labeling))

def _truncated_cluster_name(cluster_name):
    '''
    Displays a max of two mutation names associated with the cluster (e.g. 9;15;19;23;26 -> 9;15)
    Does nothing if the cluster name is not in above format
    '''
    assert(isinstance(cluster_name, str))
    split_name = cluster_name.split(";")
    truncated_name = ";".join(split_name) if len(split_name) <= 2 else ";".join(split_name[:2])
    return truncated_name

# Adapted from PairTree
def _calc_llh(F_hat, R, V, omega_v, epsilon=1e-5):
    '''
    Args:
        F_hat: estimated subclonal frequency matrix (num_nodes x num_mutation_clusters)
        R: Refernce matrix (num_samples x num_mutation_clusters)
        V: Variant matrix (num_samples x num_mutation_clusters)
    Returns:
        Data fit using the Binomial likelihood (p(x|F_hat)). See PairTree
        supplement (9.3.2) for details.
    '''

    N = R + V
    S, K = F_hat.shape
    #print("K", K)
    #print("omega V\n", omega_v)

    for matrix in V, N, omega_v:
        assert(matrix.shape == (S, K-1))

    # TODO: how do we make sure the non-cancerous root clone subclonal frequencies here are 1
    #assert(np.allclose(1, phi[0]))
    #print("F_hat", F_hat.shape, "\n", F_hat)
    P = torch.mul(omega_v, F_hat[:,1:])
    #print("P", P.shape, "\n", P)

    # TODO: why these cutoffs?
    #P = torch.maximum(P, epsilon)
    #P = torch.minimum(P, 1 - epsilon)

    bin_dist = Binomial(N, P)
    F_llh = bin_dist.log_prob(V)
    #print("F_llh", F_llh.shape, "\n", F_llh)
    # TODO: why divide by np.log(2)
    #phi_llh = stats.binom.logpmf(V, N, P) / np.log(2)
    assert(not torch.any(F_llh.isnan()))
    assert(not torch.any(F_llh.isinf()))

    # TODO: why the division by K-1 and S?
    llh_per_sample = -torch.sum(F_llh, axis=1) / (K-1)
    nlglh = torch.sum(llh_per_sample) / S
    #print("nlglh", nlglh)
    return (F_llh, llh_per_sample, nlglh)

def objective(V, A, T, ref_matrix, var_matrix, U, B, w_e, w_m, w_s, w_c, w_l, alpha=100.0, verbose=False):
    '''
    Args:
        V: Vertex labeling of the full tree (num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        T: Adjacency matrix (directed) of the internal tree (num_internal_nodes x num_internal_nodes)
        ref_matrix: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var_matrix: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        U: Mixture matrix (num_sites x num_internal_nodes)
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)

    Returns:
        Loss to score this tree and labeling combo.
    '''

    # Migration number
    VA = V @ A
    site_adj = VA @ V.T
    m = torch.sum(site_adj) - torch.trace(site_adj)

    # Seeding site number
    # remove the same site transitions from the site adj matrix
    site_adj_no_diag = torch.mul(site_adj, 1-torch.eye(site_adj.shape[0], site_adj.shape[1]))
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=1)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    binarized_row_sums_site_adj = torch.sigmoid(alpha * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj)

    # Comigration number
    W = VA.T @ VA # W tells us if two nodes' parents are the same color
    X = V.T @ V # X tells us if two nodes are the same color
    Y = torch.sum(torch.mul(VA.T, 1-V.T), axis=1) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X)
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    P = vertex_labeling_util.get_path_matrix_tensor(A)
    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=1)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y))
    binarized_site_adj = torch.sigmoid(alpha * (2 * site_adj - 1))
    c = torch.sum(binarized_site_adj) - torch.trace(binarized_site_adj) + repeated_temporal_migrations
    # Data fit
    F_hat = (U @ B)

    # TODO: don't hardcode omega_v here (omegas are all 1/2 since we're assuming there are no CNAs)
    omega_v = torch.ones(ref_matrix.shape) * 0.5
    F_llh, llh_per_sample, nlglh = _calc_llh(F_hat, ref_matrix, var_matrix, omega_v)

    # Regularization to make some values of U -> 0
    l1 = torch.sum(U)

    loss = w_e*nlglh + w_m*m + w_s*s + w_c*c + w_l*l1

    if verbose:
        print("Migration number:", m.item())
        print("Comigration number:", c.item())
        print("Seeding site number:", s.item())
        print("Neg log likelihood:", round(nlglh.item(), 3))
        print("L1:", l1.item())
        print("Loss:", round(loss.item(), 3))

    return loss

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
    return y


def compute_losses(U, X, T, ref_matrix, var_matrix, B, p, batch_size, temp, hard, w_e, w_m, w_s, w_c, w_l):
    '''
    Takes input X (batch_size x num_nodes x num_sites) and computes loss
    for each gumbel-softmax estimated training example
    '''

    def vertex_labeling(U, X, i, p):
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
        #L = torch.nn.functional.one_hot(U_i.nonzero()[:,1], num_classes=num_sites).T
        # TODO: is this indexing the way to handle two latent vars?? probs not
        X_i = X[i,:,:] # internal labeling
        if p is None:
            return torch.hstack((X_i, L))
        return torch.hstack((p, X_i, L))

    def full_adj_matrix(T, U, i):
        U_i = U[i,:,:][:,1:] # don't include column for normal cells
        #num_leaves = U_i.nonzero().shape[0]
        num_leaves = (U_i > U_CUTOFF).nonzero().shape[0]
        num_internal_nodes = T.shape[0]
        full_adj = torch.nn.functional.pad(input=T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
        leaf_idx = num_internal_nodes
        # Iterate through the internal nodes that have nonzero values
        for internal_node_idx in (U_i > U_CUTOFF).nonzero()[:,1]:
            full_adj[internal_node_idx, leaf_idx] = 1
            leaf_idx += 1
        return full_adj


    losses_list = []
    V_list = []
    full_trees = []
    softmax_X = gumbel_softmax(X, temp, hard)
    for idx in range(batch_size):
        V = vertex_labeling(U, softmax_X, idx, p)
        full_T = full_adj_matrix(T, U, idx)
        loss = objective(V, full_T, T, ref_matrix, var_matrix, U[idx,:,:], B, w_e, w_m, w_s, w_c, w_l)
        losses_list.append(loss)
        V_list.append(V)
        full_trees.append(full_T)

    return V_list, torch.stack(losses_list), full_trees

def gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites,
                                p=None, node_idx_to_label=None,
                                w_e=1.0, w_m=1.0, w_s=1.0, w_c=1.0, w_l=1.0,
                                max_iter=100, lr = 0.1,
                                init_temp=20, final_temp=0.1, batch_size=128,
                                custom_colors=None, primary=None, visualize=True,
                                show_top_trees=False):
    '''
    Args:
        T: Adjacency matrix (directed) of the internal nodes (shape: num_internal_nodes x num_internal_nodes)
        ref_matrix: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var_matrix: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung_met"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        p: one-hot vector (shape: num_anatomical_sites x 1) indicating the location
        of the primary tumor (root vertex must be labeled with the primary)
        node_idx_to_label: dictionary mapping vertex indices (corresponding to their index in T) to custom labels

    Returns:
        Corresponding info on the best learned tree:
        (1) edges of the tree (e.g. [('0', '1'), ('1', '2;3')])
        (2) vertex labeling as a dictionary (e.g. {'0': 'P', '1;3': 'M1'}),
        (3) edges for the migration graph (e.g. [('P', 'M1')])
    '''
    assert(T.shape[0] == T.shape[1] == B.shape[0])
    assert(ref_matrix.shape[1] == var_matrix.shape[1] == B.shape[1])
    assert(ref_matrix.shape[0] == var_matrix.shape[0] == len(ordered_sites))
    assert(ref_matrix.shape == var_matrix.shape)

    num_sites = ref_matrix.shape[0]
    num_internal_nodes = T.shape[0]

    psi = -1 * torch.rand(batch_size, num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi.requires_grad = True # we're learning psi
    # If we don't know the anatomical site of the primary tumor, we need to learn it
    num_nodes_to_label = -1
    if p is None:
        num_nodes_to_label = num_internal_nodes
    else:
        assert(p.shape[1] == 1)
        assert(p.shape[0] == ref_matrix.shape[0]) # num_anatomical_sites
        num_nodes_to_label = num_internal_nodes - 1 # we don't need to learn the root labeling

    X = -1 * torch.rand(batch_size, num_sites, num_nodes_to_label)
    X.requires_grad = True # we're learning X (this is the vertex labeling V)

    # add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # add a column of ones to indicate that every subclone has the non-cancerous mutations
    B = torch.hstack ([torch.ones(B.shape[0]).reshape(-1,1), B])

    # Temperature and annealing
    temp = init_temp
    decay = (init_temp - final_temp) / max_iter
    hard = True

    optimizer = torch.optim.Adam([psi, X], lr=lr)
    min_loss = torch.tensor(float("Inf"))
    min_loss_labeling = None
    min_U = None
    all_min_loss_labeled_trees = None
    min_tree = None
    losses = []

    for i in range(max_iter):
        optimizer.zero_grad()
        # Using the softmax enforces that the row sums are 1, since the proprtions of
        # subclones in a given site should sum to 1
        U = torch.softmax(psi, dim=2)
        V, losses_tensor, full_trees = compute_losses(U, X, T, ref_matrix, var_matrix, B, p, batch_size, temp, hard, w_e, w_m, w_s, w_c, w_l)
        loss = torch.mean(losses_tensor)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        temp -= decay # drop temperature

        with torch.no_grad():
            V, losses_tensor, full_trees = compute_losses(U, X, T, ref_matrix, var_matrix, B, p, batch_size, temp, hard, w_e, w_m, w_s, w_c, w_l)
            min_loss_iter = torch.min(losses_tensor)
            idx = torch.argmin(losses_tensor)

            _, min_loss_indices = torch.topk(losses_tensor, 3, largest=False)
            #indices = (losses_tensor == min_loss_iter).nonzero()
            if min_loss_iter < min_loss:
                # TODO: put this into a helper function
                min_loss = min_loss_iter
                min_loss_labeling = V[idx]
                min_tree = full_trees[idx]
                min_U = U[idx,:,:]
                min_psi = psi[idx,:,:]

                all_min_loss_labeled_trees = set()
                for i in min_loss_indices:
                    labeled_tree = LabeledTree(full_trees[i], V[i], U[i], losses_tensor[i])
                    all_min_loss_labeled_trees.add(labeled_tree)

    if visualize:
        vertex_labeling_util.plot_losses(losses)

    with torch.no_grad():
        best_tree = None
        min_loss = float("inf")
        print("all_min_loss_labeled_trees", all_min_loss_labeled_trees)
        for i, min_loss_labeled_tree in enumerate(all_min_loss_labeled_trees):
            labeling, tree, U = min_loss_labeled_tree.labeling, min_loss_labeled_tree.tree, min_loss_labeled_tree.U
            loss = objective(labeling, tree, T, ref_matrix, var_matrix, U, B, w_e, w_m, w_s, w_c, w_l, verbose=show_top_trees)

            if show_top_trees:
                print(f"Tree {i+1}")
                U_clipped = U.detach().numpy()
                U_clipped[np.where(U_clipped<U_CUTOFF)] = 0
                print(f"U > {U_CUTOFF}\n")
                col_labels = ["norm"] + [_truncated_cluster_name(node_idx_to_label[k]) if k in node_idx_to_label else "0" for k in range(U_clipped.shape[1] - 1)]
                print(col_labels)
                df = pd.DataFrame(U_clipped, columns=col_labels, index=ordered_sites)
                print(df)
                print("F_hat")
                print(U @ B)

                if visualize:
                    vertex_labeling_util.plot_tree(labeling, tree, ordered_sites, custom_colors, node_idx_to_label)
                    vertex_labeling_util.plot_migration_graph(labeling, tree, ordered_sites, custom_colors, primary)
                print("-"*100 + "\n")

            if loss < min_loss:
                best_tree = min_loss_labeled_tree
                min_loss = loss

        if visualize: print("\nBest tree")
        loss = objective(best_tree.labeling, best_tree.tree, T, ref_matrix, var_matrix, best_tree.U, B, w_e, w_m, w_s, w_c, w_l, verbose=True)
        U_clipped = U.detach().numpy()
        U_clipped[np.where(U_clipped<U_CUTOFF)] = 0
        logger.debug(f"\nU > {U_CUTOFF}\n")
        col_labels = ["norm"] + [_truncated_cluster_name(node_idx_to_label[k]) if k in node_idx_to_label else "0" for k in range(U_clipped.shape[1] - 1)]
        df = pd.DataFrame(U_clipped, columns=col_labels, index=ordered_sites)
        logger.debug(df)
        logger.debug("\nF_hat")
        logger.debug(U @ B)

    best_tree_edges, best_tree_vertex_name_to_site_map = vertex_labeling_util.plot_tree(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, node_idx_to_label, show=visualize)
    best_mig_graph_edges = vertex_labeling_util.plot_migration_graph(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, primary, show=visualize)

    return best_tree_edges, best_tree_vertex_name_to_site_map, best_mig_graph_edges

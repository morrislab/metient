
import torch
import logging
import sys
import numpy as np
import datetime

from src.util import vertex_labeling_util
from torch.distributions.binomial import Binomial

import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

logger = logging.getLogger('SGD')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n\r%(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# TODO: how do we validate this
U_CUTOFF = 0.05
# TODO: better way to handle this?
G_IDENTICAL_CLONE_VALUE = -10.0

from pprint import pprint

print("CUDA GPU:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class PrintConfig:
    def __init__(self, visualize=True, verbose=False, viz_intermeds=False, k_best_trees=1):
        '''
        visualize: bool, whether to visualize loss, best tree, and migration graph
        verbose: bool, whether to print debug info
        viz_intermeds: bool, whether to visualize intermediate solutions to best tree
        k_best_trees: int, number of best tree solutions to visualize (if 1, only show best tree)
        '''
        self.visualize = visualize
        self.verbose = verbose 
        self.viz_intermeds = viz_intermeds
        self.k_best_trees = k_best_trees

class Weights:
    def __init__(self, data_fit=1.0, mig=1.0, comig=1.0, seed_site=1.0, reg=1.0, gen_dist=1.0, organotrop=1.0):
        self.data_fit = data_fit
        self.mig = mig
        self.comig = comig
        self.seed_site = seed_site
        self.reg = reg
        self.gen_dist = gen_dist
        self.organotrop = organotrop

class LabeledTree:
    def __init__(self, tree, labeling, U, branch_lengths):
        self.tree = tree
        self.labeling = labeling
        self.U = U
        self.branch_lengths = branch_lengths

    # TODO: write unit tests for this
    def __eq__(self, other):
        return ( isinstance(other, LabeledTree) and
               str(np.where(self.tree == 1)[0]) == str(np.where(other.tree == 1)[0]) and
               str(np.where(self.tree == 1)[1]) == str(np.where(other.tree == 1)[1]) and
               str(np.where(self.labeling == 1)[0]) == str(np.where(other.labeling == 1)[0]) and
               str(np.where(self.labeling == 1)[1]) == str(np.where(other.labeling == 1)[0])
               )

    def __hash__(self):
        hsh = hash((str(np.where(self.tree == 1)[0]), str(np.where(self.tree == 1)[1]),
                    str(np.where(self.labeling == 1)[0]), str(np.where(self.labeling == 1)[1])))
        return hsh

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

def objective(V, A, ref_matrix, var_matrix, U, B, G, O, weights, epoch, max_iter, print_config=None):
    '''
    Args:
        V: Vertex labeling of the full tree (num_sites x num_nodes)
        A: Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
        ref_matrix: Reference matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to reference allele
        var_matrix: Variant matrix (num_anatomical_sites x num_mutation_clusters). Num. reads that map to variant allele
        U: Mixture matrix (num_sites x num_internal_nodes)
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Matrix of organotropism values between sites (shape: num_anatomical_sites x  num_anatomical_sites).

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
    alpha = 100.0
    binarized_row_sums_site_adj = torch.sigmoid(alpha * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj)

    # Comigration number
    W = VA.T @ VA # W tells us if two nodes' parents are the same color
    X = V.T @ V # X tells us if two nodes are the same color
    Y = torch.sum(torch.mul(VA.T, 1-V.T), axis=1) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X)
    # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # and (3) there's a path from node i to node j
    # TODO: this is computationally expensive, maybe we could cache path matrices we've calculated before?
    P = vertex_labeling_util.get_path_matrix_tensor(A.cpu().numpy())
    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=1)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y))
    binarized_site_adj = torch.sigmoid(alpha * (2 * site_adj - 1))
    c = torch.sum(binarized_site_adj) - torch.trace(binarized_site_adj) + repeated_temporal_migrations

    # Data fit
    F_hat = (U @ B)

    # TODO: don't hardcode omega_v here (omegas are all 1/2 since we're assuming there are no CNAs)
    omega_v = torch.ones(ref_matrix.shape) * 0.5
    F_llh, llh_per_sample, nlglh = _calc_llh(F_hat, ref_matrix, var_matrix, omega_v)

    g = 0
    if G != None:
    # Weigh by genetic distance
    # calculate if 2 nodes are in diff sites and there's an edge between them (i.e there is a migration edge)
        R = torch.mul(A, (1-X))
        R = -1.0*torch.mul(R, G)
        g = torch.sum(R)

    o = 0
    if O != None:
        #organ_penalty = torch.mul((1 - O) , site_adj_no_diag)
        organ_penalty = -1.0*torch.mul(O, site_adj_no_diag)
        o = torch.sum(organ_penalty)

    # Regularization to make some values of U -> 0
    reg = torch.sum(U)

    lam1, lam2 = 1.0, 1.0
    if epoch != -1: # to evaluate loss values after training is done   
        # l = max(0.01, 1.0/(1.0+np.exp(-1.0*(epoch-20))))
        l = (epoch+1)*(1.0/max_iter)
        lam1 = 1.0 - l
        lam2 = l
        # if epoch % 50 == 0:
        #     print(f"epoch {epoch}, lam1: {lam1}, lam2: {lam2}")

    # sigmoid-like schedule, learns leaf nodes first and then vertex labeling
    loss = (lam1*(weights.data_fit*nlglh + weights.reg*reg)) + (lam2*(weights.mig*m + weights.seed_site*s + weights.comig*c + weights.gen_dist*g + weights.organotrop*o))

    loss_components = {"mig": m.item(), "comig": c.item(), "seeding": s.item(), "nll": round(nlglh.item(), 3), "reg": reg.item(), "gen": 0 if g == 0 else round(g.item(), 3), "loss": loss.item()}
    if print_config and print_config.verbose:
        print("Migration number:", m.item())
        print("Comigration number:", c.item())
        print("Seeding site number:", s.item())
        print("Neg log likelihood:", round(nlglh.item(), 3))
        print("Reg:", reg.item())
        if g != 0:
            print("Genetic distance:", round(g.item(), 3))

        if o != 0:
            print("Organotropism penalty:", round(o.item(), 3))
            print("site_adj_no_diag\n", site_adj_no_diag)
            print("(1 - O)", (1 - O))
            print("O\n", O)
            print("organ_penalty\n", organ_penalty)
        print("Loss:", round(loss.item(), 3))

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


def compute_losses(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, epoch, max_iter):
    '''
    Takes latent variables U and X (both of size batch_size x num_internal_nodes x num_sites)
    and computes loss for each gumbel-softmax estimated training example.
    '''

    assert(U.shape[0] == X.shape[0])

    batch_size = U.shape[0]

    def vertex_labeling(i, U, X, p):
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
        return torch.hstack((p, X_i, L))

    def full_adj_matrix(i, T, U, G):
        '''
        All values non-zero values of U represent extant clones (leaf nodes of the full tree).
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
        V = vertex_labeling(idx, U, softmax_X, p)
        full_T, full_G = full_adj_matrix(idx, T, U, G)
        loss, loss_components = objective(V, full_T, ref_matrix, var_matrix, U[idx,:,:], B, full_G, O, weights, epoch, max_iter)
        losses_list.append(loss)
        loss_components_list.append(loss_components)
        V_list.append(V)
        full_trees_list.append(full_T)
        full_branch_lengths_list.append(full_G)

    return torch.stack(losses_list), V_list, full_trees_list, full_branch_lengths_list, softmax_X_soft, loss_components_list

def get_averaged_tree(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, max_iter):
    '''
    Returns an averaged tree over all (TODO: all or top k?) converged trees
    by weighing each tree edge by the softmax of the likelihood of that tree 
    '''
    losses_tensor, V, full_trees, full_branch_lengths, _, _ = compute_losses(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, -1,  max_iter)
    losses_tensor, min_loss_indices = torch.topk(losses_tensor, len(losses_tensor), largest=False, sorted=True)
    print("losses tensor\n", losses_tensor)

def print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, 
                    node_idx_to_label, ordered_sites, max_iter, print_config):
    loss, loss_components = objective(labeled_tree.labeling, labeled_tree.tree, ref_matrix, var_matrix, labeled_tree.U, B, labeled_tree.branch_lengths, O, weights, -1, max_iter, print_config)
    U_clipped = labeled_tree.U.cpu().detach().numpy()
    U_clipped[np.where(U_clipped<U_CUTOFF)] = 0
    logger.debug(f"\nU > {U_CUTOFF}\n")
    col_labels = ["norm"] + [_truncated_cluster_name(node_idx_to_label[k]) if k in node_idx_to_label else "0" for k in range(U_clipped.shape[1] - 1)]
    df = pd.DataFrame(U_clipped, columns=col_labels, index=ordered_sites)
    logger.debug(df)
    logger.debug("\nF_hat")
    logger.debug(labeled_tree.U @ B)
    return loss_components

def print_best_trees(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights,
                    node_idx_to_label, ordered_sites, print_config, intermediate_data, 
                    custom_colors, primary, max_iter):
    losses_tensor, V, full_trees, full_branch_lengths, _, _ = compute_losses(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, -1, max_iter)
    _, min_loss_indices = torch.topk(losses_tensor, print_config.k_best_trees, largest=False, sorted=True)
    print("print_config.k_best_trees", print_config.k_best_trees)
    min_loss_labeled_trees_and_losses = []
    for i in min_loss_indices:
        labeled_tree = LabeledTree(full_trees[i], V[i], U[i], full_branch_lengths[i])
        min_loss_labeled_trees_and_losses.append((labeled_tree, losses_tensor[i]))

    for i, tup in enumerate(min_loss_labeled_trees_and_losses):
        labeled_tree = tup[0]
        if i == 0:
            
            if print_config.viz_intermeds:

                best_tree_idx = min_loss_indices[0]
                for itr, data in enumerate(intermediate_data):
                    losses_tensor, full_trees, V, U, full_branch_lengths, soft_X = intermediate_data[itr][0], intermediate_data[itr][1], intermediate_data[itr][2], intermediate_data[itr][3], intermediate_data[itr][4], intermediate_data[itr][5]
                    print("="*30 + " INTERMEDIATE TREE " + "="*30+"\n")
                    print(f"Iteration: {itr*20}, Intermediate best tree idx {best_tree_idx}")
                    softx_df = pd.DataFrame(soft_X[best_tree_idx].cpu().detach().numpy().T, columns=ordered_sites, index=[node_idx_to_label[i] for i in range(1,len(node_idx_to_label))]) # skip first index (which is root, and we know the vert. label)
                    print("soft_X\n", softx_df)
                    

                    labeled_tree = LabeledTree(full_trees[best_tree_idx], V[best_tree_idx], U[best_tree_idx], full_branch_lengths[best_tree_idx])
                    print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, print_config, max_iter)
                    vertex_labeling_util.plot_tree(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, node_idx_to_label)
                    vertex_labeling_util.plot_migration_graph(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, primary)


            if print_config.visualize: 
                print("*"*30 + " BEST TREE " + "*"*30+"\n")

            best_tree = labeled_tree
            best_tree_loss_info = print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, max_iter, print_config)
            best_tree_edges, best_tree_vertex_name_to_site_map = vertex_labeling_util.plot_tree(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, node_idx_to_label, show=print_config.visualize)
            best_mig_graph_edges = vertex_labeling_util.plot_migration_graph(best_tree.labeling, best_tree.tree, ordered_sites, custom_colors, primary, show=print_config.visualize)

            #print("-"*100 + "\n")

        elif print_config.k_best_trees > 1:
            print_tree_info(labeled_tree, ref_matrix, var_matrix, B, O, weights, node_idx_to_label, ordered_sites, print_config)
            vertex_labeling_util.plot_tree(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, node_idx_to_label)
            vertex_labeling_util.plot_migration_graph(labeled_tree.labeling, labeled_tree.tree, ordered_sites, custom_colors, primary)
            print("-"*100 + "\n")

    return best_tree_edges, best_tree_vertex_name_to_site_map, best_mig_graph_edges, best_tree_loss_info
    
def gumbel_softmax_optimization(T, ref_matrix, var_matrix, B, ordered_sites, weights,
                                print_config, p=None, node_idx_to_label=None, G=None, O=None,
                                max_iter=200, lr=0.1, init_temp=20, final_temp=0.1,
                                batch_size=128, custom_colors=None, primary=None):
    '''
    Args:
        T: Adjacency matrix (directed) of the internal nodes (shape: num_internal_nodes x num_internal_nodes)
        ref_matrix: Reference matrix (num_anatomical_sites x num_mutation_clusters), i.e., num. reads that map to reference allele
        var_matrix: Variant matrix (num_anatomical_sites x num_mutation_clusters), i.e., num. reads that map to variant allele
        B: Mutation matrix (shape: num_internal_nodes x num_mutation_clusters)
        ordered_sites: array of the anatomical site names (e.g. ["breast", "lung_met"])
        with length =  num_anatomical_sites) where the order matches the order of sites
        in the ref_matrix and var_matrix
        weights: Weight object for how much to penalize each component of the loss
        print_config: PrintConfig object with options on how to visualize output
        p: one-hot vector (shape: num_anatomical_sites x 1) indicating the location
        of the primary tumor (root vertex must be labeled with the primary)
        node_idx_to_label: dictionary mapping vertex indices (corresponding to their index in T) to custom labels
        G: Matrix of genetic distances between internal nodes (shape: num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        weight_init_primary: whether to initialize weights higher to favor vertex labeling of primary for all internal nodes
        O: Matrix of organotropism values between sites (shape: num_anatomical_sites x  num_anatomical_sites).

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

    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and antomical locations of the extant clones (U > U_CUTOFF)
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

    # We're learning X, which is the vertex labeling of the internal nodes
    X = torch.rand(batch_size, num_sites, num_nodes_to_label)

    # TODO: should we do this? or should we do LR scheduling
    # if weight_init_primary:
    #     X[:,0,:] = 1

    X.requires_grad = True

    # add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # add a column of ones to indicate that every subclone has the non-cancerous mutations
    B = torch.hstack ([torch.ones(B.shape[0]).reshape(-1,1), B])

    # Temperature and annealing
    temp = init_temp
    decay = (init_temp - final_temp) / max_iter
    hard = True
    min_loss = torch.tensor(float("Inf"))
    optimizer = torch.optim.Adam([psi, X], lr=lr)
    losses = []
    max_patience_epochs = 20
    early_stopping_ctr = 0
    eps = 1e-2
    anneal_rate = 0.003
    last_loss = None

    intermediate_data = []
    # Optimize
    temps = []

    start_time = datetime.datetime.now()

    all_loss_components = []
    for i in range(max_iter):
        optimizer.zero_grad()
        # Using the softmax enforces that the row sums are 1, since the proprtions of
        # subclones in a given site should sum to 1
        U = torch.softmax(psi, dim=2)
        losses_tensor, V, full_trees, full_branch_lengths, softmax_Xs, loss_components = compute_losses(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, i, max_iter)
        # TODO: better way to calc loss across trees?
        loss = torch.mean(losses_tensor)
        loss.backward()
        losses.append(loss.item())

        d = {}
        for key in loss_components[0]:
            if key not in d:
                d[key] = 0
            for e in loss_components:
                d[key] += e[key]
        d = {key:d[key]/len(loss_components) for key in d} # calc averages across batch
        all_loss_components.append(d)

        optimizer.step()
        # temp -= decay # drop temperature
        
        if i % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * i), final_temp)
        temps.append(temp)

        with torch.no_grad():

            if i % 20 == 0:
                intermediate_data.append([losses_tensor, full_trees, V, U, full_branch_lengths, softmax_Xs])
               
    if print_config.visualize:
        #vertex_labeling_util.plot_losses(losses)
        vertex_labeling_util.plot_loss_components(all_loss_components, weights)
        #vertex_labeling_util.plot_temps(temps)

    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    if print_config.verbose:
        print(f"Time elapsed: {time_elapsed}")

    with torch.no_grad():
        edges, vert_to_site_map, mig_graph_edges, loss_info = print_best_trees(U, X, T, ref_matrix, var_matrix, B, p, G,                                                                   O, temp, hard, weights, 
                                                                               node_idx_to_label, ordered_sites,
                                                                               print_config, intermediate_data, 
                                                                               custom_colors, primary, max_iter)

        avg_tree = get_averaged_tree(U, X, T, ref_matrix, var_matrix, B, p, G, O, temp, hard, weights, max_iter)

    return edges, vert_to_site_map, mig_graph_edges, loss_info, time_elapsed

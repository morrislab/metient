import torch
from metient.util import vertex_labeling_util as vutil
from torch.distributions.binomial import Binomial
import numpy as np

class ObservedClonesSolver:
    def __init__(self, num_sites, num_internal_nodes, ref, var, omega, idx_to_observed_sites,
                 B, input_T, G, node_idx_to_label, weights, config, estimate_observed_clones):
        self.ref = ref
        self.var = var
        self.omega = omega
        self.B = B
        self.input_T = input_T
        self.G = G
        self.weights = weights
        self.config = config
        self.num_sites = num_sites
        self.num_internal_nodes = num_internal_nodes
        self.node_idx_to_label = node_idx_to_label
        self.estimate_observed_clones = estimate_observed_clones
        self.idx_to_observed_sites = idx_to_observed_sites
    
    def run(self):
        if not self.estimate_observed_clones:
            T, G, L = vutil.full_adj_matrix_from_internal_node_idx_to_sites_present(self.input_T, self.G, self.idx_to_observed_sites, 
                                                                                    self.num_sites, self.config['identical_clone_gen_dist'])
            return None, self.input_T, T, G, L, self.node_idx_to_label, self.num_internal_nodes, self.idx_to_observed_sites

        return find_umap(self)

def find_umap(u_solver):
    
    # We're learning psi, which is the mixture matrix U (U = softmax(psi)), and tells us the existence
    # and anatomical locations of the extant clones (U > U_CUTOFF)
    #psi = -1 * torch.rand(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    psi = torch.ones(u_solver.num_sites, u_solver.num_internal_nodes + 1) # an extra column for normal cells
    psi.requires_grad = True 
    u_optimizer = torch.optim.Adam([psi], lr=u_solver.config['lr'])

    i = 0
    u_prev = psi
    u_diff = 1e9
    losses = []
   
    while u_diff > 1e-6 and i < 300:
        u_optimizer.zero_grad()
        U, u_loss = compute_u_loss(psi, u_solver.ref, u_solver.var, u_solver.omega, u_solver.B, u_solver.weights)
        u_loss.backward()
        u_optimizer.step()
        u_diff = torch.abs(torch.norm(u_prev - U))
        u_prev = U
        i += 1
        losses.append(u_loss.detach().numpy())
    
    with torch.no_grad():
        full_T, full_G, L, idx_to_observed_sites = vutil.full_adj_matrix_using_inferred_observed_clones(U, u_solver.input_T, u_solver.G, u_solver.num_sites, u_solver.config['identical_clone_gen_dist'])

        # Remove any leaf nodes that aren't detected at > U_CUTOFF in any sites. These are not well estimated
        U_clones = U[:,1:]
        removal_indices = []
        for node_idx in range(U_clones.shape[1]):
            children = vutil.get_child_indices(u_solver.input_T, [node_idx])
            if node_idx not in idx_to_observed_sites and len(children) == 0:
                removal_indices.append(node_idx)
        #print("node indices not well estimated", removal_indices)
        U, input_T, T, G, node_idx_to_label, idx_to_observed_sites = vutil.remove_leaf_indices_not_observed_sites(removal_indices, U, u_solver.input_T, 
                                                                                                                  full_T, full_G, u_solver.node_idx_to_label, idx_to_observed_sites)
        num_internal_nodes = u_solver.num_internal_nodes - len(removal_indices)

    return U, input_T, T, G, L, node_idx_to_label, num_internal_nodes, idx_to_observed_sites

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
    
    return U, clone_proportion_loss

def no_cna_omega(shape):
    '''
    Returns omega values assuming no copy number alterations (0.5)
    Shape is (num_anatomical_sites x num_mutation_clusters)
    '''
    return torch.ones(shape) * 0.5
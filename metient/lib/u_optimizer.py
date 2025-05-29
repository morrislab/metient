import torch
from torch.distributions.binomial import Binomial
import numpy as np
import pandas as pd
import numpy as np

from metient.util import vertex_labeling_util as vutil
from metient.util import data_extraction_util as dutil
from metient.util.globals import MIN_VARIANCE

class ObservedClonesSolver:
    def __init__(self, num_sites, num_internal_nodes, ref, var, omega, idx_to_observed_sites,
                 input_T, G, node_collection, weights, config, estimate_observed_clones, ordered_sites):
        self.ref = ref
        self.var = var
        self.omega = omega
        self.input_T = input_T
        self.G = G
        self.weights = weights
        self.config = config
        self.num_sites = num_sites
        self.num_internal_nodes = num_internal_nodes
        self.node_collection = node_collection
        self.estimate_observed_clones = estimate_observed_clones
        self.idx_to_observed_sites = idx_to_observed_sites
        self.ordered_sites = ordered_sites
    
    def run(self):
        if not self.estimate_observed_clones:
            T, G, L, node_collection = vutil.full_adj_matrix_from_internal_node_idx_to_sites_present(self.input_T, self.G, self.idx_to_observed_sites, 
                                                                                                     self.num_sites, self.config['identical_clone_gen_dist'],
                                                                                                     self.node_collection, self.ordered_sites)
            return None, self.input_T, T, G, L, node_collection, self.num_internal_nodes, self.idx_to_observed_sites

        # return fit_u_map(self)
        return self._fit_u_mle()

    def _fit_u_mle(self):
        """
        Fits the observed clone proportions matrix U using the projection algorithm (finds an MLE estimate of U)
        """
        from metient.lib.projection import fit_F

        V, R, omega_V = self.var.T, self.ref.T, self.omega.T
        V_hat = V + 1
        T_hat = V + R + 2
        W = V_hat*(1 - V_hat/T_hat) / (T_hat*omega_V)**2

        # Convert inputs to cpu and numpy float64, as expected by the projection algorithm
        V = V.detach().cpu().numpy().astype(np.float64)
        R = R.detach().cpu().numpy().astype(np.float64)
        W = W.detach().cpu().numpy().astype(np.float64)
        omega = omega_V.detach().cpu().numpy().astype(np.float64)

        W = np.maximum(MIN_VARIANCE, W)

        parents = self._convert_adjmatrix_to_parents()

        F, U, F_llh = fit_F(parents, 
                            V, 
                            R,
                            omega,
                            W)

        U = torch.from_numpy(U).T
        return self._build_full_tree_with_witness_nodes(U)

    def _convert_adjmatrix_to_parents(self):
        """
        Converts the input adjacency matrix to a parent array
        ( a numpy array where each index represents a node, and the value at that index represents
        that nodes direct ancestor (parent) in the tree)
        """
        # Convert input_T to CPU if it's a CUDA tensor
        if torch.is_tensor(self.input_T) and self.input_T.is_cuda:
            input_T = self.input_T.detach().cpu().numpy()
        else:
            input_T = self.input_T
            
        # Add germline root to the tree
        new_T = np.zeros((input_T.shape[0] + 1, input_T.shape[0] + 1))
        new_T[1:, 1:] = input_T
        new_T[0, 1] = 1  # New root connects to old root
        adj = np.copy(new_T)
        np.fill_diagonal(adj, 0)
        return np.argmax(adj[:,1:], axis=0)

    def _build_full_tree_with_witness_nodes(self, U):
        """
        Attaches witness nodes to the inputted tree using the observed clone proportions matrix U

        Args:
            U: Observed clone proportions matrix (num_sites x num_internal_nodes)
            u_solver: ObservedClonesSolver object

        Returns:
            U: Observed clone proportions matrix (num_sites x num_internal_nodes)
            input_T: Input tree (num_internal_nodes x num_internal_nodes)
            T: Expanded tree, which includes internal nodes and leaf/witness nodes (num_tree_nodes x num_tree_nodes)
            G: Genetic distance matrix (num_tree_nodes x num_tree_nodes)
            L: Leaf node labels (num_sites x num_leaf_nodes)
            node_collection: Node collection that contains info on all nodes
            num_internal_nodes: Number of internal nodes
            idx_to_observed_sites: Node index to observed sites
        """
        with torch.no_grad():
            full_T, full_G, L, idx_to_observed_sites, node_collection = vutil.full_adj_matrix_using_inferred_observed_clones(U, self.input_T, self.G, self.num_sites, 
                                                                                                                            self.config['identical_clone_gen_dist'],
                                                                                                                            self.node_collection, self.ordered_sites)
            # Remove any leaf nodes that aren't detected at > U_CUTOFF in any sites. These are not well estimated
            U_no_normal = U[:,1:] # first column is normal cells
            removal_indices = []
            for node_idx in range(U_no_normal.shape[1]):
                children = vutil.get_child_indices(self.input_T, [node_idx])
                if node_idx not in idx_to_observed_sites and len(children) == 0:
                    removal_indices.append(node_idx)
            # print("node indices not well estimated", removal_indices)

            U, input_T, T, G, node_collection, idx_to_observed_sites = vutil.remove_leaf_indices_not_observed_sites(removal_indices, U, self.input_T, 
                                                                                                                    full_T, full_G, node_collection, idx_to_observed_sites)
            num_internal_nodes = self.num_internal_nodes - len(removal_indices)
            
        return U, input_T, T, G, L, node_collection, num_internal_nodes, idx_to_observed_sites
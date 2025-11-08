import numpy as np
import torch
import networkx as nx
import math
from queue import Queue
from metient.util.globals import *
from collections import deque
import gc
import scipy.sparse as sp
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_columns', None)

LAST_P = None

######################################################
##################### CLASSES ########################
######################################################

class MigrationHistoryNode:
    __slots__ = ['idx', 'label', 'is_witness', 'is_polytomy_resolver_node']
    def __init__(self, idx, label, is_witness=False, is_polytomy_resolver_node=False):
        self.idx = idx
        assert isinstance(label, list)
        label = [str(x) for x in label]
        self.label = label # list of mut names or ids
        self.is_witness = is_witness
        self.is_polytomy_resolver_node = is_polytomy_resolver_node

# Collection of MigrationHistoryNodes which are used for a single migration history
class MigrationHistoryNodeCollection:
    def __init__(self, mig_hist_nodes):
        idx_to_mig_hist_node = {}
        for node in mig_hist_nodes:
            idx_to_mig_hist_node[node.idx] = node
        self.idx_to_mig_hist_node = idx_to_mig_hist_node
    
    @classmethod
    def from_dict(cls, dct):
        nodes = []
        for idx in dct:
            item = dct[idx]
            node = MigrationHistoryNode(idx, item[0], is_witness=item[1], is_polytomy_resolver_node=item[2])
            nodes.append(node)
        instance = cls(nodes)  # Create an instance using the default constructor
        return instance

    def get_nodes(self):
        return list(self.idx_to_mig_hist_node.values())
        
    def idx_to_label(self):
        idx_to_label = {}
        for idx in self.idx_to_mig_hist_node:
            idx_to_label[idx] =  self.idx_to_mig_hist_node[idx].label
        return idx_to_label

    def update_index(self, old_idx, new_idx):

        if old_idx in self.idx_to_mig_hist_node:
            old_node = self.idx_to_mig_hist_node.pop(old_idx)
            old_node.idx = new_idx
            self.idx_to_mig_hist_node[new_idx] = old_node
        else:
            raise KeyError(f"Key '{old_idx}' not found in dictionary.")

    def get_node(self, idx):
        return self.idx_to_mig_hist_node[idx]
    
    def add_node(self, new_node):
        assert new_node.idx not in self.idx_to_mig_hist_node
        self.idx_to_mig_hist_node[new_node.idx] = new_node
    
    def swap_indices(self, key1, key2):
        d = self.idx_to_mig_hist_node
        assert key1 in d and key2 in d
        # Both keys exist, swap their values
        d[key1], d[key2] = d[key2], d[key1]
        d[key1].idx = key1
        d[key2].idx = key2
    
    def remove_indices_and_reindex(self, indices_to_remove):
        # Create a new dictionary to hold the re-indexed entries
        new_dict = {}
        original_dict = self.idx_to_mig_hist_node
        old_index_to_new_index = {}
        # Initialize the new index
        new_index = 0
        # Iterate through the original dictionary in sorted index order
        for old_index in sorted(original_dict.keys()):
            # Skip the indices that need to be removed
            if old_index in indices_to_remove:
                continue
            
            # Assign the new index to the current node
            new_dict[new_index] = original_dict[old_index]
            new_dict[new_index].idx = new_index
            old_index_to_new_index[old_index] = new_index
            # Increment the new index
            new_index += 1
        
        self.idx_to_mig_hist_node = new_dict
        return old_index_to_new_index

# Defines a unique adjacency matrix and vertex labeling
class MigrationHistory:
    def __init__(self, tree, labeling):
        if (tree.shape[0] != tree.shape[1]):
            raise ValueError("Adjacency matrix should have shape (num_nodes x num_nodes)")
        if (tree.shape[0] != labeling.shape[1]):
            raise ValueError("Vertex labeling matrix should have shape (num_sites x num_nodes)")

        self.tree = tree
        self.labeling = labeling

    def _nonzero_tuple(self, tensor):
        # Get the indices of non-zero elements and convert to a hashable form
        if tensor.is_sparse:
            tensor = tensor.coalesce()
            indices = tensor.indices()
        else:
            indices = torch.nonzero(tensor, as_tuple=False)
        return tuple(map(tuple, indices.tolist()))

    def __hash__(self):
        # Compute a hash based on the positions of non-zero entries
        return hash((self._nonzero_tuple(self.labeling), self._nonzero_tuple(self.tree)))

    def __eq__(self, other):
        # Check for equality based on the positions of non-zero entries in both tensors
        if not isinstance(other, MigrationHistory):
            return False
        
        # Check labeling equality
        if self._nonzero_tuple(self.labeling) != self._nonzero_tuple(other.labeling):
            return False
            
        # Handle case where one tree is sparse and other is dense
        if self.tree.is_sparse and not other.tree.is_sparse:
            # Convert sparse to dense indices for comparison
            sparse_indices = self.tree.coalesce().indices()
            dense_indices = torch.nonzero(other.tree, as_tuple=False).t()
            return torch.equal(sparse_indices, dense_indices)
        elif not self.tree.is_sparse and other.tree.is_sparse:
            # Convert sparse to dense indices for comparison
            sparse_indices = other.tree.coalesce().indices() 
            dense_indices = torch.nonzero(self.tree, as_tuple=False).t()
            return torch.equal(sparse_indices, dense_indices)
        else:
            # Both same type, use original comparison
            return self._nonzero_tuple(self.tree) == self._nonzero_tuple(other.tree)

    def __str__(self):
        A = str(torch.where(self.tree != 0))
        V = str(torch.where(self.labeling == 1))
        return f"Tree: {A}\nVertex Labeling: {V}"

def convert_metric_to_int(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return int(x.item())  
    raise ValueError(f"Got unexpected value for metric {x}")

def convert_metric_to_float(x):
    if isinstance(x, float):
        return x
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return float(x.item())  
    raise ValueError(f"Got unexpected value for metric {x}")

# Convenience object to package information needed for a final solution
class VertexLabelingSolution:
    def __init__(self, loss, m, c, s, g, o, e, V, soft_V, T, G, node_collection):
        self.loss = loss
        self.m = convert_metric_to_int(m)
        self.c = convert_metric_to_int(c)
        self.s = convert_metric_to_int(s)
        self.g = convert_metric_to_float(g)
        self.o = convert_metric_to_float(o)
        self.e = convert_metric_to_float(e)
        self.V = V
        self.T = T
        self.G = G
        self.node_collection = node_collection
        self.soft_V = soft_V

    # Override the comparison operator
    def __lt__(self, other):
        return self.loss < other.loss
    
    def _nonzero_tuple(self, tensor):
        # Get the indices of non-zero elements and convert to a hashable form
        indices = torch.nonzero(tensor, as_tuple=False)
        return tuple(map(tuple, indices.tolist()))

    def __hash__(self):
        # Compute a hash based on the positions of non-zero entries
        return hash((self._nonzero_tuple(self.V), self._nonzero_tuple(self.T)))

    def __eq__(self, other):
        # Check for equality based on the positions of non-zero entries in both tensors
        if not isinstance(other, VertexLabelingSolution):
            return False
        return (self._nonzero_tuple(self.V) == self._nonzero_tuple(other.V) and
                self._nonzero_tuple(self.T) == self._nonzero_tuple(other.T))

# Convenience object to package information needed for known 
# labelings/indices of vertex labeling matrix V
class FixedVertexLabeling:
    def __init__(self, known_indices, unknown_indices, known_labelings, optimal_root_nodes, old_to_new):
        self.known_indices = known_indices
        self.unknown_indices = unknown_indices
        self.known_labelings = known_labelings
        self.optimal_root_nodes = optimal_root_nodes
        self.old_to_new = old_to_new

######################################################
############# CALCULATING PARSIMONY METRICS ##########
######################################################

def migration_number(site_adj):
    '''
    Args:
        - site_adj: sample_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j
    Returns:
        - migration number: number of total migrations between sites (no same site to same site migrations)
    '''
    site_adj_trace = torch.diagonal(site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    m = torch.sum(site_adj, dim=(1, 2)) - site_adj_trace
    return m

def seeding_site_number(site_adj_no_diag):
    '''
    Args:
        - site_adj_no_diag: sample_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j, and no same site migrations are included
    Returns:
        - seeding site number: number of sites that have outgoing edges
    '''
    row_sums_site_adj = torch.sum(site_adj_no_diag, axis=2)
    # can only have a max of 1 for each site (it's either a seeding site or it's not)
    binarized_row_sums_site_adj = torch.sigmoid(BINARY_ALPHA * (2*row_sums_site_adj - 1)) # sigmoid for soft thresholding
    s = torch.sum(binarized_row_sums_site_adj, dim=(1))
    return s

def comigration_number_approximation(site_adj):
    '''
    Args:
        - site_adj: sample_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j
    Returns:
        - comigration number: an approximation of the comigration number that doesn't include a penalty for
        repeated temporal migrations which is expensive to compute
    '''
    binarized_site_adj = torch.sigmoid(BINARY_ALPHA * (2 * site_adj - 1))
    bin_site_trace = torch.diagonal(binarized_site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    c = torch.sum(binarized_site_adj, dim=(1,2)) - bin_site_trace
    return c

def comigration_number(site_adj, A, VA, V, VT, update_path_matrix, identical_T):
    '''
    Handles the case where the adjacency matrix is sparse, in which case we can use slower per-batch operations
    to compute the comigration number.

    Args:
        - site_adj: sample_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j
        - A: Adjacency matrix (directed) of the full tree (sample_size x num_nodes x num_nodes)
        - VA: V*A (will be converted to sparse)
        - V: Vertex labeling one-hot matrix (sample_size x num_sites x num_nodes)
        - VT: transpose of V 
        - update_path_matrix: whether we need to update the path matrix or we can use a cached version
        (need to update when we're actively resolving polytomies)
        - identical_T: bool, whether all adjacency matrices in T along the sample size dimension are identical
    Returns:
        - comigration number: a subset of the migration edges between two anatomical sites, such that 
        the migration edges occur on distinct branches of the clone tree
    '''

    # First calculate base comigration number (without temporal repeats)
    base_c = comigration_number_approximation(site_adj)
    
    # Get path matrix if needed
    global LAST_P
    if LAST_P is not None and not update_path_matrix:
        P = LAST_P.to(A.device)
        if P.shape != A.shape:
            assert(A.shape[0]==1)
            P = P[0]
    else:
        P = path_matrix(A, remove_self_loops=True, identical_T=identical_T)
        LAST_P = P
    
    # Node colors
    node_colors = torch.argmax(V, dim=1)  # [batch, num_nodes]
    parent_colors = torch.argmax(VA.transpose(1, 2), dim=2)

    # Nodes that differ from their parent
    diff_nodes = torch.where(node_colors != parent_colors)
    temporal_migrations = torch.zeros(node_colors.shape[0], device=V.device)

    for batch in range(node_colors.shape[0]):
        batch_diff_nodes = diff_nodes[1][diff_nodes[0] == batch]
        root_node = get_root_index(A[batch])
        batch_diff_nodes = batch_diff_nodes[batch_diff_nodes != root_node]

        if len(batch_diff_nodes) == 0:
            continue
        
        diff_node_colors = node_colors[batch, batch_diff_nodes]
        diff_parent_colors = parent_colors[batch, batch_diff_nodes]
        combined_colors = diff_node_colors * V.shape[1] + diff_parent_colors
        unique_combos = torch.unique(combined_colors)

        # Use coalesced sparse indices if sparse, otherwise convert to dense for indexing
        if A.is_sparse:
            P_batch = P[batch].coalesce()
            indices = P_batch.indices()  # [2, nnz]
        else:
            P_batch = P[batch]

        for combo in unique_combos:
            # Nodes with same self color + parent color
            nodes_with_combo = batch_diff_nodes[combined_colors == combo]
            if len(nodes_with_combo) <= 1:
                continue

            counted_parents = set()
            nodes_set = set(int(n) for n in nodes_with_combo)

            # Count the number of nodes with same self color + parent color, that can reach other nodes
            # with the same combo via the path matrix. 
            for node in nodes_with_combo:
                if A.is_sparse:
                    # Sparse-safe reachability check
                    mask = (indices[0] == node) & torch.tensor(
                        [int(j) in nodes_set and j != node for j in indices[1]], device=indices.device
                    )
                    if mask.any():
                        counted_parents.add(int(node))
                else:
                    # Dense reachability
                    reachable = P_batch[node, nodes_with_combo]
                    if reachable.sum() > 0:
                        counted_parents.add(int(node))

            temporal_migrations[batch] += len(counted_parents)

    return base_c + temporal_migrations

def genetic_distance_score(G, m, A, V, VT):
    '''
    Args:
        - G: Matrix of genetic distances between internal nodes (shape: sample_size x num_internal_nodes x num_internal_nodes).
             Lower values indicate lower branch lengths, i.e. more genetically similar.
        - m: vector of migration numbers (length = sample_size)
        - A: Adjacency matrix (directed) of the full tree (sample_size x num_nodes x num_nodes)
        - V: Vertex labeling one-hot matrix (sample_size x num_sites x num_nodes)
        - VT: transpose of V 
    Returns:
        - genetic distance score, for each sample in the first dimension
    '''
    g = torch.zeros(m.shape, device=A.device)
    if G != None:
        X = VT @ V # 1 if two nodes are the same color
        # TODO: is there a way to do this without converting A to a dense matrix?
        A = A.to_dense()
        # Calculate if 2 nodes are in diff sites and there's an edge between them (i.e. there is a migration edge)
        R = torch.mul(A, (1-X))
        adjusted_G = -torch.log(G+0.01)
        R = torch.mul(R, adjusted_G)
        g = torch.sum(R, dim=(1,2))/(m + 1) # to prevent division by 0
    return g

def organotropism_score(O, site_adj_no_diag, p, bs):
    '''
    Args:
        - O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        - site_adj_no_diag: sample_size x num_sites x num_sites matrix, where each num_sites x num_sites
        matrix has the number of migrations from site i to site j, and no self loops
        - p: one-hot vector indicating site of the primary
        - bs: sample_size (number of samples)
    Returns:
        - organotropism score, for each sample in the first dimension
    '''
    o = torch.zeros(bs, device=site_adj_no_diag.device)
    if O != None:
        O = O.to(site_adj_no_diag.device)
        # the organotropism frequencies can only be used on the first 
        # row, which is for the migrations from primary cancer site to
        # other metastatic sites (we don't have frequencies for every 
        # site to site migration)
        prim_site_idx = torch.nonzero(p)[0][0]
        O = O.repeat(bs,1).reshape(bs, O.shape[0])
        adjusted_freqs = -torch.log(O+0.01)
        num_mig_from_prim = site_adj_no_diag[:,prim_site_idx,:]
        organ_penalty = torch.mul(num_mig_from_prim, adjusted_freqs)
        o = torch.sum(organ_penalty, dim=(1))/(torch.sum(num_mig_from_prim, dim=(1)) + 1) # to prevent division by 0
    return o

def ancestral_labeling_metrics(V, A, G, O, p, update_path_matrix, compute_full_c, identical_T):
    
    single_A = A
    bs = V.shape[0]
    num_sites = V.shape[1]
    A = A if len(A.shape) == 3 else repeat_n(single_A, bs) 
    
    # Compute matrices used for all parsimony metrics
    VA = torch.bmm(A.permute(0,2,1), V.permute(0,2,1)).permute(0,2,1)
    VT = torch.transpose(V, 2, 1)
    site_adj = VA @ VT
    # Remove the same site transitions from the site adjacency matrix
    site_adj_no_diag = torch.mul(site_adj, repeat_n(1-torch.eye(num_sites, num_sites, device=site_adj.device), bs))

    # 1. Migration number
    m = migration_number(site_adj)

    # 2. Seeding site number
    s = seeding_site_number(site_adj_no_diag)

    # 3. Comigration number
    if compute_full_c:
        c = comigration_number(site_adj, A, VA, V, VT, update_path_matrix, identical_T)
    else:
        c = comigration_number_approximation(site_adj)

    # 4. Genetic distance
    g = genetic_distance_score(G, m, A, V, VT)

    # 5. Organotropism
    o = organotropism_score(O, site_adj_no_diag, p, bs)

    return m, c, s, g, o

def calc_entropy(V, soft_V):
    '''
    Args:
        - V: one-hot vertex labeling matrix
        - soft_V: underlying theta parameters of V
    Returns:
        - entropy for the categorical variable representing each node's vertex label
    '''
    eps = 1e-7 # to avoid nans when values in soft_V get very close to 0
    return torch.sum(torch.mul(soft_V, torch.log2(soft_V+eps)), dim=(1, 2))

def get_repeating_weight_vector(bs, weight_list, device):
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
    weights_vec = torch.tensor(repeated_list, device=device)

    return weights_vec

def get_mig_weight_vector(bs, weights, device):
    return get_repeating_weight_vector(bs, weights.mig, device)

def get_seed_site_weight_vector(bs, weights, device):
    return get_repeating_weight_vector(bs, weights.seed_site, device)

def clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=1):

    # Combine all 5 components with their weights
    # Explore different weightings
    if isinstance(weights.mig, list) and isinstance(weights.seed_site, list):
        mig_weights_vec = get_mig_weight_vector(bs, weights, m.device)
        seeding_sites_weights_vec = get_seed_site_weight_vector(bs, weights, s.device)
        mig_loss = torch.mul(mig_weights_vec, m)
        seeding_loss = torch.mul(seeding_sites_weights_vec, s)
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
    else:
        mig_loss = weights.mig*m
        seeding_loss = weights.seed_site*s
        labeling_loss = (mig_loss + weights.comig*c + seeding_loss + weights.gen_dist*g + weights.organotrop*o+ weights.entropy*e)
    return labeling_loss

def clone_tree_labeling_objective(V, soft_V, T, G, O, p, weights, 
                                  update_path_matrix=True, compute_full_c=True, identical_T=False):
    '''
    Args:
        V: Vertex labeling of the full tree (sample_size x num_sites x num_nodes)
        T: Adjacency matrix (directed) of the full tree (sample_size x num_nodes x num_nodes)
        G: Matrix of genetic distances between internal nodes (shape: sample_size x num_internal_nodes x num_internal_nodes).
        Lower values indicate lower branch lengths, i.e. more genetically similar.
        O: Array of frequencies with which the primary cancer type seeds site i (shape: num_anatomical_sites).
        p: one-hot vector indicating site of the primary
        weights: Weights object
        update_path_matrix: bool, whether we need to update the path matrix (1:1 with T) or we can use the last
            cached version (this is expensive to compute)
        compute_full_c: bool, whether we need to compute the full comigration number or we can calculate an approzimation
            (this is expensive to compute)
        identical_T: bool, whether all adjacency matrices in T along the sample size dimension are identical

    Returns:
        Loss to score the ancestral vertex labeling of the given tree. This combines (1) migration number, (2) seeding site
        number, (3) comigration number, and optionally (4) genetic distance and (5) organotropism.
    '''
    V = add_batch_dim(V)
    soft_V = add_batch_dim(soft_V)
    m, c, s, g, o = ancestral_labeling_metrics(V, T, G, O, p, update_path_matrix, compute_full_c, identical_T)
    # Entropy
    e = calc_entropy(V, soft_V)

    labeling_loss = clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights, bs=V.shape[0])
    return labeling_loss, (m, c, s, g, o, e)

def stack_fixed_labeling(X, fixed_labeling, p):
    """
    Helper function to stack vertex labeling with fixed labelings.
    Handles both cases where old_to_new mapping is present or None.
    
    Args:
        X: Tensor of shape (bs, num_sites, num_unknown_nodes) containing unknown labelings
        fixed_labeling: FixedVertexLabeling object containing known indices and labelings
        p: Primary site labeling tensor
    
    Returns:
        full_X: Tensor containing both known and unknown labelings properly mapped
    """
    bs = X.shape[0]
    # Initialize with space for p + known/unknown nodes
    
    known_labelings = repeat_n(fixed_labeling.known_labelings, bs)
    p = repeat_n(p, bs).squeeze(-1)
    
    if fixed_labeling.old_to_new is None:
        full_X = torch.zeros((bs, X.shape[1], 1 + len(fixed_labeling.known_indices)+len(fixed_labeling.unknown_indices)))
        # Set primary site labeling first
        full_X[:, :, 0] = p
        # Simple case: direct mapping of indices
        full_X[:, :, fixed_labeling.unknown_indices] = X
        full_X[:, :, fixed_labeling.known_indices] = known_labelings
    else:
        # Complex case: use old_to_new mapping
        full_X = torch.zeros((bs, X.shape[1], 1 + len(fixed_labeling.optimal_root_nodes) + len(fixed_labeling.unknown_indices)))
        # Set primary site labeling first
        full_X[:, :, 0] = p
        root_indices = [idx for idx in fixed_labeling.known_indices if idx in fixed_labeling.optimal_root_nodes]
        root_src_idx = torch.tensor([fixed_labeling.known_indices.index(idx) for idx in root_indices])
        root_dst_idx = torch.tensor([fixed_labeling.old_to_new.get(idx, idx) for idx in root_indices])
        
        unknown_indices = fixed_labeling.unknown_indices 
        unknown_src_idx = torch.arange(len(unknown_indices))
        unknown_dst_idx = torch.tensor([fixed_labeling.old_to_new.get(idx, idx) for idx in unknown_indices], dtype=torch.long)
        
        full_X[:, :, root_dst_idx] = fixed_labeling.known_labelings[:, root_src_idx].unsqueeze(0).expand(X.shape[0], -1, -1)
        full_X[:, :, unknown_dst_idx] = X[:, :, unknown_src_idx]
    
    return full_X

def stack_vertex_labeling(L, X, p, poly_res, fixed_labeling):
    '''
    Use leaf labeling L and X (both of size sample_size x num_sites X num_internal_nodes)
    to get the anatomical sites of the leaf nodes and the internal nodes (respectively). 
    Stack the root labeling to get the full vertex labeling V. 
    '''
    # Expand leaf node labeling L to be repeated sample_size times
    bs = X.shape[0]
    L = repeat_n(L, bs)

    if fixed_labeling != None:
        full_X = stack_fixed_labeling(X, fixed_labeling, p)
    else:
        # Include p at the start for the non-fixed labeling case
        p = repeat_n(p, bs)
        full_X = torch.cat((p, X), dim=2)

    # Concatenate with leaf labelings
    # Order is: internal nodes, new poly nodes (is resolving polytomies), leaf nodes from U
    full_vert_labeling = torch.cat((full_X, L), dim=2)
    return full_vert_labeling

######################################################
######### POST U MATRIX ESTIMATION UTILITIES #########
######################################################

def get_leaf_labels_from_U(U):
    
    U = U[:,1:] # don't include column for normal cells
    internal_node_idx_to_sites = {}
    for node_idx in range(U.shape[1]):
        for site_idx in range(U.shape[0]):
            node_U = U[site_idx,node_idx]
            if node_U > U_CUTOFF:
                if node_idx not in internal_node_idx_to_sites:
                    internal_node_idx_to_sites[node_idx] = []
                internal_node_idx_to_sites[node_idx].append(site_idx)
    
    return internal_node_idx_to_sites

def full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, idx_to_sites_present, 
                                                            num_sites, G_identical_clone_val, 
                                                            node_collection, ordered_sites):
    '''
    All non-zero values of U represent extant clones (witness nodes of the full tree).
    For each of these non-zero values, we add an edge from parent clone to extant clone.
    '''
    num_leaves = sum(len(lst) for lst in idx_to_sites_present.values())
    full_adj = torch.nn.functional.pad(input=input_T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
    witness_idx = input_T.shape[0]
    # Also add branch lengths (genetic distances) for the edges we're adding.
    # Since all of these edges we're adding represent genetically identical clones,
    # we are going to add a very small but non-zero value.
    full_G = torch.nn.functional.pad(input=input_G, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0) if input_G is not None else None

    leaf_labels = []
    # Iterate through the internal nodes that we want to attach witness nodes to
    for internal_node_idx in idx_to_sites_present:
        # Attach a leaf node for every site that this internal node is observed in
        for site in idx_to_sites_present[internal_node_idx]:
            full_adj[internal_node_idx, witness_idx] = 1
            label = node_collection.get_node(internal_node_idx).label + [ordered_sites[site]]
            leaf_node = MigrationHistoryNode(witness_idx, label, is_witness=True, is_polytomy_resolver_node=False)
            node_collection.add_node(leaf_node)
            if input_G is not None:
                full_G[internal_node_idx, witness_idx] = G_identical_clone_val
            witness_idx += 1
            leaf_labels.append(site)

    assert len(leaf_labels) !=0, f"No nodes present in any sites"

    # Anatomical site labels of the leaves
    L = torch.nn.functional.one_hot(torch.tensor(leaf_labels), num_classes=num_sites).T
    return full_adj, full_G, L, node_collection
    
def full_adj_matrix_using_inferred_observed_clones(U, input_T, input_G, num_sites, G_identical_clone_val,
                                                   node_collection, ordered_sites):
    '''
    Use inferred observed clones to fill out T and G by adding leaf nodes
    '''
    internal_node_idx_to_sites = get_leaf_labels_from_U(U)
    full_adj, full_G, L, node_collection = full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, internal_node_idx_to_sites, 
                                                                                                   num_sites, G_identical_clone_val,
                                                                                                   node_collection, ordered_sites)
    
    return full_adj, full_G, L, internal_node_idx_to_sites, node_collection

def remove_leaf_indices_not_observed_sites(removal_indices, U, input_T, T, G, node_collection, idx_to_observed_sites):
    '''
    Remove clone tree leaf nodes that are not detected in any sites. 
    These are not well estimated
    '''
    if len(removal_indices) == 0:
        return U, input_T, T, G, node_collection, idx_to_observed_sites
    
    for remove_idx in removal_indices:
        child_indices = get_child_indices(T, [remove_idx])
        assert len(child_indices) == 0

    # Remove indices from input_T, full T (now with observed clones), U and G
    T = np.delete(T, removal_indices, 0)
    T = np.delete(T, removal_indices, 1)

    # Remove indices from T, U and G
    input_T = np.delete(input_T, removal_indices, 0)
    input_T = np.delete(input_T, removal_indices, 1)

    U = np.delete(U, removal_indices, 1)

    if G != None: 
        G = np.delete(G, removal_indices, 0)
        G = np.delete(G, removal_indices, 1)

    # Reindex the idx to label dict
    old_index_to_new_index = node_collection.remove_indices_and_reindex(removal_indices)
    new_idx_to_observed_sites = {}
    for old_idx in idx_to_observed_sites:
        new_idx = old_index_to_new_index[old_idx]
        new_idx_to_observed_sites[new_idx] = idx_to_observed_sites[old_idx]

    return U, input_T, T, G, node_collection, new_idx_to_observed_sites

######################################################
################## RANDOM UTILITIES ##################
######################################################

def multiply_with_sparse_check(m1, m2):
    """
    Multiplies a dense matrix m1 with a sparse matrix m2
    
    If m2 is a sparse tensor, it uses sparse-dense multiplication.
    If m2 is a dense tensor, it uses standard dense multiplication.
    """
    if isinstance(m2, torch.sparse.SparseTensor):
        # If m2 is sparse, multiply using sparse-dense multiplication
        return torch.sparse.mm(m2.transpose(0, 1), V.T).T
    else:
        # If m2 is dense, multiply using standard multiplication
        return m1 @ m2
    
def print_gpu_memory():
    if torch.cuda.is_available():
        # Get current device index
        device = torch.cuda.current_device()

        # Print the name of the GPU
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        
        # Total memory
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # in GB
        
        # Reserved memory by tensors
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9  # in GB
        
        # Memory actually being used by tensors
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9  # in GB
        
        # Available free memory
        free_memory = reserved_memory - allocated_memory
        
        print(f"Total memory: {total_memory:.2f} GB")
        print(f"Reserved memory: {reserved_memory:.2f} GB")
        print(f"Allocated memory: {allocated_memory:.2f} GB")
        print(f"Free memory: {free_memory:.2f} GB\n")
    else:
        print("CUDA is not available")


def mutation_matrix_with_normal_cells(T):
    B = mutation_matrix(T)
    # Add a row of zeros to account for the non-cancerous root node
    B = torch.vstack([torch.zeros(B.shape[1]), B])
    # Add a column of ones to indicate that every clone is a descendent of the non-cancerous root node
    B = torch.hstack([torch.ones(B.shape[0]).reshape(-1,1), B])
    return B

def adjacency_matrix_to_edge_list(adj_matrix):
    edges = []
    for i, j in tree_iterator(adj_matrix):
        edges.append((i, j))
    return edges

def reindex_dict(original_dict, indices_to_remove):
    # Create a new dictionary to hold the re-indexed entries
    new_dict = {}
    
    old_index_to_new_index = {}
    # Initialize the new index
    new_index = 0
    # Iterate through the original dictionary in sorted index order
    for old_index in sorted(original_dict.keys()):
        # Skip the indices that need to be removed
        if old_index in indices_to_remove:
            continue
        
        # Assign the new index to the current label
        new_dict[new_index] = original_dict[old_index]
        old_index_to_new_index[old_index] = new_index
        # Increment the new index
        new_index += 1
    
    return new_dict, old_index_to_new_index

def add_batch_dim(x):
    if len(x.shape) == 3:
        return x
    return x.reshape(1, x.shape[0], x.shape[1])

def to_tensor(t):
    if torch.is_tensor(t):
        return t
    return torch.stack(t)

def parents_to_adj_matrix(parents):
    '''
    Convert a parents vector to a sparse adjacency matrix
    '''
    # Convert parents to tensor if it's a numpy array
    if isinstance(parents, np.ndarray):
        parents = torch.from_numpy(parents).to('cpu')
    # Get child nodes (all indices except root)
    child_nodes = torch.arange(len(parents), device='cpu')

    # Filter out the root node (-1 parent)
    mask = parents != -1
    parent_nodes = parents[mask]
    child_nodes = child_nodes[mask]

    # Create sparse adjacency matrix
    indices = torch.stack([parent_nodes, child_nodes])  # [Parents, Children]
    values = torch.ones(len(parent_nodes))  # Edge existence
    size = (len(parents), len(parents))  # Square adjacency matrix

    adj_matrix = torch.sparse_coo_tensor(indices, values, size, device='cpu')
    
    return adj_matrix

def create_reweighted_solution_set_from_pckl(pckl, O, p, weights):
    # Make a solution set from the pickled files
    Ts, Vs, soft_Vs, Gs = pckl[OUT_PARENTS_KEY], pckl[OUT_LABElING_KEY], pckl[OUT_SOFTV_KEY], pckl[OUT_GEN_DIST_KEY]
    node_collections = [MigrationHistoryNodeCollection.from_dict(dct) for dct in pckl[OUT_IDX_LABEL_KEY]]
    solution_set = []
    for T, V, soft_V, G, node_collection in zip(Ts, Vs, soft_Vs, Gs, node_collections):
        T = parents_to_adj_matrix(T)
        V,soft_V,G = torch.tensor(V, device='cpu'),torch.tensor(soft_V, device='cpu'),torch.tensor(G, device='cpu')
        loss, new_metrics = clone_tree_labeling_objective(V,soft_V, T, G, O, p, weights, True)
        solution_set.append(VertexLabelingSolution(loss, *new_metrics,V,soft_V, T, G,node_collection))
    return solution_set

def calculate_sample_size(num_nodes, num_sites, solve_polytomies):
    '''
    Calculate the number of samples to initialize for a run based on 
    the number of tree nodes, the number of anatomical sites, and if we're
    solving polytomies
    '''
    min_size = 1024
    min_size += num_nodes*num_sites*4

    if solve_polytomies:
        min_size *= 2

    # cap this to a reasonably high sample size
    min_size = min(min_size, 60000)
    print(f"Input tree has {num_nodes} nodes, calculated sample size: {min_size}", )
    return min_size

def tree_iterator(T):
    ''' 
    Iterate an adjacency matrix, yielding i and j for all values = 1.
    Handles dense NumPy arrays, dense PyTorch tensors, and sparse PyTorch tensors.
    '''
    # Case 1: If input is a sparse PyTorch tensor
    if isinstance(T, torch.Tensor) and T.is_sparse:
        non_zero_indices = T.coalesce().indices()  # Get non-zero indices from sparse tensor
        
        # Yield i, j for all non-zero values (equal to 1 in adjacency matrix)
        for i in range(non_zero_indices.size(1)):
            row, col = non_zero_indices[:, i]
            yield int(row), int(col)

    # Case 2: If input is a dense PyTorch tensor
    elif isinstance(T, torch.Tensor):
        T = T.detach().cpu()  # Ensure it's on the CPU
        non_zero_indices = torch.nonzero(T != 0, as_tuple=False)  # Get indices where value == 1

        # Yield i, j for all non-zero values
        for row, col in non_zero_indices:
            yield int(row), int(col)

    # Case 3: If input is a NumPy array
    elif isinstance(T, np.ndarray):
        # Convert NumPy array to a SciPy sparse matrix for efficiency
        T_sparse = sp.coo_matrix(T)  # Convert to sparse COO format

        # Yield i, j for all non-zero values
        for row, col in zip(T_sparse.row, T_sparse.col):
            yield int(row), int(col)

    else:
        raise TypeError("Input must be a sparse PyTorch tensor, dense PyTorch tensor, or NumPy array.")


def list_gpu_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:  # Check if the tensor is on the GPU
                    print(f"Tensor on GPU - shape: {obj.shape}, dtype: {obj.dtype}")
        except Exception as e:
            pass

def bfs_iterator(tree, start_node):
    '''
    Iterate an adjacency matrix in breadth first search order
    '''
    queue = deque([start_node])  # Start the queue with the given start node index
    visited = set(queue)         # Mark the start node as visited

    # BFS loop that yields nodes in the order they are visited
    while queue:
        current = queue.popleft()
        yield current  # Yield the current node

        # Check each adjacency in the row for the current node
        for neighbor, is_connected in enumerate(tree[current]):
            if is_connected and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def has_leaf_node(adjacency_matrix, start_node, num_internal_nodes):
    num_nodes = adjacency_matrix.size(0)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    queue = Queue()

    queue.put(start_node)
    visited[start_node] = True

    while not queue.empty():
        current_node = int(queue.get())
        # Get neighbors of the current node
        neighbors = torch.nonzero(adjacency_matrix[current_node]).squeeze(1)

        # Enqueue unvisited neighbors
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if not visited[neighbor]:
                queue.put(neighbor)
                visited[neighbor] = True
                # it's a leaf node if its not in the clone tree idx label dict
                if neighbor > num_internal_nodes:
                    return True
    return False

def get_root_index(T):
    '''
    Returns the root idx (node with no inbound edges) from adjacency matrix T
    '''
    # Get column indices of all edges
    if T.is_sparse:
        T = T.coalesce()
        dest_nodes = set(T.indices()[1].tolist())
    else:
        dest_nodes = set(torch.nonzero(T, as_tuple=True)[1].tolist())

    # Root is the only node that is not a destination
    all_nodes = set(range(T.shape[0]))
    candidates = all_nodes - dest_nodes

    msg = "More than one" if len(candidates) > 1 else "No"
    assert (len(candidates) == 1), f"{msg} root node detected"

    return list(candidates)[0]


def reverse_bfs_order(A):
    '''
    Returns nodes in reverse bfs order of adjacency matrix A
    '''
    root_idx = get_root_index(A)
    nodes = []
    for x in bfs_iterator(A, root_idx):
        nodes.append(x)
    nodes.reverse()
    return nodes

def get_leaves(A):
    '''
    Returns leaves of adjacency matrix A
    '''
    return [int(x) for x in torch.where(A.sum(dim=1)==0)[0]]

def swap_keys(d, key1, key2):
    if key1 in d and key2 in d:
        # Both keys exist, swap their values
        d[key1], d[key2] = d[key2], d[key1]
    elif key1 in d and key2 not in d:
        # Only key1 exists, move its value to key2
        d[key2] = d.pop(key1)
    elif key2 in d and key1 not in d:
        # Only key2 exists, move its value to key1
        d[key1] = d.pop(key2)
    # If neither key exists, do nothing
    return d

def find_leaf_nodes(sparse_tensor):
    """
    Given a sparse COO tensor representing an adjacency matrix, find the leaf nodes.
    
    Parameters:
    - sparse_tensor: Sparse COO tensor representing the adjacency matrix (n x n).
    
    Returns:
    - leaf_nodes: A list of indices of leaf nodes (nodes with no outgoing edges).
    """
    # Get the "from" nodes (row indices) of the sparse matrix
    from_nodes = sparse_tensor.coalesce().indices()[0]

    # Identify all nodes in the graph (based on the number of rows/columns in the matrix)
    all_nodes = torch.arange(sparse_tensor.shape[0])

    # Leaf nodes are those that are not present in the "from" node list
    leaf_nodes = all_nodes[~torch.isin(all_nodes, from_nodes)].tolist()

    return leaf_nodes

def restructure_matrices_root_index_zero(adj_matrix, ref_matrix, var_matrix, node_collection, gen_dist_matrix, idx_to_observed_sites):
    '''
    Restructure the inputs so that the node at index 0 becomes the root node.
    '''
    og_root_idx = get_root_index(adj_matrix)
    return restructure_matrices(og_root_idx, 0, adj_matrix, ref_matrix, var_matrix, 
                                node_collection, gen_dist_matrix, idx_to_observed_sites, None, None)

def restructure_matrices(source_root_idx, target_root_idx, adj_matrix, ref_matrix, var_matrix, 
                         node_collection, gen_dist_matrix, idx_to_observed_sites, V, U):
    '''
    Restructure the inputs so that the order of nodes has source_root_idx and target_root_idx swapped

    Returns:
        - Restructured adjacency matrix, reference matrix, variant matrix, node_idt_to_label dcitionary,
        and genetic distance matrix
    '''
    if source_root_idx == target_root_idx:
        # Nothing to restructure here!
        return adj_matrix, ref_matrix, var_matrix, node_collection, gen_dist_matrix, idx_to_observed_sites, V, U
    
    new_order = [x for x in range(len(adj_matrix))]
    new_order[source_root_idx] = target_root_idx
    new_order[target_root_idx] = source_root_idx
    # Swap rows and columns to move the first row and column to the desired position
    swapped_adjacency_matrix = adj_matrix[new_order, :][:, new_order]

    # Swap columns 
    swapped_ref_matrix = ref_matrix[:, new_order] if ref_matrix != None else None
    swapped_var_matrix = var_matrix[:, new_order] if var_matrix != None else None
    swapped_V = V[:, new_order] if V != None else None
    
    if U == None:
        swapped_U = None
    else:
        normal_cells = U[:,0].view(-1, 1)
        U_order = [x+1 for x in new_order if x < U.shape[1]-1]
        swapped_U = U[:, U_order]
        swapped_U = torch.cat((normal_cells, swapped_U), dim=1)
        
    if gen_dist_matrix == None:
        swapped_gen_dist_matrix = None
    else:
        swapped_gen_dist_matrix = gen_dist_matrix[new_order, :][:, new_order]
    
    if idx_to_observed_sites == None:
        idx_to_observed_sites = None
    else:
        idx_to_observed_sites = swap_keys(idx_to_observed_sites, source_root_idx, target_root_idx)

    node_collection.swap_indices(source_root_idx, target_root_idx)
    
    return swapped_adjacency_matrix, swapped_ref_matrix, swapped_var_matrix, node_collection, swapped_gen_dist_matrix, idx_to_observed_sites, swapped_V, swapped_U

def nodes_w_leaf_nodes(adj_matrices, num_internal_nodes):
    '''
    Args:
        - adj_matrices: 3d matrix, where each inner matrix is a 2d adjacency matric
        - num_internal_nodes: number of nodes in the clone tree which are not leaf nodes
          indicating clone presences

    Returns:
        A list of lists, where list i is for adjacency matrix i, and the inner list is a list
        of booleans indicated whether node j has a leaf clone presence node or not
    '''
    # Convert the 3D adjacency matrix to a PyTorch tensor
    adj_tensor = torch.tensor(adj_matrices, dtype=torch.float32)
    
    # Get dimensions
    _, n, _ = adj_tensor.size()

    # Step 2: Create a mask for columns with indices >= num_internal_nodes
    col_indices = torch.arange(n)
    col_mask = col_indices >= num_internal_nodes

    # Step 3: Apply the column mask to the 3D adjacency matrix
    # Broadcast the col_mask to the shape of the last dimension of adj_tensor
    filtered_cols = adj_tensor[:, :, col_mask]

    # Step 4: Check for each row in each 2D matrix if there's at least one '1' in the filtered columns
    mask = filtered_cols.any(dim=2)

    return mask

def top_k_integers_by_count(lst, k, min_num_sites, cutoff):
    # find unique sites and their counts
    unique_sites, counts = np.unique(lst, return_counts=True)
    # filter for sites that occur at least min_num_sites times
    filtered_values = unique_sites[counts >= min_num_sites]
    # recount after filtering
    unique_sites, counts = np.unique(filtered_values, return_counts=True)
    # sort based on counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    # get the top k integers and their counts
    if len(unique_sites) > k and cutoff:
        return list(unique_sites[sorted_indices[:k]])
        
    return list(unique_sites)

def traverse_until_Uleaf_or_mult_children(node, input_T, internal_node_idx_to_sites):
    """
    Traverse down the tree on a linear branch until a leaf node is found, and then
    return that node
    
    Special case where a node isn't detected in any sites and is on a long linear branch,
    so there are no children or grandchildren observed to help bias towards
    """
    if node in internal_node_idx_to_sites:
        return node
    
    children = input_T[node].nonzero()[:,0].tolist()
    
    if len(children) == 1:
        return traverse_until_Uleaf_or_mult_children(children[0], input_T, internal_node_idx_to_sites)
    
    return None  # no node has a leaf or node has more than one child
   
def get_resolver_sites_of_node_and_children(input_T, node_idx_to_sites, num_children, node_idx, include_children, min_num_sites, cutoff):
    '''
    Looking at the sites that node_idx and node_idx's children are in,
    figure out the sites that the polytomy resolver nodes should be initialized with
    (any sites that are detected >= thres times)
    '''
   
    clone_tree_children = input_T[node_idx].nonzero()[:,0].tolist()
    # calculated based on clone tree children and leaf nodes estimated form U
    num_possible_resolvers = math.floor(num_children/2)
    # get the sites that the node and node's children are in
    if include_children:
        node_and_children = clone_tree_children+ [int(node_idx)]
    else:
        node_and_children = [int(node_idx)]
    sites = []
    for key in node_and_children:
        if key in node_idx_to_sites:
            sites.extend(node_idx_to_sites[key])
    # print("node_idx", node_idx, "node_and_children", node_and_children, "sites", sites)
    # Special case where the node of interest (node_idx) isn't detected in any sites
    # and is on a linear branch 
    if not include_children and len(sites) == 0 and len(clone_tree_children) == 1:
        leaf_on_linear_branch = traverse_until_Uleaf_or_mult_children(node_idx, input_T, node_idx_to_sites)
        # print("leaf_on_linear_branch", leaf_on_linear_branch)
        if leaf_on_linear_branch != None:
            sites = node_idx_to_sites[leaf_on_linear_branch]
    
    return top_k_integers_by_count(sites, num_possible_resolvers, min_num_sites, cutoff)
    
def get_k_or_more_children_nodes(input_T, T, internal_node_idx_to_sites, k, include_children, min_num_sites, cutoff=True):
    '''
    returns the indices and proposed labeling for nodes
    that are under nodes with k or more children
    e.g. 
    input_T =[[0,1,1,],
             [0,0,0],
             [0,0,0],]
    T = [[0,1,1,1,0,0,0],
         [0,0,0,0,1,0,1],
         [0,0,0,0,0,1,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],]
    internal_node_idx_to_sites = {0:[1], 1:[0,1], 2:[1}
    k = 3
    include_children = True
    min_num_sites = 1
    returns ([0], [[1]]) (node at index 0 has 3 children, and the resolver node under it should be
    place in site 1, since node 0 and its child node 1 are both detected in site 1)

    if include_children is False, we only look at node 0's U values and not its children's U values
    '''
    
    T_dense = T.to_dense() if T.is_sparse else T
    row_sums = torch.sum(T_dense, axis=1)
    node_indices = list(torch.where(row_sums >= k)[0])
    del T_dense
    filtered_node_indices = []
    all_resolver_sites = []
    for node_idx in node_indices:
        # get the sites that node_idx and node_idx's children are estimated in U
        resolver_sites = get_resolver_sites_of_node_and_children(input_T, internal_node_idx_to_sites, row_sums[node_idx], node_idx, include_children, min_num_sites, cutoff)
        # print("resolver_sites", resolver_sites)
        # a resolver node wouldn't help for this polytomy
        if len(resolver_sites) == 0:
            continue
        filtered_node_indices.append(int(node_idx))
        all_resolver_sites.append(resolver_sites)
    return filtered_node_indices, all_resolver_sites

def find_first_branching_point(adj_matrix):
    out_degrees = adj_matrix.sum(dim=1)  # Calculate out-degrees of each node
    for node, out_degree in enumerate(out_degrees):
        if out_degree > 1:
            return node
    return None  # Return None if no branching point is found

import torch
import networkx as nx
from itertools import product

# Helper function to build a NetworkX tree from an adjacency matrix
def build_tree(adj_matrix):
    G = nx.DiGraph()
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

# Helper function to find the first common branching point in two graphs
def find_first_branching_point(adj_matrix):
    out_degrees = adj_matrix.sum(dim=1)  # Calculate out-degrees of each node
    for node, out_degree in enumerate(out_degrees):
        if out_degree > 1:
            return node
    return None  # Return None if no branching point is found

# Helper function to extract subtrees rooted at the given node
def extract_subtrees(G, root):
    subtrees = []
    for child in G.successors(root):
        subtree = nx.dfs_tree(G, source=child)
        subtrees.append(subtree)
    return subtrees

# Helper function to combine subtrees and labels from multiple solutions
def combine_subtrees(subtrees_list, labels_list, common_root):
    combinations = []
    num_subtrees = len(subtrees_list[0])
    
    # Generate all combinations of subtrees from each solution
    for indices in product(range(num_subtrees), repeat=len(subtrees_list)):
        combined_adj_matrix = torch.zeros_like(subtrees_list[0][0].adj_matrix)
        combined_labels = torch.zeros_like(labels_list[0])
        
        for i, idx in enumerate(indices):
            subtree = subtrees_list[i][idx]
            labels = labels_list[i]
            # Combine adjacency matrices
            for edge in subtree.edges:
                combined_adj_matrix[edge[0], edge[1]] = 1
            # Combine labels
            for node in subtree.nodes:
                combined_labels[:, node] = labels[:, node]
        
        combinations.append((combined_adj_matrix, combined_labels))
    return combinations

def sparse_tensors_equal(A: torch.Tensor, B: torch.Tensor) -> bool:
    """
    Compare two sparse tensors for equality.

    Args:
        A (torch.Tensor): First sparse tensor to compare
        B (torch.Tensor): Second sparse tensor to compare

    Returns:
        bool: True if tensors have identical indices and values, False otherwise
    """
    if A.shape != B.shape:
        return False

    A = A.coalesce()
    B = B.coalesce()

    return (torch.equal(A.indices(), B.indices()) and
            torch.equal(A.values(), B.values()))

def get_child_indices_sparse_t(T, indices):
    """Get child indices from a sparse tensor adjacency matrix.
    
    Args:
        T: Sparse tensor adjacency matrix
        indices: Parent node indices to find children for
        
    Returns:
        List of child node indices
    """
    # Convert indices to tensor if needed
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, device=T.device)
    elif len(indices.shape) == 0:
        indices = indices.unsqueeze(0)
        
    # Get coalesced indices and create index mask in one step
    T_indices = T.coalesce().indices()
    mask = torch.isin(T_indices[0], indices)
    
    # Return child indices directly from masked tensor
    return T_indices[1, mask].tolist()

def get_child_indices(T, indices):
    '''
    Args:
        T: Sparse or dense adjacency matrix
        indices: int or list of ints
    Returns:
        list of ints: direct children of the nodes at indices
    '''
    if T.is_sparse:
        return get_child_indices_sparse_t(T, indices)

    all_child_indices = []

    if not isinstance(indices, list):
        indices = [indices]

    for parent_idx in indices:
        children = torch.where(T[parent_idx,:] > 0)[0]
        for child_idx in children:
            if child_idx not in all_child_indices:
                all_child_indices.append(int(child_idx))

    return all_child_indices

def get_parent_idx_sparse_t(T, child_idx):
    T = T.coalesce()
    indices = T.indices()
    # Find where column index matches child_idx
    mask = indices[1] == child_idx
    # Get corresponding row index (parent)
    parent = indices[0][mask]
    
    if len(parent) == 0:
        raise ValueError("Parent index not found")
    return int(parent[0])

def get_parent(T, node):
    if T.is_sparse:
        return get_parent_idx_sparse_t(T, node)

    num_nodes = len(T)
    parents = [i for i in range(num_nodes) if T[i][node] != 0]
    assert len(parents) == 1
    return parents[0]

def find_parents_children(T, node):
    num_nodes = len(T)
    
    # Find parents: Look in the node's column
    parents = [i for i in range(num_nodes) if T[i][node] != 0]
    
    # Find children: Look in the node's row
    children = [j for j in range(num_nodes) if T[node][j] != 0]
    
    return parents, children

def repeat_n(x, n):
    '''
    Repeats tensor x 'n' times along the first axis, returning a tensor
    w/ dim (n, x.shape[0], x.shape[1])
    '''
    if n == 0:
        return x
    if x.is_sparse:
        return repeat_sparse(x, n)
    return x.repeat(n,1).reshape(n, x.shape[0], x.shape[1])

def repeat_sparse(x, n):
    if not x.is_coalesced():
        x = x.coalesce()
        
    # x: the input sparse tensor
    indices = x.indices()  # Get the indices of the sparse tensor
    values = x.values()    # Get the values of the sparse tensor
    size = x.size()        # Get the size of the sparse tensor

    # Create batch indices for all repetitions (n)
    batch_indices = torch.arange(n).repeat_interleave(indices.shape[1]).to(x.device)
    
    # Expand original indices to match the batch size (adding a new dimension)
    expanded_indices = indices.unsqueeze(0).repeat(n, 1, 1)  # Shape: (n, num_dims, num_non_zero)
    expanded_indices = expanded_indices.permute(1, 0, 2).reshape(expanded_indices.shape[1], expanded_indices.shape[0] * expanded_indices.shape[2])

    # Concatenate batch indices to expanded original indices
    final_indices = torch.cat([batch_indices.unsqueeze(0),expanded_indices], dim=0)
    # Repeat the values across the batch dimension
    repeated_values = values.repeat(n)

    # Define the new size, which includes the batch dimension
    new_size = torch.Size([n, *size])
    
    # Create a new sparse tensor with batch-wise repeated indices and values
    return torch.sparse_coo_tensor(final_indices, repeated_values, new_size)

def topological_sort_dag(T):
    """
    Perform topological sorting on the DAG using NetworkX.
    """
    # Convert adjacency matrix to NetworkX DiGraph
    G = nx.DiGraph(T.cpu().numpy())
    
    # Return the nodes in topological sorted order
    topo_sort = list(nx.topological_sort(G))
    topo_sort.reverse()
    return topo_sort

def purdom_transitive_closure(T, remove_self_loops):
    """
    Optimized version of Purdom's algorithm to compute transitive closure 
    focusing only on outgoing edges.
    
    T: Sparse adjacency matrix of the DAG.
    Returns: Sparse binary reachability matrix
    """
    T = T.to_dense().byte()
    
    # Initialize reachability matrix in-place
    reachability_matrix = T.clone()

    # Perform topological sort of the DAG
    topo_sort = topological_sort_dag(T)
    # Traverse the nodes in topological order
    for u in topo_sort:
        # Get the outgoing neighbors of node u
        neighbors_u = (T[u] != 0).nonzero(as_tuple=False)
        neighbors_u = [idx.item() for idx in neighbors_u]

        if neighbors_u:
            for v in neighbors_u:
                # Propagate reachability for the outgoing edges from u
                reachability_matrix[u] |= reachability_matrix[v]
    
    if not remove_self_loops:
        diag_ones = torch.eye(T.shape[0])
        # Add the diagonal ones to the original matrix
        reachability_matrix = reachability_matrix + diag_ones

    return reachability_matrix.to_sparse().coalesce()

def path_matrix_dense(T, remove_self_loops=False):
    '''
    T is a numpy ndarray or tensor adjacency matrix (where Tij = 1 if there is a path from i to j)
    remove_self_loops: bool, whether to retain 1s on the diagonal

    Returns path matrix that tells us if path exists from node i to node j    
    '''

    bs = 1 if len(T.shape) == 2 else T.shape[0]

    I = torch.eye(T.shape[1], device=T.device).repeat(bs, 1, 1)  # Repeat identity matrix along batch dimension

    B = torch.logical_or(T, I).int()  # Convert to int for more efficient matrix multiplication
    # Initialize path matrix with direct connections
    P = B.clone()
    
    # Floyd-Warshall algorithm
    for k in range(T.shape[1]):
        # Compute shortest paths including node k
        B = torch.logical_or(B, B[:, :, k].unsqueeze(2) & B[:, k, :].unsqueeze(1))
        P_old = torch.nonzero(P)
        # Update path matrix
        P |= B
        
    if remove_self_loops:
        P = torch.logical_xor(P, I.int())
    return P.squeeze(0) if len(T.shape) == 2 else P


def path_matrix(T, remove_self_loops=False, identical_T=False):
    '''
    Compute the transitive closure of adjacency matrix T using Purdom's algorithm
    '''
    if not T.is_sparse:
        return path_matrix_dense(T, remove_self_loops)

    # If there's a single 2D matrix, compute its closure directly
    if len(T.shape) == 2:
        return purdom_transitive_closure(T, remove_self_loops)
    
    # For batched matrices
    closures = []
    og_bs = T.shape[0]
    bs = 1 if identical_T else og_bs
    for i in range(bs):
        P = purdom_transitive_closure(T[i], remove_self_loops)
        closures.append(P)

    if identical_T:
        stacked_closure = repeat_n(closures[0], og_bs)
    else:
        stacked_closure = torch.stack(closures)
    return stacked_closure

def mutation_matrix(A):
    '''
    A is an numpy ndarray or tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return path_matrix(A.T, remove_self_loops=False)

def get_adj_matrix_from_edge_list(edge_list):
    T = []
    nodes = set([node for edge in edge_list for node in edge])
    T = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for edge in edge_list:
        T[ord(edge[0]) - 65][ord(edge[1]) - 65] = 1
    return torch.tensor(T, dtype = torch.float32)

def is_tree(adj_matrix):
    # rows, cols = torch.where(adj_matrix == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    edges = adjacency_matrix_to_edge_list(adj_matrix)
    g = nx.Graph()
    g.add_edges_from(edges)
    return (not nx.is_empty(g) and nx.is_tree(g))

def pareto_front(solutions, all_pars_metrics):
    """ 
    Args:
        - solutions: list of VertexLabelingSolutions of length n
        - all_pars_metrics: list of tuples with parsimony mentrics
        for (migration #, comigration #, seeding site #) of length n

    Returns:
        - a list of Pareto optimal parsimony metrics and VertexLabelingSolutions
    """
    
    # Start with an empty Pareto front
    pareto_front = []

    # Loop through each solution in the list
    for candidate_metric, soln in zip(all_pars_metrics, solutions):
        # Assume candidate is not dominated; check against all others
        if not any(all(other[0][i] <= candidate_metric[i] for i in range(len(candidate_metric))) and
                   any(other[0][i] < candidate_metric[i] for i in range(len(candidate_metric))) 
                   for other in pareto_front):
            # If no one in the current Pareto front dominates the candidate, add it
            pareto_front.append((candidate_metric, soln))
            # Remove any from the pareto_front that is dominated by the new candidate
            pareto_front = [front for front in pareto_front if not all(candidate_metric[i] <= front[0][i] for i in range(len(candidate_metric))) or not any(candidate_metric[i] < front[0][i] for i in range(len(candidate_metric)))]

    pareto_metrics = [front[0] for front in pareto_front]
    pareto_solutions = [front[1] for front in pareto_front]
    return pareto_metrics, pareto_solutions

def build_tree_structure(adj_matrix):
    n = adj_matrix.shape[0]
    children = {i: [] for i in range(n)}
    parents = {}
    for parent in range(n):
        for child in range(n):
            if adj_matrix[parent, child] != 0:
                children[parent].append(child)
                parents[child] = parent
    root = next(i for i in range(n) if i not in parents)
    return children, parents, root

def post_order_traversal(children, root):
    visited = set()
    order = []
    def dfs(node):
        for child in children.get(node, []):
            if child not in visited:
                dfs(child)
        visited.add(node)
        order.append(node)
    dfs(root)
    return order

def compute_cost_matrix(children, post_order, observed, n, k):
    cost = torch.full((k, n), float('inf'))
    for node in range(n):
        if node in observed:
            cost[:, node] = float('inf')
            cost[observed[node], node] = 0

    for node in post_order:
        if node not in children or not children[node]:
            continue
        for label in range(k):
            total = 0
            for child in children[node]:
                child_cost = cost[:, child]
                penalties = (torch.arange(k) != label).float()
                total += torch.min(child_cost + penalties)
            cost[label, node] = total
    return cost

def assign_optimal_labels(children, cost, root, root_label, k):
    n = cost.shape[1]
    labels = [None] * n
    labels[root] = root_label
    def assign(node):
        for child in children.get(node, []):
            best_label = None
            best_score = float('inf')
            for l in range(k):
                transition = 0 if l == labels[node] else 1
                score = cost[l, child].item() + transition
                if score < best_score:
                    best_score = score
                    best_label = l
            labels[child] = best_label
            assign(child)
    assign(root)
    return labels

def build_one_hot_matrix(labels, n, k, root):
    mat = torch.zeros((k, n), dtype=torch.float32)
    for node, label in enumerate(labels):
        mat[label, node] = 1.0
    return torch.cat((mat[:, :root], mat[:, root+1:]), dim=1)

def run_fitch_hartigan(v_solver, results):
    adj_matrix = v_solver.input_T
    node_idx_to_observed_sites = v_solver.idx_to_observed_sites
    root_label = torch.argmax(v_solver.p, dim=0).item()
    
    n = adj_matrix.shape[0]
    k = v_solver.num_sites

    children, parents, root = build_tree_structure(adj_matrix)
    post_order = post_order_traversal(children, root)
    cost = compute_cost_matrix(children, post_order, node_idx_to_observed_sites, n, k)
    labels = assign_optimal_labels(children, cost, root, root_label, k)
    single_soln_matrix = build_one_hot_matrix(labels, n, k, root)

    V = stack_vertex_labeling(v_solver.L, add_batch_dim(single_soln_matrix), v_solver.p, None, None)
    T = repeat_n(v_solver.full_T, 1)
    if v_solver.config['use_sparse_T']:
        T = T.to_sparse()

    metrics = ancestral_labeling_metrics(V, T, v_solver.full_G, v_solver.O, v_solver.p,
                                         update_path_matrix=True, compute_full_c=True, identical_T=True)
    print("Fitch-hartigan result:", metrics)
    results.append((V.cpu(), torch.zeros_like(V), T.cpu(), None, (*[m.cpu() for m in metrics], torch.zeros(1))))

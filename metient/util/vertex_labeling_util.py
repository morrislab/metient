import numpy as np
import torch
import networkx as nx
import math
from queue import Queue
from collections import OrderedDict
from metient.util.globals import *

import scipy.sparse as sp
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class LabeledTree:
    def __init__(self, tree, labeling):
        if (tree.shape[0] != tree.shape[1]):
            raise ValueError("Adjacency matrix should have shape (num_nodes x num_nodes)")
        if (tree.shape[0] != labeling.shape[1]):
            raise ValueError("Vertex labeling matrix should have shape (num_sites x num_nodes)")

        self.tree = tree
        self.labeling = labeling

    def __eq__(self, other):
        return ( isinstance(other, LabeledTree) and 
                (torch.equal(torch.nonzero(self.labeling), torch.nonzero(other.labeling))) and
                (torch.equal(torch.nonzero(self.tree), torch.nonzero(other.tree))))

    def __hash__(self):
        return hash(self.labeling.numpy().tobytes())*hash(self.tree.numpy().tobytes())

    def __str__(self):
        A = str(np.where(self.tree == 1))
        V = str(np.where(self.labeling == 1))
        return f"Tree: {A}\nVertex Labeling: {V}"

def calculate_batch_size(T, sites):
    num_nodes = T.shape[0]
    num_sites = len(sites)
    min_size = 2048
    if num_nodes > 15:
        min_size += 1024 * (num_nodes // 2)
        if num_sites > 3:
            min_size += 256 * (num_sites)

    elif num_sites > 4:
        min_size += 256 * (num_sites // 2)

    # cap this to a reasonably high sample size
    min_size = min(min_size, 60000)

    return min_size

def tree_iterator(T):
    '''
    iterate an adjacency matrix, returning i and j for all values = 1
    '''
    # enumerating through a torch tensor is pretty computationally expensive,
    # so convert to a sparse matrix to efficiently access non-zero values
    T = T if isinstance(T, np.ndarray) else T.detach().numpy()
    T = sp.coo_matrix(T)
    for i, j in zip(T.row, T.col):
        yield i,j

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
    returns the root idx (node with no inbound edges) from adjacency matrix T
    '''

    candidates = set([x for x in range(len(T))])
    for i, j in tree_iterator(T):
        candidates.remove(j)
    msg = "More than one" if len(candidates) > 1 else "No"
    assert (len(candidates) == 1), f"{msg} root node detected"

    return list(candidates)[0]

def restructure_matrices(adj_matrix,ref_matrix, var_matrix, node_idx_to_label, gen_dist_matrix):
    """
    Restructure the adjacency matrix, ref matrix and var matrix so that the node at index 0 becomes the root node.

    Parameters:
    - adj_matrix: 2D tensor representing the adjacency matrix.

    Returns:
    - Restructured adjacency matrix.
    """
    root_idx = get_root_index(adj_matrix)
    new_order = [x for x in range(len(adj_matrix))]
    new_order[root_idx] = 0
    new_order[0] = root_idx

    # Swap rows and columns to move the first row and column to the desired position
    swapped_adjacency_matrix = adj_matrix[new_order, :][:, new_order]
    swapped_ref_matrix = ref_matrix[:, new_order] if ref_matrix != None else None
    swapped_var_matrix = var_matrix[:, new_order] if var_matrix != None else None
    if gen_dist_matrix == None:
        swapped_gen_dist_matrix = None
    else:
        swapped_gen_dist_matrix = gen_dist_matrix[new_order, :][:, new_order]

    original_root_label = node_idx_to_label[root_idx]
    original_node0_label = node_idx_to_label[0]
    node_idx_to_label[0] = original_root_label
    node_idx_to_label[root_idx] = original_node0_label

    return swapped_adjacency_matrix, swapped_ref_matrix, swapped_var_matrix, node_idx_to_label, swapped_gen_dist_matrix

def print_U(U, B, node_idx_to_label, ordered_sites, ref, var):
    cols = ["GL"]+[";".join(node_idx_to_label[i][:2]) for i in range(len(node_idx_to_label))]
    U_df = pd.DataFrame(U.detach().numpy(), index=ordered_sites, columns=cols)

    print("U\n", U_df)
    F_df = pd.DataFrame((var/(ref+var)).numpy(), index=ordered_sites, columns=cols[1:])
    print("F\n", F_df)
    Fhat_df = pd.DataFrame((U @ B).detach().numpy()[:,1:], index=ordered_sites, columns=cols[1:])
    print("F hat\n", Fhat_df)

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
    so there are no children or grandchildren's observed to help bias towards
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
    figure out the sites that the resolver nodes should be initialized with
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
    print("node_idx", node_idx, "node_and_children", node_and_children, "sites", sites)
    # Special case where the node of interest (node_idx) isn't detected in any sites
    # and is on a linear branch 
    if not include_children and len(sites) == 0 and len(clone_tree_children) == 1:
        leaf_on_linear_branch = traverse_until_Uleaf_or_mult_children(node_idx, input_T, node_idx_to_sites)
        print("leaf_on_linear_branch", leaf_on_linear_branch)
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
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0],]
    internal_node_idx_to_sites = {0:[1], 1:[0,1], 2:[1}
    k = 3
    include_children = True
    min_num_sites = 1
    returns ([0], [[1]]) (node at index 0 has 3 children, and the resolver node under it should be
    place in site 1, since node 0 and its child node 1 are both detected in site 1)

    if include_children is False, we only look at node 0's U values and not its children's U values
    '''
    row_sums = torch.sum(T, axis=1)
    node_indices = list(np.where(row_sums >= k)[0])
    filtered_node_indices = []
    all_resolver_sites = []
    for node_idx in node_indices:
        # get the sites that node_idx and node_idx's children are estimated in U
        resolver_sites = get_resolver_sites_of_node_and_children(input_T, internal_node_idx_to_sites, row_sums[node_idx], node_idx, include_children, min_num_sites, cutoff)
        print("resolver_sites", resolver_sites)
        # a resolver node wouldn't help for this polytomy
        if len(resolver_sites) == 0:
            continue
        filtered_node_indices.append(node_idx)
        all_resolver_sites.append(resolver_sites)
    return filtered_node_indices, all_resolver_sites

def get_child_indices(T, indices):
    '''
    returns the indices of direct children of the nodes at indices
    '''
    all_child_indices = []

    for parent_idx in indices:
        children = np.where(T[parent_idx,:] > 0)[0]
        for child_idx in children:
            if child_idx not in all_child_indices:
                all_child_indices.append(child_idx)

    return all_child_indices

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
    return x.repeat(n,1).reshape(n, x.shape[0], x.shape[1])

def get_path_matrix(T, remove_self_loops=False):
    bs = 0 if len(T.shape) == 2 else T.shape[0]
    # Path matrix that tells us if path exists from node i to node j
    I = repeat_n(torch.eye(T.shape[1]), bs)
    # M is T with self loops.
    # Convert to bool to get more efficient matrix multiplicaton
    B = torch.logical_or(T,I).int()
    # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
    # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
    k = np.ceil(np.log2(T.shape[1]))
    for _ in range(int(k)):
        B = torch.matmul(B, B)
    if remove_self_loops:
        B = torch.logical_xor(B,I)
    P = torch.sigmoid(BINARY_ALPHA * (2*B - 1))
    return P

def get_path_matrix_tensor(A):
    '''
    A is a numpy adjacency matrix
    '''
    return torch.tensor(get_path_matrix(A, remove_self_loops=True), dtype = torch.float32)

def get_mutation_matrix_tensor(A):
    '''
    A is an numpy ndarray or tensor adjacency matrix (where Aij = 1 if there is a path from i to j)

    returns a mutation matrix B, which is a subclone x mutation binary matrix, where Bij = 1
    if subclone i has mutation j.
    '''
    return torch.tensor(get_path_matrix(A.T, remove_self_loops=False), dtype = torch.float32)

def get_adj_matrix_from_edge_list(edge_list):
    T = []
    nodes = set([node for edge in edge_list for node in edge])
    T = [[0 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for edge in edge_list:
        T[ord(edge[0]) - 65][ord(edge[1]) - 65] = 1
    return torch.tensor(T, dtype = torch.float32)

def is_tree(adj_matrix):
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges)
    return (not nx.is_empty(g) and nx.is_tree(g))

def annealing_spiking(x, anneal_rate, init_temp, spike_interval=20):
    # Calculate the base value from the last spike or the starting value
    current = init_temp * np.exp(-anneal_rate * x)
    last_spike = init_temp

    # Iterate over each spike point to adjust the current value
    for i in range(spike_interval, x + 1, spike_interval):
        spike_base = last_spike * np.exp(-anneal_rate * spike_interval)
        last_spike = spike_base + 0.5 * (last_spike - spike_base)

    # If x is exactly on a spike, return the last spike value
    if x % spike_interval == 0 and x != 0:
        return last_spike
    # Otherwise, anneal from the last spike
    else:
        return last_spike * np.exp(-anneal_rate * (x % spike_interval))

######################################################
######### POST U MATRIX ESTIMATION UTILITIES #########
######################################################

# def get_leaf_labels_from_U(U):
#     U = U[:,1:] # don't include column for normal cells
#     num_sites = U.shape[0]
#     L = torch.nn.functional.one_hot((U > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
#     return L

def get_leaf_labels_from_U(U, input_T):
    U = U[:,1:] # don't include column for normal cells
    P = get_path_matrix(input_T, remove_self_loops=True)
    internal_node_idx_to_sites = {}
    for node_idx in range(U.shape[1]):
        descendants = np.where(P[node_idx] == 1)[0]
        for site_idx in range(U.shape[0]):
            node_U = U[site_idx,node_idx]

            is_present = False
            if node_U > 0.01:
                if len(descendants) == 0: # leaf node in the internal clone tree
                    is_present = True
                else:
                    descendants_U = sum(U[site_idx,descendants])
                    if node_U/(node_U+descendants_U) > 0.1:
                        is_present = True
        
            if is_present:
                if node_idx not in internal_node_idx_to_sites:
                    internal_node_idx_to_sites[node_idx] = []
                internal_node_idx_to_sites[node_idx].append(site_idx)
    return internal_node_idx_to_sites

def full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val):
    '''
    All non-zero values of U represent extant clones (leaf nodes of the full tree).
    For each of these non-zero values, we add an edge from parent clone to extant clone.
    '''
    num_leaves = sum(len(lst) for lst in idx_to_sites_present.values())
    full_adj = torch.nn.functional.pad(input=input_T, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0)
    leaf_idx = input_T.shape[0]
    # Also add branch lengths (genetic distances) for the edges we're adding.
    # Since all of these edges we're adding represent genetically identical clones,
    # we are going to add a very small but non-zero value.
    full_G = torch.nn.functional.pad(input=input_G, pad=(0, num_leaves, 0, num_leaves), mode='constant', value=0) if input_G is not None else None

    leaf_labels = []
    # Iterate through the internal nodes that we want to attach leaf nodes to
    for internal_node_idx in idx_to_sites_present:
        # Attach a leaf node for every site that this internal node is observed in
        for site in idx_to_sites_present[internal_node_idx]:
            full_adj[internal_node_idx, leaf_idx] = 1
            if input_G is not None:
                full_G[internal_node_idx, leaf_idx] = G_identical_clone_val
            leaf_idx += 1
            leaf_labels.append(site)
    print("idx_to_sites_present", idx_to_sites_present)
    print("leaf_labels", leaf_labels)
    # Anatomical site labels of the leaves
    L = torch.nn.functional.one_hot(torch.tensor(leaf_labels), num_classes=num_sites).T
    return full_adj, full_G, L
    
def full_adj_matrix_using_inputted_observed_clones(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val):
    '''
    Use inputted observed clones to fill out T and G by adding leaf nodes
    '''
    # Make a fake U to make life easy 
    # U = torch.zeros(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    # for idx in idx_to_sites_present:
    #     sites = idx_to_sites_present[idx]
    #     for site in sites:
    #         U[site,idx+1] = 1
    # print("U", U)
    full_adj, full_G, L = full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, idx_to_sites_present, num_sites, G_identical_clone_val)
    return L, full_adj, full_G

def full_adj_matrix_using_inferred_observed_clones(U, input_T, input_G, num_sites, G_identical_clone_val):
    '''
    Use inferred observed clones to fill out T and G by adding leaf nodes
    '''
    internal_node_idx_to_sites = get_leaf_labels_from_U(U,input_T)
    full_adj, full_G, L = full_adj_matrix_from_internal_node_idx_to_sites_present(input_T, input_G, internal_node_idx_to_sites, num_sites, G_identical_clone_val)
    
    return full_adj, full_G, L, internal_node_idx_to_sites

#########################################
###### POLYTOMY RESOLUTION UTILIES ######
#########################################

class PolytomyResolver():

    def __init__(self, T, G, num_sites, num_leaves, bs, node_idx_to_label, nodes_w_polys, resolver_sites, identical_clone_value):
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
        children_of_polys = get_child_indices(T, nodes_w_polys)
        #print("children_of_polys", children_of_polys)

        # 4. Initialize a matrix to learn the polytomy structure
        num_nodes_full_tree = T.shape[0]
        poly_adj_matrix = repeat_n(torch.zeros((num_nodes_full_tree, len(children_of_polys)), dtype=torch.float32), bs)
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

        resolver_labeling = torch.zeros(num_sites, len(resolver_indices))
        t = 0
        for sites in resolver_sites:
            for site in sites:
                resolver_labeling[site, t] = 1
                t += 1
        #print("resolver_labeling", resolver_labeling)

        # print("resolver_indices", resolver_indices)
        offset = 0
        for parent_idx in nodes_w_polys:
            child_indices = get_child_indices(T, [parent_idx])
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
                node_idx_to_label[j] = [f"{i}pol{j}"]
                if G != None:
                    G[i,j] = identical_clone_value

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
                potential_child_indices = get_child_indices(T, [parent_idx])
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
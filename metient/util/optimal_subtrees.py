import torch
from metient.util import vertex_labeling_util as vutil

def get_sparse_optimal_subtree_indices(V, P):
    """    
    Parameters:
        V (torch.Tensor): A dense binary tensor (batch_size, num_sites, num_nodes).
        P (torch.sparse.Tensor): A sparse path matrix tensor (batch_size, num_nodes, num_nodes).
        
    Returns:
        torch.Tensor: A tensor resulting from the logical operations.
    """
    bs = P.shape[0]

    cand_optimal_subtree_indices = []
    for i in range(bs):
        _V = V[i]
        _X = (_V.T @ _V)
        _P = P[i].to_dense()
        # i,j = 1 if node i and node j are not connected or they're in the same site
        same_color_subtrees = torch.logical_not(torch.logical_and(1 - _X, _P))
        # Get the indices of rows where all elements are True (all nodes have the same label)
        _optimal_subtree_indices = torch.nonzero(torch.all(same_color_subtrees, dim=1))
        # Tells us how many descendants each node has
        row_sums = torch.sum(_P, dim=1)
        indexed_row_sums = torch.tensor([row_sums[idx[0]] for idx in _optimal_subtree_indices]).unsqueeze(1)
        bs_list = [i for _ in range(len(indexed_row_sums))]
        cand_optimal_subtree_indices.extend([[item[0],int(item[1]),int(item[2])] for item in zip(bs_list, _optimal_subtree_indices, indexed_row_sums)])
        
        del _V, _X, _P, same_color_subtrees
    # Sort the optimal_subtrees by the number of children they have,
    # so that when we are solving for polytomies, we get the largest optimal subtrees possible
    cand_optimal_subtree_indices = sorted(cand_optimal_subtree_indices, key=lambda x: x[2], reverse=True)
    return cand_optimal_subtree_indices

def get_optimal_subtree_indices(T, V, P):

    if T.is_sparse:
        return get_sparse_optimal_subtree_indices(V, P)
    
    VT = torch.transpose(V, 2, 1)
    # i,j = 1 if node i and node j have the same label 
    X = VT @ V
    # i,j = 1 if node i and node j are not connected or they're in the same site
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

    return cand_optimal_subtree_indices

def get_descendants(P, batch_num, parent_idx):
    """    
    Parameters:
        P (torch.sparse.Tensor): A dense or sparse path matrix tensor (batch_size, num_nodes, num_nodes).
        batch_num: The batch number to retrieve from
        parent_idx: The index of the parent node whose descendants will be returned 
        
    Returns:
        List of descendant indices of parent_idx (i.e. each descendant can be reached from parent_idx)
    """
    if not P.is_sparse:
        return [t.item() for t in torch.nonzero(P[batch_num,parent_idx])]
    P = P.coalesce()
    sparse_indices = P.indices()

    # Filter by batch and row
    batch_mask = sparse_indices[0] == batch_num
    row_mask = sparse_indices[1] == parent_idx
    final_mask = batch_mask & row_mask

    # Extract the non-zero column indices for the specified batch and row
    non_zero_columns = sparse_indices[2][final_mask]
    non_zero_columns = [t.item() for t in non_zero_columns]
    return non_zero_columns

def find_optimal_subtree_nodes(T, V, v_solver, num_internal_nodes):
    '''
    Args:
        - T: all possible solutions for adjacency matrices
        - V: all possible solutions for vertex labelings
    Returns:
        A list of node indices and their descendants which belong to optimal subtrees (i.e.)
        all nodes in the subtree have the same color/label, and a list of the batch numbers
        that these optimal subtrees were found 
    '''

    solve_polytomies = v_solver.config['solve_polytomies']
    P = vutil.path_matrix(T, remove_self_loops=False, identical_T=(not solve_polytomies))
    cand_optimal_subtree_indices = get_optimal_subtree_indices(T, V, P)
    seen_nodes = set()
    optimal_batch_nums, optimal_subtree_nodes = [],[]
    for cand in cand_optimal_subtree_indices:
        batch_num = int(cand[0])
        optimal_subtree_root = int(cand[1])
        descendants = set(get_descendants(P, batch_num, optimal_subtree_root))
        # Don't bother with nodes we've already seen
        seen_subset = not optimal_subtree_root in seen_nodes and not (descendants - set([optimal_subtree_root])).issubset(seen_nodes)
        if seen_subset:
            # Add the optimal_subtree_root and all its descendants
            # Don't fix witness nodes (num. descendants == 0), since we already know their labeling, 
            # and if they are under an optimal polytomy branch, they would be getting added by an optimal
            # subtree rooted by an ancestor
            if len(descendants) == 0: 
                continue
            # Don't fix a subtree where there are no leaf nodes 
            # (this is rare, and these nodes aren't well estimated)
            leaf_node_in_optimal_subtree = False
            for descendant in descendants:
                if descendant >= num_internal_nodes:
                    leaf_node_in_optimal_subtree = True
            if not leaf_node_in_optimal_subtree:
                continue

            current_node_set = [optimal_subtree_root]
            seen_nodes.add(optimal_subtree_root)
            for descendant in descendants:
                if descendant not in seen_nodes:
                    current_node_set.append(descendant)
                    seen_nodes.add(descendant)
            if len(current_node_set) > 0:
                optimal_batch_nums.append(batch_num)
                optimal_subtree_nodes.append(current_node_set)
    return optimal_subtree_nodes, optimal_batch_nums

def get_parent_idx_sparse_t(T, optimal_batch_num, child_idx):
    T = T.coalesce()
    indices = T.indices()
    bss,iss,jss = indices[0], indices[1], indices[2]
    for b,i,j in zip(bss,iss,jss):
        if b == optimal_batch_num and j == child_idx:
            return int(i)
    assert(False, "Parent index not found")

def get_child_indices_sparse_t(T, optimal_batch_num, parent_idx):
    T = T.coalesce()
    indices = T.indices()
    
    # Extract indices for batches, rows, and columns
    bss, iss, jss = indices[0], indices[1], indices[2]
    
    # Apply boolean masks to filter for the conditions
    mask = (bss == optimal_batch_num) & (iss == parent_idx)
    
    # Extract the corresponding child indices using the mask
    child_indices = jss[mask].tolist()  # Convert the result to a list
    
    return child_indices

def init_optimal_x_polyres(X, poly_res, optimal_subtree_nodes, optimal_batch_nums, T, V, v_solver):
    
    poly_resolver_to_optimal_children = {}
    known_indices = []
    known_labelings = []
    # TODO: re-initialize after first optimization?
    if poly_res != None:
        # Anywhere the resolver is not -inf, reset the starting values to 1s
        poly_res.latent_var[poly_res.latent_var != float('-inf')] = 1
            
    # Fix node labels and node edges
    for optimal_subtree_set,optimal_batch_num in zip(optimal_subtree_nodes,optimal_batch_nums):
        print("optimal_subtree_set", optimal_subtree_set)
        for i, node_idx in enumerate(optimal_subtree_set):
            # print("node_idx", node_idx)
            # If this is a witness node or the root index, we already know its vertex labeling
            node_children = get_child_indices_sparse_t(T, optimal_batch_num, node_idx)
            # Don't lock in the labeling of an unused polytomy node
            is_unused_poly_resolver = poly_res != None and node_idx in poly_res.resolver_indices and len(node_children) < 2
            # print("node_children", node_children, "is_unused_poly_resolver", is_unused_poly_resolver)
            if (node_idx <= X.shape[2] and node_idx != 0) and not is_unused_poly_resolver:
                optimal_site = int(V[optimal_batch_num,:,node_idx].nonzero(as_tuple=False))
                idx = node_idx - 1 # X doesn't include root node
                known_indices.append(idx)
                known_labelings.append(torch.eye(v_solver.num_sites)[optimal_site].T)
                X[:,optimal_site,idx] = 1
                non_optimal_sites = [i for i in range(v_solver.num_sites) if i != optimal_site]
                X[:,non_optimal_sites,idx] = float("-inf")
                # print("optimal_site", optimal_site)

            # If this node is the child of a polytomy resolver node, fix its location
            # if the parent (the polytomy resolver node) belongs to the same optimal subtree
            if poly_res != None and node_idx in poly_res.children_of_polys:
                poly_idx = poly_res.children_of_polys.index(node_idx)
                parent_idx = get_parent_idx_sparse_t(T,optimal_batch_num,node_idx)
                optimal_children = get_child_indices_sparse_t(T, optimal_batch_num, parent_idx)
                if parent_idx not in optimal_subtree_set or parent_idx not in poly_res.resolver_indices or len(optimal_children) < 2:
                    continue
                poly_res.latent_var[:,parent_idx, poly_idx] = 1
                non_parents = [i for i in range(T.shape[1]) if i != parent_idx]
                poly_res.latent_var[:,non_parents,poly_idx] = float("-inf")
                poly_res.latent_var[:,non_parents,poly_idx] = float("-inf")
                # print("fixing", parent_idx, node_idx)

                # Don't let any other non-optimal children of this polytomy resolver move around
                if parent_idx in poly_res.resolver_indices:
                    if parent_idx not in poly_resolver_to_optimal_children:
                        poly_resolver_to_optimal_children[parent_idx] = optimal_children

    print("poly_resolver_to_optimal_children", poly_resolver_to_optimal_children)
    if poly_res != None:
        # Fix all other polytomy children s.t. they cannot move to be a child of the fixed node_idx
        for parent_idx in poly_resolver_to_optimal_children:
            optimal_children = poly_resolver_to_optimal_children[parent_idx]
            if len(optimal_children) < 2:
                continue
            optimal_children_poly_indices = [poly_res.children_of_polys.index(i) for i in optimal_children]
            other_children = [i for i in range(poly_res.latent_var.shape[2]) if i not in optimal_children_poly_indices]
            poly_res.latent_var[:,parent_idx,other_children] = float("-inf")
            # print("fixing children of", parent_idx)
        
        half_bs = X.shape[0]//2
        # x-1 because X doesn't include root node
        result = [(x-1, i) for i, x in enumerate(poly_res.resolver_indices) if x-1 not in known_indices]
        if result:  # Ensure the result is not empty
            unknown_resolver_indices, mask = zip(*result)
            resolved_indices = vutil.repeat_n(poly_res.resolver_labeling[:,mask], half_bs)
            resolved_indices[resolved_indices == 0] = float('-inf')
            X[:half_bs,:,unknown_resolver_indices] = resolved_indices
            # print(X.shape)
            # print("poly_res.resolver_labeling.shape", poly_res.resolver_labeling.shape)
            # print(poly_res.resolver_labeling)
            # print("unknown_resolver_indices", unknown_resolver_indices)
            # print("with mask", poly_res.resolver_labeling[:,mask])
            # print("vutil.repeat_n(poly_res.resolver_labeling[:,mask], half_bs)",vutil.repeat_n(poly_res.resolver_labeling[:,mask], half_bs).shape, )
            # print("X[:half_bs,:,unknown_resolver_indices]",X[:half_bs,:,unknown_resolver_indices].shape)
            # print(X[0], "\n",X[-1])

        poly_res.latent_var.requires_grad = True

    fixed_labeling = None
    if len(known_indices) != 0:
        unknown_indices = [x for x in range(v_solver.num_nodes_to_label) if x not in known_indices]
        known_labelings = torch.stack(known_labelings, dim=1)
        X = X[:,:,unknown_indices] # only include the unknown indices for inference
        fixed_labeling = vutil.FixedVertexLabeling(known_indices, unknown_indices, known_labelings)
        print("unknown_indices", len(unknown_indices), len(known_indices), [x+1 for x in unknown_indices], "known_indices", [x+1 for x in known_indices])
    
    v_solver.fixed_labeling = fixed_labeling
    X.requires_grad = True

    return X, v_solver

def find_optimal_subtrees(T, V, v_solver):
    '''
    After the first round of optimization, there are optimal subtrees (subtrees where
    the labelings of *all* nodes is the same), which we can keep fixed, since there
    are no other more optimal labelings rooted at this branch.

    Two things we can fix: the labeling of the nodes in optimal subtrees,
    and the edges of the subtrees if polytomy resolution is being used. Search all
    samples to find optimal subtrees, since there might not be one solution with all 
    optimal subtrees.
    '''
    
    num_internal_nodes = v_solver.num_nodes_to_label + 1 # root node
    
    # Re-initialize T with the tree with the best subtree structure
    poly_res = v_solver.poly_res
    if poly_res != None:
        poly_res.latent_var.requires_grad = False
    
    # Find samples with optimal subtrees
    optimal_subtree_nodes, optimal_batch_nums = find_optimal_subtree_nodes(T, V, v_solver, num_internal_nodes)

    return optimal_subtree_nodes, optimal_batch_nums

def collapse_optimal_subtrees_with_expansion_fn(v_solver, V, T, optimal_nodes, optimal_batch_nums):
    """
    Collapses optimal subtrees in V and T for efficiency during optimization,
    and returns a function to expand them later.
    
    Args:
        v_solver: The vertex labeling solver instance
        V: Vertex labeling matrix
        T: Tree adjacency matrix
        optimal_nodes: List of nodes that are roots of optimal subtrees
        optimal_batch_nums: Batch numbers corresponding to optimal solutions
        
    Returns:
        tuple: (collapsed_V, collapsed_T, expand_fn, original_V, original_T)
    """
    # Collapse the trees
    collapsed_V, collapsed_T = opt_sub.collapse_optimal_subtrees(V, T, optimal_nodes)
    
    # Create expansion function that will be called later
    def expand_fn(V_to_expand, T_to_expand, original_V, original_T):
        expanded_V, expanded_T = opt_sub.expand_optimal_subtrees(
            V_to_expand, 
            T_to_expand,
            original_V,  # original V with optimal subtrees
            original_T,  # original T with optimal subtrees
            optimal_nodes,
            optimal_batch_nums
        )
        return expanded_V, expanded_T
    
    return collapsed_V, collapsed_T, expand_fn, V, T
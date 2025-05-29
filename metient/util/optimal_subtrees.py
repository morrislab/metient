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
        P: Path matrix (batch_size, num_nodes, num_nodes).
        batch_num: The batch number to retrieve from
        parent_idx: The index of the parent node whose descendants will be returned 
        
    Returns:
        List of descendant indices of parent_idx (i.e. each descendant can be reached from parent_idx)
    """
    if not P.is_sparse:
        return [t.item() for t in torch.nonzero(P[batch_num,parent_idx])]
    
    # Coalesce only if needed
    if not P.is_coalesced():
        P = P.coalesce()

    P_indices = P.indices()
    # Get coalesced indices and create combined mask
    mask = (P_indices[0] == batch_num) & (P_indices[1] == parent_idx)
    
    # Return filtered column indices directly as list
    return P_indices[2][mask].tolist()

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
        # Getting descendants is expensive, so don't bother if we've already seen this node
        # since cand_optimal_subtree_indices is sorted by the number of descendants
        if optimal_subtree_root in seen_nodes:
            continue
        descendants = set(get_descendants(P, batch_num, optimal_subtree_root))
        # Don't bother with a subset of nodes we've already added to an optimal subtree
        not_seen_subset = not (descendants - set([optimal_subtree_root])).issubset(seen_nodes)
        if not_seen_subset:
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


def _collect_node_fixes(node_idx, optimal_batch_num, X, V, T, poly_res):
    """Determine if and how a node's labeling should be fixed.
    
    Args:
        node_idx (int): Index of the node to check
        optimal_batch_num (int): Batch number for this optimal solution
        X (torch.Tensor): Vertex labeling tensor to be modified
        V (torch.Tensor): Current vertex labeling solutions
        T (torch.Tensor): Tree adjacency matrix
        poly_res: Polytomy resolver object or None
    
    Returns:
        tuple: (should_fix, node_idx, optimal_site) or None if node should not be fixed
    """
    is_unused_poly_resolver = False
    if poly_res is not None:
        node_children = vutil.get_child_indices(T[optimal_batch_num],[node_idx])
        is_unused_poly_resolver = (node_idx in poly_res.resolver_indices and 
                                   len(node_children) < 2)
    
    if (node_idx <= X.shape[2] and node_idx != 0) and not is_unused_poly_resolver:
        optimal_site = int(V[optimal_batch_num,:,node_idx].nonzero(as_tuple=False))
        return (node_idx, optimal_site, optimal_batch_num)
    return None

def _collect_poly_fixes(node_idx, optimal_batch_num, optimal_subtree_set, T, poly_res):
    """Determine if and how a polytomy resolver should be fixed.
    
    Args:
        node_idx (int): Index of the node to check
        optimal_batch_num (int): Batch number for this optimal solution
        optimal_subtree_set (list): Set of nodes in the current optimal subtree
        T (torch.Tensor): Tree adjacency matrix
        poly_res: Polytomy resolver object
        
    Returns:
        tuple: (parent_idx, node_idx, poly_idx, optimal_children) or None if no fixes needed
    """
    if poly_res is None or node_idx not in poly_res.children_of_polys:
        return None
        
    poly_idx = poly_res.children_of_polys.index(node_idx)
    parent_idx = vutil.get_parent(T[optimal_batch_num,:,:], node_idx)
    optimal_children = vutil.get_child_indices(T[optimal_batch_num,:,:], [parent_idx])
    
    if (parent_idx not in optimal_subtree_set or 
        parent_idx not in poly_res.resolver_indices or 
        len(optimal_children) < 2):
        return None
        
    return (parent_idx, node_idx, poly_idx, optimal_children)

def _apply_node_fixes(nodes_to_fix, X, v_solver):
    """Apply collected node labeling fixes to X tensor.
    
    Args:
        nodes_to_fix (list): List of (node_idx, optimal_site, batch_num) tuples
        X (torch.Tensor): Vertex labeling tensor to modify
        v_solver: Vertex labeling solver instance
        
    Returns:
        tuple: (known_indices, known_labelings)
    """
    known_indices = []
    known_labelings = []
    
    for node_idx, optimal_site, _ in nodes_to_fix:
        known_indices.append(node_idx)
        known_labelings.append(torch.eye(v_solver.num_sites)[optimal_site])
        idx = node_idx - 1  # X doesn't include root node
        X[:, optimal_site, idx] = 1
        non_optimal_sites = [i for i in range(v_solver.num_sites) if i != optimal_site]
        X[:, non_optimal_sites, idx] = float("-inf")
    return known_indices, known_labelings

def _apply_poly_fixes(poly_fixes, poly_res, T):
    """Apply collected polytomy resolver fixes.
    
    Args:
        poly_fixes (list): List of (parent_idx, node_idx, poly_idx, optimal_children) tuples
        poly_res: Polytomy resolver object
        T (torch.Tensor): Tree adjacency matrix
    """
    poly_resolver_to_optimal_children = {}
    
    for parent_idx, _, poly_idx, optimal_children in poly_fixes:
        # Fix the current child's position
        poly_res.latent_var[:, parent_idx, poly_idx] = 1
        non_parents = [i for i in range(T.shape[1]) if i != parent_idx]
        poly_res.latent_var[:, non_parents, poly_idx] = float("-inf")
        
        # Track optimal children for each resolver
        if parent_idx in poly_res.resolver_indices:
            poly_resolver_to_optimal_children[parent_idx] = optimal_children
            
    return poly_resolver_to_optimal_children

def _remove_nodes_from_T(T, nodes_to_remove):
    T = T.coalesce()
    indices = T.indices()
    values = T.values()
    
    # Create batch-aware mask for nodes to remove
    batch_indices = indices[0]  # (num_edges,)
    source_nodes = indices[1]  # (num_edges,)
    target_nodes = indices[2]  # (num_edges,)
    nodes_to_remove_tensor = torch.tensor(list(nodes_to_remove), device=source_nodes.device)
    # Keep edges where neither source nor target is in nodes_to_remove
    keep_mask = ~((source_nodes.unsqueeze(1) == nodes_to_remove_tensor.unsqueeze(0)).any(dim=1) | 
                    (target_nodes.unsqueeze(1) == nodes_to_remove_tensor.unsqueeze(0)).any(dim=1))
    
    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]
    # Create mapping for remaining nodes
    remaining_nodes = sorted(set(range(T.shape[1])) - nodes_to_remove)
    old_to_new = {old: new for new, old in enumerate(remaining_nodes)}
    
    # Remap indices
    new_indices[1:] = torch.tensor([
        [old_to_new[idx.item()] for idx in new_indices[1]],
        [old_to_new[idx.item()] for idx in new_indices[2]]
    ])
    
    # Create new T with reduced dimensions
    new_size = (T.shape[0], len(remaining_nodes), len(remaining_nodes))

    return torch.sparse_coo_tensor(new_indices, new_values, new_size).coalesce(), old_to_new

def init_optimal_x_polyres(X, poly_res, optimal_subtree_nodes, optimal_batch_nums, T, V, v_solver):
    """Initialize X and polytomy resolver with optimal subtrees fixed.
    
    This function fixes the labeling of nodes in optimal subtrees and their positions
    in polytomy resolutions. An optimal subtree is one where all nodes have the same label.
    
    Args:
        X (torch.Tensor): Vertex labeling tensor to be modified
        poly_res: Polytomy resolver object or None
        optimal_subtree_nodes (list): Lists of nodes in each optimal subtree
        optimal_batch_nums (list): Batch numbers corresponding to each optimal subtree
        T (torch.Tensor): Tree adjacency matrix
        V (torch.Tensor): Current vertex labeling solutions
        v_solver: Vertex labeling solver instance
        
    Returns:
        tuple: (modified X tensor, modified v_solver)
    """
    # Reset polytomy resolver if it exists
    if poly_res is not None:
        poly_res.latent_var = poly_res.latent_var.detach()
        poly_res.latent_var[poly_res.latent_var != float('-inf')] = 1

    # First pass: collect all modifications needed
    nodes_to_fix = []
    poly_fixes = []
    optimal_root_nodes = []
    
    for optimal_subtree_set, optimal_batch_num in zip(optimal_subtree_nodes, optimal_batch_nums):
        #print("optimal_subtree_set", optimal_subtree_set)
        for i, node_idx in enumerate(optimal_subtree_set):
            # Collect node labeling fixes
            node_fix = _collect_node_fixes(node_idx, optimal_batch_num, X, V, T, poly_res)
            if node_fix:
                nodes_to_fix.append(node_fix)
                if i == 0:
                    optimal_root_nodes.append(node_idx)
                
            # Collect polytomy resolver fixes
            poly_fix = _collect_poly_fixes(node_idx, optimal_batch_num, optimal_subtree_set, T, poly_res)
            if poly_fix:
                poly_fixes.append(poly_fix)

    # Second pass: apply modifications
    known_indices, known_labelings = _apply_node_fixes(nodes_to_fix, X, v_solver)
    #print("known_indices", known_indices)
    if poly_res is not None:
        
        poly_resolver_to_optimal_children = _apply_poly_fixes(poly_fixes, poly_res, T)
        
        # Once a polytomy resolver has found its optimal children, fix its other children so that they cannt move
        for parent_idx, optimal_children in poly_resolver_to_optimal_children.items():
            if len(optimal_children) < 2:
                continue
            optimal_children_poly_indices = [poly_res.children_of_polys.index(i) for i in optimal_children]
            other_children = [i for i in range(poly_res.latent_var.shape[2]) if i not in optimal_children_poly_indices]
            poly_res.latent_var[:, parent_idx, other_children] = float("-inf")

        # Handle resolver indices
        half_bs = X.shape[0]//2
        result = [(x, i) for i, x in enumerate(poly_res.resolver_indices) if x not in known_indices]
        if result:
            unknown_resolver_indices, mask = zip(*result)
            resolved_indices = vutil.repeat_n(poly_res.resolver_labeling[:,mask], half_bs)
            resolved_indices[resolved_indices == 0] = float('-inf')
            X[:half_bs,:,[x-1 for x in unknown_resolver_indices]] = resolved_indices
            
        poly_res.latent_var.requires_grad = True

    # Handle fixed labeling and matrix reduction
    fixed_labeling, old_to_new = None, None
    if known_indices:
        if v_solver.config['collapse_nodes']:
            # Identify which nodes to keep vs remove
            nodes_to_remove = set([x for x in known_indices if x not in optimal_root_nodes])
            # Create new T with reduced dimensions
            _T, old_to_new = _remove_nodes_from_T(T, nodes_to_remove)
            # Don't reformat G if we've already done it in a previous instantiation
            if v_solver.G != None and v_solver.G.shape[0] != _T.shape[1]: 
                G = v_solver.G
                G = G[torch.tensor([i for i in range(G.size(0)) if i not in nodes_to_remove], device=G.device)]
                G = G[:, torch.tensor([i for i in range(G.size(1)) if i not in nodes_to_remove], device=G.device)]
                v_solver.G = G
            v_solver.T = _T[0]
        
        # Update X tensor
        unknown_indices = [x for x in range(v_solver.num_nodes_to_label+1) if x not in known_indices and x != 0]
        known_labelings = torch.stack(known_labelings, dim=1)
        X = X[:,:,[x-1 for x in unknown_indices]]
        #print("num known indices", len(known_indices), "num unknown indices", len(unknown_indices))
        
        fixed_labeling = vutil.FixedVertexLabeling(known_indices, unknown_indices, known_labelings, 
                                                   optimal_root_nodes, old_to_new)

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

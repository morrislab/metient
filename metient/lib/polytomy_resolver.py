import torch
from collections import OrderedDict
from metient.util import vertex_labeling_util as vutil
import copy

class PolytomyResolver():

    def __init__(self, v_optimizer, nodes_w_polys, resolver_sites):
        '''
        This is post U matrix estimation, so T already has leaf nodes.
        '''
        
        # 1. Pad the adjacency matrix so that there's room for the new resolver nodes
        # nodes_w_polys are the nodes that have polytomies
        # (we place them in this order: given internal nodes, new resolver nodes, leaf nodes from U)
        T, G = v_optimizer.T, v_optimizer.G

        # TODO: Handle sparse T and G
        # Convert T to dense if it's sparse
        if T.is_sparse:
            T = T.to_dense()

        num_new_nodes = 0
        for r in resolver_sites:
            num_new_nodes += len(r)

        num_leaves = v_optimizer.L.shape[1]
        num_internal_nodes = T.shape[0]-num_leaves
        T = torch.nn.functional.pad(T, pad=(0, num_new_nodes, 0, num_new_nodes), mode='constant', value=0)
        # 2. Shift T and G to make room for the new indices (so the order is input internal nodes, new poly nodes, leaves)
        idx1 = num_internal_nodes
        idx2 = num_internal_nodes+num_leaves
        T = torch.cat((T[:,:idx1], T[:,idx2:], T[:,idx1:idx2]), dim=1)
        if G != None:
            G = torch.nn.functional.pad(G, pad=(0, num_new_nodes, 0, num_new_nodes), mode='constant', value=0)
            G = torch.cat((G[:,:idx1], G[:,idx2:], G[:,idx1:idx2]), dim=1)

        # Shift the leaf node indices in node_collection too
        # This has to be done in descending order otherwise the shifts just overwrite each other
        node_collection = v_optimizer.node_collection
        nodes = node_collection.get_nodes()
        leaf_nodes = [node for node in nodes if node.is_witness]
        leaf_nodes_descending_order = sorted(leaf_nodes, key=lambda obj: obj.idx, reverse=True)
        for node in leaf_nodes_descending_order:
            node_collection.update_index(node.idx, node.idx+num_new_nodes)

        # 3. Get each polytomy's children (these are the positions we have to relearn)
        children_of_polys = vutil.get_child_indices(T, nodes_w_polys)

        # 4. Initialize a matrix to learn the polytomy structure
        ret = initialize_polytomy_resolver_adj_matrix(T, children_of_polys, num_internal_nodes, 
                                                      num_new_nodes, v_optimizer, nodes_w_polys, resolver_sites)
        poly_adj_matrix, nodes_w_polys_to_resolver_indices, resolver_indices, resolver_labeling = ret

        # 5. Initialize potential new nodes as children of the polytomy nodes
        for i in nodes_w_polys:
            for j in nodes_w_polys_to_resolver_indices[i]:
                T[i,j] = 1.0
                parent_node = node_collection.get_node(i)
                new_node = vutil.MigrationHistoryNode(j, [f"{i}pol{j}"]+parent_node.label, is_witness=False, is_polytomy_resolver_node=True)
                node_collection.add_node(new_node)
                if G != None:
                    G[i,j] = v_optimizer.config['identical_clone_gen_dist']

        # 6. The genetic distance between a new node and its potential
        # new children which "switch" is the same distance between the new
        # node's parent and the child
        resolver_index_to_parent_idx = {}
        for poly_node in nodes_w_polys_to_resolver_indices:
            new_nodes = nodes_w_polys_to_resolver_indices[poly_node]
            for new_node_idx in new_nodes:
                resolver_index_to_parent_idx[new_node_idx] = poly_node

        if G != None:
            for new_node_idx in resolver_indices:
                parent_idx = resolver_index_to_parent_idx[new_node_idx]
                potential_child_indices = vutil.get_child_indices(T, [parent_idx])
                for child_idx in potential_child_indices:
                    G[new_node_idx, child_idx] = G[parent_idx, child_idx]
        v_optimizer.T = T
        v_optimizer.G = G
        self.latent_var = poly_adj_matrix
        self.nodes_w_polys = nodes_w_polys
        self.children_of_polys = children_of_polys
        self.resolver_indices = resolver_indices
        self.resolver_index_to_parent_idx = resolver_index_to_parent_idx
        self.resolver_labeling = resolver_labeling

    def _update_indices_with_mapping(self, indices, mapping):
        """Helper function to update indices using a mapping dictionary.
        
        Args:
            indices: List of indices to update
            mapping: Dictionary mapping old indices to new indices
        
        Returns:
            List of updated indices, using original value if not in mapping
        """
        if indices is None:
            return None
        return [mapping[x] if x in mapping else x for x in indices]

    def update_indices(self, old_to_new):
        """Update resolver mappings when nodes are added back to the tree.
        
        Args:
            old_to_new (dict): Mapping from original node indices to reduced tree indices
        """
        # Update nodes with polytomies
        self.nodes_w_polys = self._update_indices_with_mapping(self.nodes_w_polys, old_to_new)
        
        # Update children of polytomies
        self.children_of_polys = self._update_indices_with_mapping(self.children_of_polys, old_to_new)

        # Update resolver indices 
        self.resolver_indices = self._update_indices_with_mapping(self.resolver_indices, old_to_new)
        
        # Update resolver index to parent mapping
        if self.resolver_index_to_parent_idx is not None:
            new_mapping = {}
            for resolver_idx, parent_idx in self.resolver_index_to_parent_idx.items():
                new_resolver_idx = old_to_new[resolver_idx] if resolver_idx in old_to_new else resolver_idx
                new_parent_idx = old_to_new[parent_idx] if parent_idx in old_to_new else parent_idx
                new_mapping[new_resolver_idx] = new_parent_idx
            self.resolver_index_to_parent_idx = new_mapping

def initialize_polytomy_resolver_adj_matrix(T, children_of_polys, num_internal_nodes, 
                                            num_new_nodes, v_optimizer, nodes_w_polys, resolver_sites):
    num_nodes_full_tree = T.shape[0]
    bs = v_optimizer.config['sample_size']
    poly_adj_matrix = vutil.repeat_n(torch.ones((num_nodes_full_tree, len(children_of_polys)), dtype=torch.float32), bs)
    resolver_indices = [x for x in range(num_internal_nodes, num_internal_nodes+num_new_nodes)]

    nodes_w_polys_to_resolver_indices = OrderedDict()
    start_new_node_idx = resolver_indices[0]
    for parent_idx, r in zip(nodes_w_polys, resolver_sites):
        num_new_nodes_for_poly = len(r)
        if parent_idx not in nodes_w_polys_to_resolver_indices:
            nodes_w_polys_to_resolver_indices[parent_idx] = []

        for i in range(start_new_node_idx, start_new_node_idx+num_new_nodes_for_poly):
            nodes_w_polys_to_resolver_indices[parent_idx].append(i)
        start_new_node_idx += num_new_nodes_for_poly

    resolver_labeling = torch.zeros(v_optimizer.num_sites, len(resolver_indices))
    t = 0
    for sites in resolver_sites:
        for site in sites:
            resolver_labeling[site, t] = 1
            t += 1

    offset = 0
    for parent_idx in nodes_w_polys:
        child_indices = vutil.get_child_indices(T, [parent_idx])
        # make the children of polytomies start out as children of their og parent
        # with the option to "switch" to being the child of the new poly node
        poly_adj_matrix[:,parent_idx,offset:(offset+len(child_indices))] = 1.0
        # we only want to let these children choose between being the child
        # of their original parent or the child of this new poly node, which
        # we can do by setting all other indices to -inf
        mask = torch.ones(num_nodes_full_tree, dtype=torch.bool)
        new_nodes = nodes_w_polys_to_resolver_indices[parent_idx]
        mask_indices = new_nodes + [parent_idx]
        mask[[mask_indices]] = 0
        poly_adj_matrix[:,mask,offset:(offset+len(child_indices))] = float("-inf")
        offset += len(child_indices)

    poly_adj_matrix.requires_grad = True

    return poly_adj_matrix, nodes_w_polys_to_resolver_indices, resolver_indices, resolver_labeling
    
def should_remove_node(poly_res, V, T, remove_idx, children_of_removal_node):
    '''

    Returns True if migration graph is the same or better after 
    removing node at index remove_idx

    If any of the following are true:
        (1) the polytomy resolver node is the same color as its parent ,
        (2) the polytomy resolver node only has one child that is the same color as it,
    the migration history won't change by removing the polytomy resolver node. 

    If:
        (3) the polytomy resolver node is the only child of its parent, it's not resolving any polytomies
    
    '''
    parent_idx = poly_res.resolver_index_to_parent_idx[remove_idx]
    remove_idx_color = torch.argmax(V[:,remove_idx]).item()
    is_same_color_as_parent = torch.argmax(V[:,parent_idx]).item() == remove_idx_color
    is_same_color_as_child = torch.argmax(V[:,children_of_removal_node[0]]).item() == remove_idx_color
    # print(parent_idx, remove_idx_color, is_same_color_as_parent, is_same_color_as_child)
    # Case 1
    if is_same_color_as_parent:
        return True
    
    # Case 2
    if len(children_of_removal_node)== 1 and (is_same_color_as_child):
        return True
    
    # Case 3
    children_of_parent_node = vutil.get_child_indices(T, parent_idx)
    if len(children_of_parent_node) == 1:
        return True
    
    return False

def remove_nodes_from_adjacency_matrix_sparse(adj_matrix, removal_nodes):
    '''
    Reindex the adjacency matrix after removing nodes.
    '''
    # Extract indices and values from the sparse adjacency matrix
    adj_matrix = adj_matrix.coalesce()
    indices = adj_matrix.indices()  # Shape: [2, nnz]
    values = adj_matrix.values()   # Shape: [nnz]

    # Convert removal_nodes to a set for fast lookup
    removal_nodes_set = set(removal_nodes)

    # Mask for edges where the target node is in removal_nodes (parents)
    parent_mask = torch.isin(indices[1], torch.tensor(removal_nodes, device=adj_matrix.device))
    parent_nodes = indices[0, parent_mask]
    removal_targets = indices[1, parent_mask]  # The nodes being removed (targets)

    # Mask for edges where the source node is in removal_nodes (children)
    child_mask = torch.isin(indices[0], torch.tensor(removal_nodes, device=adj_matrix.device))
    child_nodes = indices[1, child_mask]
    removal_sources = indices[0, child_mask]  # The nodes being removed (sources)

    # Map each removal node to its parents and children
    removal_to_parents = {}
    removal_to_children = {}

    for target, parent in zip(removal_targets.tolist(), parent_nodes.tolist()):
        if target in removal_to_parents:
            removal_to_parents[target].append(parent)
        else:
            removal_to_parents[target] = [parent]

    for source, child in zip(removal_sources.tolist(), child_nodes.tolist()):
        if source in removal_to_children:
            removal_to_children[source].append(child)
        else:
            removal_to_children[source] = [child]

    # Build new connections: parents -> children for each removal node
    new_edges = []
    for removal_node in removal_nodes_set:
        parents = removal_to_parents.get(removal_node, [])
        children = removal_to_children.get(removal_node, [])

        # Add connections between each parent and child
        for parent in parents:
            for child in children:
                if parent != child:  # Avoid self-loops
                    new_edges.append([parent, child])

    # Convert new edges to a tensor
    new_edges = torch.tensor(new_edges, device=adj_matrix.device).T if new_edges else torch.empty((2, 0), dtype=torch.long, device=adj_matrix.device)
    all_edges = torch.cat((indices, new_edges), dim=1) if new_edges.size(1) > 0 else indices

    # Remove duplicates
    all_edges = torch.unique(all_edges, dim=1)

    # Remove edges that involve removal nodes
    valid_mask = ~(torch.isin(all_edges[0], torch.tensor(removal_nodes, device=adj_matrix.device)) |
                   torch.isin(all_edges[1], torch.tensor(removal_nodes, device=adj_matrix.device)))
    all_edges = all_edges[:, valid_mask]

    # Reindex the nodes
    num_nodes = adj_matrix.size()[0]
    new_index_mapping = {}
    new_index = 0
    for old_index in range(num_nodes):
        if old_index not in removal_nodes_set:
            new_index_mapping[old_index] = new_index
            new_index += 1

    # Map old indices to new indices in edges
    reindexed_edges = torch.stack([
        torch.tensor([new_index_mapping[node.item()] for node in all_edges[0]], device=adj_matrix.device),
        torch.tensor([new_index_mapping[node.item()] for node in all_edges[1]], device=adj_matrix.device)
    ])

    # Create the reindexed sparse adjacency matrix
    new_size = (new_index, new_index)  # The size reduces after removal
    new_values = torch.ones(reindexed_edges.size(1), dtype=values.dtype, device=adj_matrix.device)
    new_adj_matrix = torch.sparse_coo_tensor(reindexed_edges, new_values, size=new_size, dtype=adj_matrix.dtype, device=adj_matrix.device)

    return new_adj_matrix


def remove_nodes_from_adjacency_matrix(T, removal_nodes):
    """
    Reindex the adjacency matrix after removing nodes.
    
    Args:
        T (torch.sparse_coo_tensor): The sparse adjacency matrix.
        remove_nodes (list): A list of node indices to be removed.

    Returns:
        torch.sparse_coo_tensor: The reindexed sparse adjacency matrix.
        dict: Mapping of old indices to new indices.
    """
    if T.is_sparse:
        return remove_nodes_from_adjacency_matrix_sparse(T, removal_nodes)

    T = T.clone().detach()
    # Attach children of the node to remove to their original parent
    for remove_idx in removal_nodes:
        parent_idx = torch.where(T[:,remove_idx] > 0)[0][0]
        child_indices = vutil.get_child_indices(T, [remove_idx])
        for child_idx in child_indices:
            T[parent_idx,child_idx] = 1.0

    # Get the device of the input tensor
    device = T.device
    # Create indices on the same device as adj_matrix
    keep_indices = torch.tensor([i for i in range(T.size(0)) if i not in removal_nodes], device=device)
    # Remove rows of T
    T = T[keep_indices]
    # Remove columns of T
    T = T[:, keep_indices]
    return T


def remove_nodes(removal_indices, V, T, G, node_collection):
    '''
    Remove polytomy resolver nodes from V, T, G and node_idx_to_label
    if they didn't actually help
    '''

    T = remove_nodes_from_adjacency_matrix(T, removal_indices)
    # Remove columns from V
    V = V[:, torch.tensor([i for i in range(V.size(1)) if i not in removal_indices], device=V.device)]

    if G != None: 
        G = G[torch.tensor([i for i in range(G.size(0)) if i not in removal_indices], device=G.device)]
        G = G[:, torch.tensor([i for i in range(G.size(1)) if i not in removal_indices], device=G.device)]
        
    # Reindex the idx to label dict
    node_collection.remove_indices_and_reindex(removal_indices)
    return V, T, G, node_collection

def remove_extra_resolver_nodes(solution_set, poly_res, weights, O, p):
    '''
    If there are any resolver nodes that were added to resolve polytomies but they 
    weren't used (i.e. 1. they have no children or 2. they don't change the 
    migration history), remove them
    '''

    if poly_res == None:
        return solution_set

    for i,soln in enumerate(solution_set):
        modified_soln = None
        
        V, T = soln.V, soln.T
        nodes_to_remove = []
        for new_node_idx in poly_res.resolver_indices:
            children_of_new_node = vutil.get_child_indices(T, new_node_idx)
            if len(children_of_new_node) == 0:
                nodes_to_remove.append(new_node_idx)
            elif should_remove_node(poly_res, V, T, new_node_idx, children_of_new_node):
                nodes_to_remove.append(new_node_idx)
        if len(nodes_to_remove) != 0:
            new_node_collection = copy.deepcopy(soln.node_collection)
            new_V, new_T, new_G, new_node_collection = remove_nodes(nodes_to_remove, V, T, soln.G, new_node_collection)
            loss, new_metrics = vutil.clone_tree_labeling_objective(new_V, soln.soft_V, new_T, new_G, O, p, weights, True)
            modified_soln = vutil.VertexLabelingSolution(loss,*new_metrics,new_V,soln.soft_V,new_T,new_G,new_node_collection)
            solution_set[i] = modified_soln
    
    return
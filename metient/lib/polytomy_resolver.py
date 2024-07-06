import torch
from collections import OrderedDict
from metient.util import vertex_labeling_util as vutil
#from metient.lib.v_optimzer import VertexLabelingSolver

import numpy as np

class PolytomyResolver():

    def __init__(self, v_optimizer, nodes_w_polys, resolver_sites):
        '''
        This is post U matrix estimation, so T already has leaf nodes.
        '''
        
        # 1. Pad the adjacency matrix so that there's room for the new resolver nodes
        # nodes_w_polys are the nodes that have polytomies
        # (we place them in this order: given internal nodes, new resolver nodes, leaf nodes from U)
        T, G = v_optimizer.T, v_optimizer.G
        bs = v_optimizer.config['batch_size']

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

        # 3. Get each polytomy's children (these are the positions we have to relearn)
        children_of_polys = vutil.get_child_indices(T, nodes_w_polys)

        # 4. Initialize a matrix to learn the polytomy structure
        num_nodes_full_tree = T.shape[0]
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
        
        # 5. Initialize potential new nodes as children of the polytomy nodes
        for i in nodes_w_polys:
            for j in nodes_w_polys_to_resolver_indices[i]:
                T[i,j] = 1.0
                v_optimizer.node_idx_to_label[j] = [f"{i}pol{j}"]
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

def is_same_mig_hist_with_node_removed(poly_res, T, num_internal_nodes, V, remove_idx, children_of_removal_node, p, prev_m, prev_c, prev_s):
    '''
    Returns True if migration graph is the same or better after 
    removing node at index remove_idx
    '''

    '''
    If any of the following are true:
        (1) the polytomy resolver node is the same color as its parent,
        (2) the polytomy resolver node only has one child that is the same color as it,

        # (1) the polytomy resolver node is the same color as its parent and only has one child,
        # (2) the polytomy resolver node only has one child that is the same color as it,
        # (3) the polytomy resolver node is the same color as its parent and all of its children (that are internal nodes) are the same color
        # (4) the polytomy resolver node is the same color as its parent and only one of its children is a different color
    the migration history won't change by removing the polytomy resolver node. 
    
    If that is not true, check to see if the migration history changes by removing the node
    '''
    parent_idx = poly_res.resolver_index_to_parent_idx[remove_idx]
    # print(remove_idx, parent_idx, torch.argmax(V[:,parent_idx]).item(),torch.argmax(V[:,remove_idx]).item(), torch.argmax(V[:,parent_idx]).item()==torch.argmax(V[:,remove_idx]).item())
    remove_idx_color = torch.argmax(V[:,remove_idx]).item()
    is_same_color_as_parent = torch.argmax(V[:,parent_idx]).item() == remove_idx_color
    is_same_color_as_child = torch.argmax(V[:,children_of_removal_node[0]]).item() == remove_idx_color
    # Case 1
    if is_same_color_as_parent:
        return True
    
    # Case 2
    if len(children_of_removal_node)==1 and (is_same_color_as_child):
        return True
    
    # is_same_color_as_all_children = True
    # num_children_diff_color = 0
    # for child in children_of_removal_node:
    #     is_same_color_as_child = torch.argmax(V[:,child]).item() == remove_idx_color
    #     # Is an internal node child and is not same color
    #     if not is_same_color_as_child and child < num_internal_nodes:
    #         is_same_color_as_all_children = False
    #     if not is_same_color_as_child:
    #         num_children_diff_color += 1
    # # Case 3
    # if is_same_color_as_parent and is_same_color_as_all_children:
    #     return True
    # # Case 4
    # if is_same_color_as_parent and num_children_diff_color == 1:
    #     return True
    return False

    candidate_T, candidate_V = T.detach().clone(), V.detach().clone()
    # Attach children of the node to remove back to their original parent
    for child_idx in children_of_removal_node:
        candidate_T[parent_idx,child_idx] = 1.0
    candidate_T = np.delete(candidate_T, remove_idx, 0)
    candidate_T = np.delete(candidate_T, remove_idx, 1)
    candidate_V = np.delete(candidate_V, remove_idx, 1)
    new_m, new_c, new_s, _, _ = vutil.ancestral_labeling_metrics(vutil.add_batch_dim(candidate_V), candidate_T, None, None, p, True)
    # print(prev_m, prev_c, prev_s, new_m, new_c, new_s, ((prev_m >= int(new_m)) and (prev_c >= int(new_c)) and (prev_s >= int(new_s))))
    return ((prev_m >= int(new_m)) and (prev_c >= int(new_c)) and (prev_s >= int(new_s)))

def remove_nodes(removal_indices, V, T, G, node_idx_to_label):
    '''
    Remove polytomy resolver nodes from V, T, G and node_idx_to_label
    if they didn't actually help
    '''
    T = T.clone().detach()
    V = V.clone().detach()
    
    # Attach children of the node to remove to their original parent
    for remove_idx in removal_indices:
        parent_idx = np.where(T[:,remove_idx] > 0)[0][0]
        child_indices = vutil.get_child_indices(T, [remove_idx])
        for child_idx in child_indices:
            T[parent_idx,child_idx] = 1.0
    # Remove indices from T, V and G
    T = np.delete(T, removal_indices, 0)
    T = np.delete(T, removal_indices, 1)
    V = np.delete(V, removal_indices, 1)

    if G != None: 
        G = G.clone().detach()
        G = np.delete(G, removal_indices, 0)
        G = np.delete(G, removal_indices, 1)

    # Reindex the idx to label dict
    new_node_idx_to_label, _= vutil.reindex_dict(node_idx_to_label, removal_indices)
    
    return V, T, G, new_node_idx_to_label

def remove_extra_resolver_nodes(solution_set, num_internal_nodes, poly_res, weights, O, p):
    '''
    If there are any resolver nodes that were added to resolve polytomies but they 
    weren't used (i.e. 1. they have no children or 2. they don't change the 
    migration history), remove them
    '''

    if poly_res == None:
        return solution_set
    # prev_ms, prev_cs, prev_ss, _, _ = vutil.ancestral_labeling_metrics(vutil.to_tensor(best_Vs), vutil.to_tensor(best_Ts), None, None, p, True)
    out_solution_set = []
    for soln in solution_set:
        V, T, prev_m, prev_c, prev_s = soln.V, soln.T, soln.m, soln.c, soln.s
        nodes_to_remove = []
        for new_node_idx in poly_res.resolver_indices:
            children_of_new_node = vutil.get_child_indices(T, [new_node_idx])
            if len(children_of_new_node) == 0:
                nodes_to_remove.append(new_node_idx)
            elif is_same_mig_hist_with_node_removed(poly_res, T, num_internal_nodes,  V, new_node_idx, children_of_new_node, p, prev_m, prev_c, prev_s):
                nodes_to_remove.append(new_node_idx)

        if len(nodes_to_remove) != 0:
            new_V, new_T, new_G, new_node_idx_to_label = remove_nodes(nodes_to_remove, V, T, soln.G, soln.idx_to_label)
            loss, (new_m,new_c,new_s) = vutil.clone_tree_labeling_objective(new_V, soln.soft_V, new_T, new_G, O, p, weights, True)
            out_solution_set.append(vutil.VertexLabelingSolution(loss,new_m,new_c,new_s,new_V,soln.soft_V,new_T,new_G,new_node_idx_to_label))
        else:
            out_solution_set.append(soln)

    return out_solution_set
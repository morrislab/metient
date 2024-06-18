from heapq import heapify, heappush, heappushpop, nlargest

class MinHeap():
    def __init__(self, k):
        self.h = []
        self.length = k
        self.items = set()
        heapify(self.h)
        
    def add(self, loss, A, V, soft_V, i=0):
        # Maintain a max heap so that we can efficiently 
        # get rid of larger loss value Vs
        tree = vutil.LabeledTree(A, V)
        if (len(self.h) < self.length) and (tree not in self.items): 
            self.items.add(tree)
            heappush(self.h, VertexLabelingSolution(loss, V, soft_V, i))
        # If loss is greater than the max loss we
        # already have, don't bother adding this 
        # solution (hash checking below is expensive)
        elif loss > self.h[0].loss:
            return
        # If we've reached capacity, push the new
        # item and pop off the max item
        elif tree not in self.items:
            self.items.add(tree)
            removed = heappushpop(self.h, VertexLabelingSolution(loss, V, soft_V, i))
            removed_tree = vutil.LabeledTree(A, removed.V)
            self.items.remove(removed_tree)
        
    def get_top(self):
        # due to override in comparison operator, this
        # actually returns the n smallest values
        return nlargest(self.length, self.h)


# mig_vec = get_mig_weight_vector(batch_size, input_weights)
# seed_vec = get_seed_site_weight_vector(batch_size, input_weights)
# for sln in final_solutions:
#     print(sln.loss, sln.i, mig_vec[sln.i], seed_vec[sln.i])
# with open(os.path.join(output_dir, f"{run_name}.txt"), 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(file_output)
# with open(os.path.join(output_dir, f"{run_name}_best_weights.txt"), 'w', newline='') as file:
#     file.write(f"{best_pars_weights[0]}, {best_pars_weights[1]}")


### Used to save tutorial inputs
from metient.util import data_extraction_util as dutil
import numpy as np

def get_pyclone_conipher_clusters(df, clusters_tsv_fn):
    pyclone_df = pd.read_csv(clusters_tsv_fn, delimiter="\t")
    mut_name_to_cluster_id = dict()
    cluster_id_to_mut_names = dict()
    # 1. Get mapping between mutation names and PyClone cluster ids
    for _, row in df.iterrows():
        mut_items = row['character_label'].split(":")
        cluster_id = pyclone_df[(pyclone_df['CHR']==int(mut_items[1]))&(pyclone_df['POS']==int(mut_items[2]))&(pyclone_df['REF']==mut_items[3])]['treeCLUSTER'].unique()
        assert(len(cluster_id) <= 1)
        if len(cluster_id) == 1:
            cluster_id = int(cluster_id.item())
            mut_name_to_cluster_id[row['character_label']] = cluster_id
            if cluster_id not in cluster_id_to_mut_names:
                cluster_id_to_mut_names[cluster_id] = set()
            else:
                cluster_id_to_mut_names[cluster_id].add(row['character_label'])
       
    # 2. Set new names for clustered mutations
    cluster_id_to_cluster_name = {k:";".join(list(v)) for k,v in cluster_id_to_mut_names.items()}
    return cluster_id_to_cluster_name, mut_name_to_cluster_id
    
import pandas as pd
patients = ["CRUK0003", "CRUK0010", "CRUK0013", "CRUK0029" ]
tracerx_dir = os.path.join(os.getcwd(), "metient", "data", "tracerx_nsclc")
for patient in patients:
    df = pd.read_csv(os.path.join(tracerx_dir, "patient_data", f"{patient}_SNVs.tsv"), delimiter="\t", index_col=0)
    cluster_id_to_cluster_name, mut_name_to_cluster_id = get_pyclone_conipher_clusters(df, os.path.join(tracerx_dir, 'conipher_outputs', 'TreeBuilding', f"{patient}_conipher_SNVstreeTable_cleaned.tsv"))
    df['var_read_prob'] = df.apply(lambda row: dutil.calc_var_read_prob(row['major_cn'], row['minor_cn'], row['purity']), axis=1)
    df['site_category'] = df.apply(lambda row: 'primary' if 'primary' in row['anatomical_site_label'] else 'metastasis', axis=1)
    df['cluster_index'] = df.apply(lambda row: mut_name_to_cluster_id[row['character_label']] if row['character_label'] in mut_name_to_cluster_id else np.nan, axis=1)
    df['character_label'] = df.apply(lambda row: row['character_label'].split(":")[0], axis=1)
    df = df.dropna(subset=['cluster_index'])
    df = df[['anatomical_site_index', 'anatomical_site_label', 'cluster_index', 'character_label',
             'ref', 'var', 'var_read_prob', 'site_category']]
    print(df['cluster_index'].unique())
    df['cluster_index'] = df['cluster_index'].astype(int)
    print(df['cluster_index'].unique())
    df.to_csv(os.path.join(os.getcwd(), "metient", "data", "tutorial","inputs", f"{patient}_SNVs.tsv"), sep="\t", index=False)

    # TODO: What should the genetic distance be??
    # resolver_indices = poly_res.resolver_indices
    # for batch_idx in range(T.shape[0]):
    #     for res_idx in resolver_indices:
    #         parent_idx = poly_res.resolver_index_to_parent_idx[res_idx]
    #         res_children = vutil.get_child_indices(T[batch_idx], [res_idx])
    #         # no children for this resolver node, so keep the original branch length
    #         if len(res_children) == 0:
    #             avg = G[parent_idx, res_idx]
    #         else:
    #             avg = torch.mean(G[parent_idx, res_children])
    #         full_G[batch_idx, parent_idx, res_idx] = avg
    # LAST_G = full_G

# Sampling with replacement the good samples during training
# if  i < max_iter*1/2:
    #     with torch.no_grad():
    #         indices = np.argsort(v_losses.detach().numpy())
    #         best_indices = indices[:int(batch_size/2)]
    #         samples = np.random.choice(best_indices, size=batch_size, replace=True)   
    #         X.grad = X.grad[samples,:,:]
    #         X = X[samples,:,:] # TODO: if using momentum does this screw anything up
            
    #         X.requires_grad = True
    #         print(v_losses, indices, best_indices, samples)

# LBFGS optimization
# def step_closure():
#     v_optimizer.zero_grad()
#     updated_Vs, v_losses, loss_comps, updated_soft_Vs, updated_Ts = closure(X,  L=L, T=T, p=p, G=G, O=O, v_temp=v_temp, t_temp=t_temp, hard=hard, weights=weights, i=i, max_iter=max_iter, poly_res=poly_res)
#     mean_loss = torch.mean(v_losses)
#     mean_loss.backward()
#     nonlocal Vs, soft_Vs, Ts
#     Vs = updated_Vs
#     soft_Vs = updated_soft_Vs
#     Ts = updated_Ts
#     return mean_loss
# mean_loss = v_optimizer.step(step_closure)
# # Define the closure for LBFGS
#     def closure(X, **kwargs):
#         L, T, p, G, O, v_temp, t_temp, hard, weights, i, max_iter, poly_res = kwargs.values()
#         Vs, v_losses, loss_comps, soft_Vs, Ts = compute_v_loss(X, L, T, p, G, O, v_temp, t_temp, hard, weights, i, max_iter, poly_res)
#         return Vs, v_losses, loss_comps, soft_Vs, Ts

# This was in init_X

# For clone tree leaf nodes that are only observed in one site,
# we don't need to learn their labeling (the label should be the
# site it's detected in)
# known_indices = []
# known_labelings = []
# for node_idx, sites in zip(nodes_w_children, biased_sites):
#     if node_idx == 0:
#         continue # we know the root labeling
#     idx = node_idx - 1
#     if len(sites) == 1 and input_T[node_idx].sum() == 0:
#         known_indices.append(node_idx - 1) # indexing w.r.t X, which doesn't include root node
#         known_labelings.append(torch.eye(num_sites)[sites[0]].T)
# known_labelings = known_labelings
# unknown_indices = [x for x in range(num_nodes_to_label) if x not in known_indices]
# #print("known", known_indices, "unknown", unknown_indices,known_labelings)
# X = X[:,:, unknown_indices] # only include the unknown indices for inference

def bottom_up(T, X, U, p, polyres):
    _, k = X.max(1)
    X_hard = torch.zeros(X.size()).scatter_(1, torch.unsqueeze(k, 1), 1.0)
    full_labeling = stack_vertex_labeling(U, X_hard, p, polyres)
    print(T.shape, X.shape)
    print_idx = 0
    indices_of_intereset = [18,22,26,12]
    print(X[print_idx,:,indices_of_intereset])
    print(X_hard[print_idx,:,indices_of_intereset])
    normal_indices = [i+1 for i in indices_of_intereset]
    print(full_labeling[print_idx,:,normal_indices])

    _, full_children_sites = torch.max(full_labeling, 1)

    # Need a list of tuples to index X, which are (batch_num, child_site, parent_node_idx)
    # At these positions, we will increase the probability that the parent node
    # gets assigned to the same site as its children
    lst = np.where(T!=0)
    positions = list(zip(lst[0], lst[2], lst[1]))
    X_positions = []
    for pos in positions:
        if pos[2] == 0: # we already know the root node index
            continue
        if pos[2]-1 in indices_of_intereset:
            print((pos[0], full_children_sites[pos[0], pos[1]], pos[2]-1))
        X_positions.append((pos[0], full_children_sites[pos[0], pos[1]], pos[2]-1))
    print(positions[:10])
    print(X_positions[:10])
    with torch.no_grad():
        X[np.array(X_positions).T] += 5
        # for i in range(T.shape[0]):
        #     _, all_children_sites = torch.max(full_labeling[i], 0)
        #     for node_idx in range(T[i].shape[0]):
        #         if node_idx == 0:
        #             continue # we know the root labeling
        #         parents, children = vutil.find_parents_children(T[i], node_idx)
        #         if len(children) > 0:
        #             idx = node_idx - 1 # no root
        #             # if i == print_idx:
        #             #     # Get the list of all children
        #             #     children_list = [torch.nonzero(row).squeeze().tolist() for row in T[i]]
        #             #     # Get the list of all parents
        #             #     parents_list = [torch.nonzero(col).squeeze().tolist() for col in T[i].t()]
        #             #     #print(children_list)
        #             #     #print("node_idx", node_idx, "parents", parents, "children", children, "this node's children sites", all_children_sites[children])
                    
        #             X[i,all_children_sites[children],idx] += 5
    print("new X", X[print_idx,:,indices_of_intereset])
    return X

# old pareto front calculation
# pattern_to_best_pars_sum = {p:float("inf") for p in set(all_patterns)}
# # Find the best parsimony sum per unique pattern
# # TODO: should this take the best comigrations, seeding sites, and migrations?
# best_overall_sum = float("inf")
# for i, (pars_metrics, pattern) in enumerate(zip(all_pars_metrics, all_patterns)):
#     pars_sum = sum(pars_metrics)
#     if pars_sum < best_overall_sum:
#         best_overall_sum = pars_sum
#     if pars_sum < pattern_to_best_pars_sum[pattern]:
#         pattern_to_best_pars_sum[pattern] = pars_sum
# print("pattern_to_best_pars_sum", pattern_to_best_pars_sum)
# # Find all pars metrics combinations that match the best sum
# best_pars_metrics = set()
# for i, pars_metrics in enumerate(all_pars_metrics):
#     pars_sum = sum(pars_metrics)
#     if pars_sum == best_overall_sum:
#         best_pars_metrics.add(pars_metrics)
# print("best_pars_metrics", best_pars_metrics)

# pruned_solutions = []
# # Go through and prune any solutions that are worse than the best sum for the
# # same pattern or made any mistakes
# for soln, pars_metrics, pattern in zip(solutions, all_pars_metrics, all_patterns):
#     # print(pattern, pars_metrics)
#     # print("made mistake", made_mistake(soln, U), "keep tree", keep_tree(pars_metrics, pattern, pattern_to_best_pars_sum, best_overall_sum, best_pars_metrics))
#     if keep_tree(pars_metrics, pattern, pattern_to_best_pars_sum, best_overall_sum, best_pars_metrics) and not made_mistake(soln, num_internal_nodes):
#         pruned_solutions.append(soln)
# if len(pruned_solutions) == 0: 
#     print("No solutions without mistakes detected")
#     # ideally this doesn't happen, but remove mistake detection so 
#     # that we return some results
#     for soln, pars_metrics, pattern in zip(solutions, all_pars_metrics, all_patterns):
#         if keep_tree(pars_metrics, pattern, pattern_to_best_pars_sum, best_overall_sum, best_pars_metrics):
#             pruned_solutions.append(soln)

# No longer needed utilities with normal pareto front calculation
def made_mistake(solution, num_internal_nodes):
    V = solution.V
    A = solution.T
    VA = V @ A
    Y = torch.sum(torch.mul(VA.T, 1-V.T), axis=1) # Y has a 1 for every node where its parent has a diff color
    nonzero_indices = torch.nonzero(Y).squeeze()

    if nonzero_indices.dim() == 0:
        return False
    for mig_node in nonzero_indices:
        # it's a leaf node itself!
        if mig_node > (num_internal_nodes-1):
            continue
        if not vutil.has_leaf_node(A, int(mig_node), num_internal_nodes):
            return True
    return False
    
def keep_tree(cand_metric, pattern, pattern_to_best_pars_sum, best_overall_sum, best_pars_metrics):
    # Don't keep a tree if there is another solution with the same seeding pattern
    # but a more parsimonious result
    #print(cand_metric)
    if sum(cand_metric) != pattern_to_best_pars_sum[pattern]:
        return False
    
    if sum(cand_metric) == best_overall_sum:
        return True

    # Don't keep any trees that are strictly worse than the best_pars_metrics
    for best_metric in best_pars_metrics:
        if cand_metric[0] < best_metric[0] or cand_metric[1] < best_metric[1] or cand_metric[2] < best_metric[2]:
            return True
    return False

def map_pattern(V, T):
    # Remove "monoclonal" or "polyclonal"
    pattern = " ".join(putil.get_verbose_seeding_pattern(V,T).split()[1:])
    # For the purposes of getting a representative set of trees, we treat
    # reseeding as a subset of multi-source, but don't distinguish them
    if pattern != "primary single-source seeding":
        pattern = "not primary single-source seeding"
    return pattern



def get_random_vals_fixed_seeds(shape):
    global RANDOM_VALS
    if RANDOM_VALS != None and shape in RANDOM_VALS:
        return RANDOM_VALS[shape]

    if RANDOM_VALS == None:
        RANDOM_VALS = dict()

    rands = torch.zeros(shape)
    for i in range(shape[0]):
        torch.manual_seed(i)
        rands[i] = torch.rand(shape[1:])
    RANDOM_VALS[shape] = rands
    return RANDOM_VALS[shape]


# def get_path_matrix(T, remove_self_loops=False):
#     bs = 0 if len(T.shape) == 2 else T.shape[0]
#     # Path matrix that tells us if path exists from node i to node j
#     I = repeat_n(torch.eye(T.shape[1]), bs)
#     # M is T with self loops.
#     # Convert to bool to get more efficient matrix multiplicaton
#     B = torch.logical_or(T,I).int()
#     # Implementing Algorithm 1 here, which uses repeated squaring to efficiently calc path matrix:
#     # https://courses.grainger.illinois.edu/cs598cci/sp2020/LectureNotes/lecture1.pdf
#     k = np.ceil(np.log2(T.shape[1]))
#     k = int(torch.ceil(torch.log2(torch.tensor(T.shape[1], dtype=torch.float))))
#     for _ in range(int(k)):
#         B = torch.matmul(B, B)
#     if remove_self_loops:
#         B = torch.logical_xor(B,I)
#     P = torch.sigmoid(BINARY_ALPHA * (2*B - 1))
#     return P



def print_averaged_tree(losses_tensor, V, full_trees, node_idx_to_label, custom_colors, ordered_sites, print_config):
    '''
    Returns an averaged tree over all (TODO: all or top k?) converged trees
    by weighing each tree edge or vertex label by the softmax of the 
    likelihood of that tree 
    '''
    _, min_loss_indices = torch.topk(losses_tensor, len(losses_tensor), largest=False, sorted=True)
    # averaged tree edges are weighted by the average of the softmax of the negative log likelihoods
    # TODO*: is this the right way to compute weights?
    def softmaxed_losses(losses_tensor):
        if not torch.is_tensor(losses_tensor):
            losses_tensor = torch.tensor(losses_tensor)
        return torch.softmax(-1.0*(torch.log2(losses_tensor)/ torch.log2(torch.tensor(1.1))), dim=0)

    weights = torch.softmax(-1.0*(torch.log2(losses_tensor)/ torch.log2(torch.tensor(1.1))), dim=0)
    #print("losses tensor\n", losses_tensor, weights)

    weighted_edges = dict() # { edge_0 : [loss_0, loss_1] }
    weighted_node_colors = dict() # { node_0 : { anatomical_site_0 : [loss_0, loss_3]}}
    for sln_idx in min_loss_indices:
        loss = losses_tensor[sln_idx]
        weight = weights[sln_idx]

        full_tree_node_idx_to_label = get_full_tree_node_idx_to_label(V[sln_idx], full_trees[sln_idx], node_idx_to_label, ordered_sites)

        for i, j in tree_iterator(full_trees[sln_idx]):
            edge = full_tree_node_idx_to_label[i][0], full_tree_node_idx_to_label[j][0]
            if edge not in weighted_edges:
                weighted_edges[edge] = []
            weighted_edges[edge].append(weight.item())

        for node_idx in full_tree_node_idx_to_label:
            site_idx = (V[sln_idx][:,node_idx] == 1).nonzero()[0][0].item()
            node_label, _ = full_tree_node_idx_to_label[node_idx]
            if node_label not in weighted_node_colors:
                weighted_node_colors[node_label] = dict()
            if site_idx not in weighted_node_colors[node_label]:
                weighted_node_colors[node_label][site_idx] = []
            weighted_node_colors[node_label][site_idx].append(loss.item())
    
    avg_node_colors = dict()
    for node_label in weighted_node_colors:
        avg_node_colors[node_label] = dict()
        avg_losses = []
        ordered_labels = weighted_node_colors[node_label]
        for site_idx in ordered_labels:
            vals = weighted_node_colors[node_label][site_idx]
            avg_losses.append(sum(vals)/len(vals))

        #softmaxed = np.exp(softmaxed)/sum(np.exp(softmaxed))
        softmaxed = softmaxed_losses(avg_losses)
        for site_idx, soft in zip(ordered_labels, softmaxed):
            avg_node_colors[node_label][site_idx] = soft
    #print("avg_node_colors\n", avg_node_colors)

    avg_edges = dict()
    for edge in weighted_edges:
        avg_edges[edge] = sum(weighted_edges[edge])/len(weighted_edges[edge])

    #print("avg_edges\n", avg_edges)

    plot_averaged_tree(avg_edges, avg_node_colors, ordered_sites, custom_colors, node_idx_to_label, show=print_config.visualize)

# TODO: make custom_node_idx_to_label a required argument

def plot_averaged_tree(avg_edges, avg_node_colors, ordered_sites, custom_colors=None, custom_node_idx_to_label=None, show=True):

    penwidth = 2.0
    alpha = 1.0

    max_edge_weight = max(list(avg_edges.values()))

    def rescaled_edge_weight(edge_weight):
        return (penwidth/max_edge_weight)*edge_weight
    
    
    G = nx.DiGraph()

    for label_i, label_j in avg_edges.keys():
        
        node_i_color = ""
        for site_idx in avg_node_colors[label_i]:
            node_i_color += f'"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_i][site_idx]}:"'
        node_j_color = ""
        for site_idx in avg_node_colors[label_j]:
            node_j_color += f'"{idx_to_color(custom_colors, site_idx, alpha=alpha)};{avg_node_colors[label_j][site_idx]}:"'
        is_leaf = False

        G.add_node(label_i, xlabel=label_i, label="", shape="circle", fillcolor=node_i_color, 
                    color="none", penwidth=3, style="wedged",
                    fixedsize="true", height=0.35, fontname=FONT, 
                    fontsize="10pt")
        G.add_node(label_j, xlabel="" if is_leaf else label_j, label="", shape="circle", 
                    fillcolor=node_j_color, color="none", 
                    penwidth=3, style="solid" if is_leaf else "wedged",
                    fixedsize="true", height=0.35, fontname=FONT, 
                    fontsize="10pt")

        # G.add_node(label_i, shape="circle", style="wedged", fillcolor=node_i_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        # G.add_node(label_j, shape="circle", style="wedged", fillcolor=node_j_color, color="none",
        #     alpha=0.5, fontname = "arial", fontsize="10pt", fixedsize="true", width=0.5)
        #print(label_i, label_j, avg_edges[(label_i, label_j)], rescaled_edge_weight(avg_edges[(label_i, label_j)]))
        # G.add_edge(label_i, label_j, color="#black", penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0.75, spline="ortho")
        style = "dashed" if is_leaf else "solid"
        penwidth = 2 if is_leaf else 2.5
        xlabel = "" if is_leaf else label_j
        G.add_edge(label_i, label_j,
                    color=f'"grey"', 
                    penwidth=rescaled_edge_weight(avg_edges[(label_i, label_j)]), arrowsize=0, fontname=FONT, 
                    fontsize="10pt", style=style)

    #assert(nx.is_tree(G))
    # we have to use graphviz in order to get multi-color edges :/
    dot = to_pydot(G).to_string()
    # hack since there doesn't seem to be API to modify graph attributes...
    dot_lines = dot.split("\n")
    dot_lines.insert(1, 'graph[splines=false]; nodesep=0.7; ranksep=0.6; forcelabels=true;')
    dot = ("\n").join(dot_lines)
    src = Source(dot) # dot is string containing DOT notation of graph
    if show:
        display(src)

def generate_legend_dot(ordered_sites, custom_colors, node_options):
    legend = nx.DiGraph()
    # this whole reversed business is to get the primary at the top of the legend...
    for i, site in enumerate(reversed(ordered_sites)):
        color = idx_to_color(custom_colors, len(ordered_sites)-1-i)
        legend.add_node(i, shape="plaintext", style="solid", label=f"{site}\r", 
                        width=0.3, height=0.2, fixedsize="true",
                        fontname=FONT, fontsize="10pt")
        legend.add_node(f"{i}_circle", fillcolor=color, color=color, 
                        style="filled", height=0.2, **node_options)

    legend_dot = to_pydot(legend).to_string()
    legend_dot = legend_dot.replace("strict digraph", "subgraph cluster_legend")
    legend_dot = legend_dot.split("\n")
    legend_dot.insert(1, 'rankdir="LR";{rank=source;'+" ".join(str(i) for i in range(len(ordered_sites))) +"}")
    legend_dot = ("\n").join(legend_dot)
    return legend_dot

# def relabel_cluster(label, shorten):
#     if not shorten:
#         return label

#     out = ""
#     # e.g. 1_M2 -> 1_M2
#     if len(label) <= 4 :
#         out = label
#     # e.g. 1;3;6;19_M2 -> 1_M2
#     elif ";" in label and "_" in label:
#         out = label[:label.find(";")] + label[label.find("_"):]
#     # e.g. 100_M2 -> 100_M2
#     elif "_" in label:
#         out = label
#     # e.g. 2;14;15 -> 2;14
#     else:
#         out = ";".join(label.split(";")[:2])
    
#     return out

# def old_is_monophyletic(adj_matrix, nodes_to_check):
#     def dfs(node, target):
#         visited[node] = True
#         if node == target:
#             return True
#         for neighbor, connected in enumerate(adj_matrix[node]):
#             if connected and not visited[neighbor] and dfs(neighbor, target):
#                 return True
#         return False

#     # Initialize variables
#     num_nodes = len(adj_matrix)
#     visited = [False] * num_nodes
#     highest_node = find_highest_level_node(adj_matrix, nodes_to_check)
#     if highest_node == get_root_index(adj_matrix):
#         return False
#     # Check if all nodes can be reached from the top level node in the seeding
#     # nodes (seeding node that is closest to the root)
#     for node in nodes_to_check:
#         visited = [False] * num_nodes
#         if not dfs(highest_node, node):
#             return False
#     return True
    
# def old_get_tracerx_seeding_pattern(V, A, ordered_sites, primary_site, full_node_idx_to_label):
#     '''
#     V: Vertex labeling matrix where columns are one-hot vectors representing the
#     anatomical site that the node originated from (num_sites x num_nodes)
#     A:  Adjacency matrix (directed) of the full tree (num_nodes x num_nodes)
#     ordered_sites: list of the anatomical site names (e.g. ["breast", "lung_met"]) 
#     with length =  num_anatomical_sites) and the order matches the order of cols in V
#     primary_site: name of the primary site (must be an element of ordered_sites)




#     TRACERx has a different definition of monoclonal vs. polyclonal:
#     "If only a single metastatic sample was considered for a case, the case-level 
#     dissemination pattern matched the metastasis level dissemination pattern. 
#     If multiple metastases were sampled and the dissemination pattern of any 
#     individual metastatic sample was defined as polyclonal, the case-level 
#     dissemination pattern was also defined as polyclonal. Conversely,if all metastatic 
#     samples follow a monoclonal dissemination pattern, all shared clusters between 
#     the primary tumour and each metastasis were extracted. If all shared clusters 
#     overlapped across all metastatic samples, the case-level dissemination pattern 
#     was classified as monoclonal, whereas,  if any metastatic sample shared 
#     additional clusters with the primary tumour, the overall dissemination pattern 
#     was defined as polyclonal."

#     and they define monophyletic vs. polyphyletics as:
#     "the origin of the seeding clusters was determined as monophyletic if all 
#     clusters appear along a single branch, and polyphyletic if clusters were
#     spread across multiple branches of the phylogenetic tree. Thus, if a 
#     metastasis was defined as monoclonal, the origin was necessarily monophyletic. 
#     For polyclonal metastases, the clusters were mapped to branches of the 
#     evolutionary tree. If multiple branches were found, the origin was determined 
#     to be polyphyletic, whereas, if only a single branch gave rise to all shared 
#     clusters, the origin was defined as monophyletic."
#     (from https://www.nature.com/articles/s41586-023-05729-x#Sec7)

#     tl;dr:   
#     Monoclonal if only one clone seeds met(s), else polyclonal
#     Monophyletic if there is a way to get from one seeding clone to all other seeding
#     clones, else polyphyletic

#     returns: verbal description of the seeding pattern
#     '''
    
#     Y = get_migration_edges(V,A)
#     G = get_migration_graph(V, A)
#     non_zero = torch.where(G > 0)
#     source_sites = non_zero[0]
#     if len(torch.unique(source_sites)) == 0:
#         return "no seeding"

#     pattern = ""
#     # 1) determine if monoclonal (no multi-eges) or polyclonal (multi-edges)
#     if len(ordered_sites) == 2:
#         pattern = "polyclonal " if ((G > 1).any()) else  "monoclonal "
#     elif ((G > 1).any()):
#         pattern = "polyclonal "
#     else:
#         shared_clusters = get_shared_clusters(V, A, ordered_sites, primary_site, full_node_idx_to_label)
#         prim_to_met_clusters = shared_clusters[ordered_sites.index(primary_site)]
#         all_seeding_clusters = set([cluster for seeding_clusters in prim_to_met_clusters for cluster in seeding_clusters])
#         monoclonal = True
#         for cluster_set in prim_to_met_clusters:
#             # if clusters that seed the primary to each met are not identical,
#             # then this is a polyclonal pattern
#             if len(cluster_set) != 0 and (set(cluster_set) != all_seeding_clusters):
#                 monoclonal = False
#                 break
#         pattern = "monoclonal " if monoclonal else "polyclonal "

#     # 2) determine if monophyletic or polyphyletic
#     if pattern == "monoclonal ":
#         pattern += "monophyletic"
#         return pattern
    
#     seeding_clusters = set()
#     for i,j in tree_iterator(Y):
#         # if j is a subclonal presence leaf node, add i as the shared cluster 
#         # (b/c i is the mutation cluster that j represents)
#         if full_node_idx_to_label[j][1] == True:
#             seeding_clusters.add(i)
#         else:
#             seeding_clusters.add(j)
        
#     phylo = "monophyletic" if is_monophyletic(A,list(seeding_clusters)) else "polyphyletic"
    
#     return pattern + phylo

# def get_seeding_clusters(V,A):
#     shared_clusters = [[[] for x in range(len(ordered_sites))] for y in range(len(ordered_sites))]
#     for i,j in tree_iterator(Y):
#         site_i = (V[:,i] == 1).nonzero()[0][0].item()
#         site_j = (V[:,j] == 1).nonzero()[0][0].item()
#         assert(site_i != site_j)
#         # if j is a subclonal presence leaf node, add i is the shared cluster 
#         # (b/c i is the mutation cluster that j represents)
#         if full_node_idx_to_label[j][1] == True:
#             shared_clusters[site_i][site_j].append(i)
#         else:
#             shared_clusters[site_i][site_j].append(j)
#     return shared_clusters

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

# def get_leaf_labels_from_U(U):
#     U = U[:,1:] # don't include column for normal cells
#     num_sites = U.shape[0]
#     L = torch.nn.functional.one_hot((U > U_CUTOFF).nonzero()[:,0], num_classes=num_sites).T
#     return L

def get_adj_matrices_from_spruce_mutation_trees(mut_trees_filename, idx_to_character_label, is_sim_data=False):
    '''
    When running MACHINA's generatemutationtrees executable (SPRUCE), it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of tuples, each containing (T, pruned_idx_to_character_label) for each
    tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - idx_to_character_label: a dict mapping indices of the adj matrix T to character
        labels 
    '''
    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_data = []
        for i, line in enumerate(f):
            if i < 3: continue
            # This marks the beginning of a tree
            if "#edges, tree" in line:
                adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
                out.append((adj_matrix, pruned_idx_to_label))
                tree_data = []
            else:
                nodes = line.strip().split()
                # fixes incompatibility with naming b/w cluster file (uses ";" separator)
                # and the ./generatemutationtrees output (uses "_" separator)
                if is_sim_data:
                    tree_data.append((";".join(nodes[0].split("_")), ";".join(nodes[1].split("_"))))
                else:
                    tree_data.append((nodes[0], nodes[1]))

        adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
        out.append((adj_matrix, pruned_idx_to_label))
    return out

# # TODO: remove polytomy stuff?
# def get_ref_var_matrices_from_machina_sim_data(tsv_filepath, pruned_idx_to_cluster_label, T):
#     '''
#     tsv_filepath: path to tsv for machina simulated data (generated from create_conf_intervals_from_reads.py)

#     tsv is expected to have columns: ['#sample_index', 'sample_label', 'anatomical_site_index',
#     'anatomical_site_label', 'character_index', 'character_label', 'f_lb', 'f_ub', 'ref', 'var']

#     pruned_idx_to_cluster_label:  dictionary mapping the cluster index to label, where 
#     index corresponds to col index in the R matrix and V matrix returned. This isn't 1:1 
#     with the 'character_label' to 'character_index' mapping in the tsv because we only keep the
#     nodes which appear in the mutation tree, and re-index after removing unseen nodes
#     (see _get_adj_matrix_from_machina_tree)

#     T: adjacency matrix of the internal nodes.

#     returns
#     (1) R matrix (num_samples x num_clusters) with the # of reference reads for each sample+cluster,
#     (2) V matrix (num_samples x num_clusters) with the # of variant reads for each sample+cluster,
#     (3) unique anatomical sites from the patient's data
#     '''

#     assert(pruned_idx_to_cluster_label != None)
#     assert(T != None)

#     pruned_cluster_label_to_idx = {v:k for k,v in pruned_idx_to_cluster_label.items()}
#     with open(tsv_filepath) as f:
#         tsv = csv.reader(f, delimiter="\t", quotechar='"')
#         # Take a pass over the tsv to collect some metadata
#         num_samples = 0 # S
#         for i, row in enumerate(tsv):
#             # Get the position of columns in the csvs
#             if i == 3:
#                 sample_idx = row.index('#sample_index')
#                 site_label_idx = row.index('anatomical_site_label')
#                 cluster_label_idx = row.index('character_label')
#                 ref_idx = row.index('ref')
#                 var_idx = row.index('var')

#             if i > 3:
#                 num_samples = max(num_samples, int(row[sample_idx]))
#         # 0 indexing
#         num_samples += 1

#     num_clusters = len(pruned_cluster_label_to_idx.keys())

#     R = np.zeros((num_samples, num_clusters))
#     V = np.zeros((num_samples, num_clusters))
#     unique_sites = []
#     with open(tsv_filepath) as f:
#         tsv = csv.reader(f, delimiter="\t", quotechar='"')
#         for i, row in enumerate(tsv):
#             if i < 4: continue
#             if row[cluster_label_idx] in pruned_cluster_label_to_idx:
#                 mut_cluster_idx = pruned_cluster_label_to_idx[row[cluster_label_idx]]
#                 R[int(row[sample_idx]), mut_cluster_idx] = int(row[ref_idx])
#                 V[int(row[sample_idx]), mut_cluster_idx] = int(row[var_idx])

#             # collect additional metadata
#             # doing this as a list instead of a set so we preserve the order
#             # of the anatomical site labels in the same order as the sample indices
#             if row[site_label_idx] not in unique_sites:
#                 unique_sites.append(row[site_label_idx])

#     # Fill the columns in R and V with the resolved polytomies' parents data
#     # (if there are resolved polytomies)
#     for cluster_label in pruned_cluster_label_to_idx:
#         if is_resolved_polytomy_cluster(cluster_label):
#             res_polytomy_idx = pruned_cluster_label_to_idx[cluster_label]
#             parent_idx = np.where(T[:,res_polytomy_idx] == 1)[0][0]
#             R[:, res_polytomy_idx] = R[:, parent_idx]
#             V[:, res_polytomy_idx] = V[:, parent_idx]

#     return torch.tensor(R, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), list(unique_sites)

# def shorten_cluster_names(idx_to_full_cluster_label, split_char):
#     idx_to_cluster_label = dict()
#     for ix in idx_to_full_cluster_label:
#         og_label_muts = idx_to_full_cluster_label[ix].split(split_char) # e.g. CUL3:2:225371655:T;TRPM6:9:77431650:C
#         idx_to_cluster_label[ix] = og_label_muts[0]
#     return idx_to_cluster_label


# TODO: take out skip polytomies functionality?
def get_adj_matrix_from_machina_tree(character_label_to_idx, tree_filename, remove_unseen_nodes=True, skip_polytomies=False):
    '''
    character_label_to_idx: dictionary mapping character_label to index (machina
    uses colors to represent subclones, so this would map 'pink' to n, if pink
    is the nth node in the adjacency matrix).
    tree_filename: path to .tree file
    remove_unseen_nodes: if True, removes nodes that
    appear in the machina tsv file but do not appear in the reported tree
    skip_polyomies: if True, checks for polytomies and skips over them. For example
    if the tree is 0 -> polytomy -> 1, returns 0 -> 1. If the tree is 0 -> polytomy
    returns 0.

    Returns:
        T: adjacency matrix where Tij = 1 if there is a path from i to j
        character_label_to_idx: a pruned character_label_to_idx where nodes that
        appear in the machina tsv file but do not appear in the reported tree are removed
    '''
    edges = []
    with open(tree_filename, 'r') as f:
        for line in f:
            nodes = line.strip().split()
            node_i, node_j = nodes[0], nodes[1]
            edges.append((node_i, node_j))
    return _get_adj_matrix_from_machina_tree(edges, character_label_to_idx, remove_unseen_nodes, skip_polytomies)


# Thsi was in _get_adj_matrix_from_machina_tree
# Fix missing connections
if skip_polytomies:
    for child_label in child_to_parent_map:
        parent_label = child_to_parent_map[child_label]
        if is_resolved_polytomy_cluster(parent_label) and parent_label in child_to_parent_map:
            # Connect the resolved polytomy's parent to the resolved polytomy's child
            res_poly_parent = child_to_parent_map[parent_label]
            if res_poly_parent in character_label_to_idx and child_label in character_label_to_idx:
                T[character_label_to_idx[res_poly_parent], character_label_to_idx[child_label]] = 1

def get_adj_matrices_from_spruce_mutation_trees_no_pruning_reordering(mut_trees_filename, idx_to_character_label):
    '''
    When running MACHINA's generatemutationtrees executable (SPRUCE), it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of trees for each tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - idx_to_character_label: a dict mapping indices of the adj matrix T to character
        labels 

    Does not prine idx_to_character_label, and does not reorder indices like 
    get_adj_matrices_from_spruce_mutation_trees does
    '''

    character_label_to_idx = {v:k for k,v in idx_to_character_label.items()}

    def _build_tree(edges):
        num_internal_nodes = len(character_label_to_idx)
        T = np.zeros((num_internal_nodes, num_internal_nodes))
        for edge in edges:
            node_i, node_j = edge[0], edge[1]
            T[character_label_to_idx[node_i], character_label_to_idx[node_j]] = 1
        return T

    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_data = []
        for i, line in enumerate(f):
            if i < 3: continue
            # This marks the beginning of a tree
            if "#edges, tree" in line:
                adj_matrix= _build_tree(tree_data)
                out.append(adj_matrix)
                tree_data = []
            else:
                nodes = line.strip().split()
                tree_data.append((nodes[0], nodes[1]))

        adj_matrix = _build_tree(tree_data)
        out.append(adj_matrix)
    return out

def is_resolved_polytomy_cluster(cluster_label):
    '''
    In MACHINA simulated data, cluster labels with non-numeric components (e.g. M2_1
    instead of 1;3;4) represent polytomies
    '''
    is_polytomy = False
    for mut in cluster_label.split(";"):
        if not mut.isnumeric() and (mut.startswith('M') or mut.startswith('P')):
            is_polytomy = True
    return is_polytomy

def is_leaf(cluster_label):
    '''
    In MACHINA simulated data, cluster labels that have underscores (e.g. 3;4_M2)
    represent leaves
    '''
    return "_" in cluster_label and not is_resolved_polytomy_cluster(cluster_label)

def get_leaf_labels_from_U(U, input_T):
    
    U = U[:,1:] # don't include column for normal cells
    P = path_matrix(input_T, remove_self_loops=True)
    internal_node_idx_to_sites = {}
    for node_idx in range(U.shape[1]):
        descendants = np.where(P[node_idx] == 1)[0]
        for site_idx in range(U.shape[0]):
            node_U = U[site_idx,node_idx]
            is_present = False
            if node_U > U_CUTOFF:
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
    print("internal_node_idx_to_sites", internal_node_idx_to_sites)
    return internal_node_idx_to_sites


def remove_extra_resolver_nodes(best_Vs, best_Ts, node_idx_to_label, G, poly_res, p):
    '''
    If there are any resolver nodes that were added to resolve polytomies but they 
    weren't used (i.e. 1. they have no children or 2. they don't change the 
    migration history), remove them
    '''

    if poly_res == None:
        return best_Vs, best_Ts, [G for _ in range(len(best_Vs))], [node_idx_to_label for _ in range(len(best_Vs))]
    
    prev_ms, prev_cs, prev_ss, _, _ = vutil.ancestral_labeling_metrics(vutil.to_tensor(best_Vs), vutil.to_tensor(best_Ts), None, None, p, True)
    out_Vs, out_Ts, out_Gs, out_node_idx_to_labels = [], [],[],[]
    for prev_m, prev_c, prev_s, V, T in zip(prev_ms, prev_cs, prev_ss, best_Vs, best_Ts):
        nodes_to_remove = []
        for new_node_idx in poly_res.resolver_indices:
            children_of_new_node = vutil.get_child_indices(T, [new_node_idx])
            if len(children_of_new_node) <= 1:
                nodes_to_remove.append(new_node_idx)
            elif is_same_mig_hist_with_node_removed(int(prev_m), int(prev_c), int(prev_s), T, V, new_node_idx, p):
                nodes_to_remove.append(new_node_idx)
        
        new_V, new_T, new_G, new_node_idx_to_label = remove_nodes(nodes_to_remove, V, T, G, node_idx_to_label)
        out_Vs.append(new_V)
        out_Ts.append(new_T)
        out_Gs.append(new_G)
        out_node_idx_to_labels.append(new_node_idx_to_label)
    return out_Vs, out_Ts, out_Gs, out_node_idx_to_labels


def is_same_mig_hist_with_node_removed(prev_m, prev_c, prev_s, T, V, remove_idx, p):
    '''
    Returns True if migration #, comigration # and seeding # are
    the same or better after removing node at index remove_idx
    '''
    # Attach all the children of the candidate removal node to
    # its parent, and then check if that changes the migration history or not
    candidate_T = T.clone().detach()
    candidate_V = V.clone().detach()
    parent_idx = np.where(T[:,remove_idx] > 0)[0][0]
    child_indices = vutil.get_child_indices(T, [remove_idx])
    for child_idx in child_indices:
        candidate_T[parent_idx,child_idx] = 1.0
    candidate_T = np.delete(candidate_T, remove_idx, 0)
    candidate_T = np.delete(candidate_T, remove_idx, 1)
    candidate_V = np.delete(candidate_V, remove_idx, 1)
    new_m, new_c, new_s, _, _ = vutil.ancestral_labeling_metrics(vutil.add_batch_dim(candidate_V), candidate_T, None, None, p, True)
    return ((prev_m >= int(new_m)) and (prev_c >= int(new_c)) and (prev_s >= int(new_s)))

def expand_solutions(solutions, all_pars_metrics, O, p, weights):
    '''
    In hard (i.e. usually large input) cases where we are unable to find a 
    primary-only seeding solution, see if we can recover one by post-processing
    final solutions and removing any met-to-met migration edges, and add these
    to our final solution set
    '''
    unique_tree_labelings = set()
    expanded_solutions = []
    expanded_pars_metrics = []
    for soln, pars_metrics in zip(solutions, all_pars_metrics):
        unique_tree_labelings.add(vutil.LabeledTree(soln.T, soln.V))
        expanded_solutions.append(soln)
        expanded_pars_metrics.append(pars_metrics)

        if pars_metrics[2] > 1:
            seeding_clusters = putil.get_seeding_clusters(soln.V,soln.T)
            new_V = copy.deepcopy(soln.V)
            for s in seeding_clusters:
                new_V[:,s] = p.T 
            loss, loss_dict = ancestral_labeling_objective(vutil.add_batch_dim(new_V), vutil.add_batch_dim(soln.soft_V), soln.T, soln.G, O, p, weights, True)
            new_soln = vutil.VertexLabelingSolution(loss, new_V, soln.soft_V, soln.T, soln.G, soln.idx_to_label, soln.i)
            new_labeled_tree = vutil.LabeledTree(new_soln.T, new_soln.V)
            m, c, s = loss_dict[MIG_KEY], loss_dict[COMIG_KEY], loss_dict[SEEDING_KEY]
            new_pars_metrics = (int(m), int(c), int(s))
            if new_pars_metrics not in expanded_pars_metrics and new_labeled_tree not in unique_tree_labelings:
                expanded_solutions.append(new_soln)
                unique_tree_labelings.add(new_labeled_tree)
                expanded_pars_metrics.append(new_pars_metrics)
    
    return expanded_solutions, expanded_pars_metrics
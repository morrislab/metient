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


# old comigration number
    # VAT = torch.transpose(VA, 2, 1)
    # W = VAT @ VA # 1 if two nodes' parents are the same color
    # Y = torch.sum(torch.mul(VAT, 1-VT), axis=2) # Y has a 1 for every node where its parent has a diff color
    # X = VT @ V # 1 if two nodes are the same color
    # shared_par_and_self_color = torch.mul(W, X) # 1 if two nodes' parents are same color AND nodes are same color
    # # tells us if two nodes are (1) in the same site and (2) have parents in the same site
    # # and (3) there's a path from node i to node j
    # global LAST_P # this is expensive to compute, so hash it if we don't need to update it
    # if LAST_P != None and not update_path_matrix:
    #     P = LAST_P.to(A.device)
    #     # Make sure if we're using a cached version (only applicable in cases where 
    #     # we're not resolving polytomies), we provide the correct shape on the first 
    #     # batch dimension. This is only when we're post-processing individual solutions,
    #     # so just grab the first 2-D path matrix
    #     if P.shape != A.shape:
    #         assert(A.shape[0]==1)
    #         P = P[0]
    # else:
    #     P = path_matrix(A, remove_self_loops=True, identical_T=identical_T)
    #     LAST_P = P
    # shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=2)
    # repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y), axis=1)
    # binarized_site_adj = torch.sigmoid(BINARY_ALPHA * (2 * site_adj - 1))
    # bin_site_trace = torch.diagonal(binarized_site_adj, offset=0, dim1=1, dim2=2).sum(dim=1)
    # c = torch.sum(binarized_site_adj, dim=(1,2)) - bin_site_trace + repeated_temporal_migrations
    # return c

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

# WAS IN POLYTOMY_RESOLVER.PY is_same_mig_hist_with_node_removed
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
    
    return False
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

# Used to be in remove_nodes in polytomy_resolver
    # # Attach children of the node to remove to their original parent
    # for remove_idx in removal_indices:
    #     parent_idx = torch.where(T[:,remove_idx] > 0)[0][0]
    #     child_indices = vutil.get_child_indices(T, [remove_idx])
    #     for child_idx in child_indices:
    #         T[parent_idx,child_idx] = 1.0

    # # Remove indices from T, V and G
    # # Remove rows of T
    # T = T[torch.tensor([i for i in range(T.size(0)) if i not in removal_indices])]
    # # Remove columns of T
    # T = T[:, torch.tensor([i for i in range(T.size(1)) if i not in removal_indices])]


# This used to be used in genetic_distance_score in vertex_labeling_util
def sparse_sum_along_dim_1_2(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    # Extract indices and values from the sparse tensor
    indices = sparse_tensor.indices()  # Tensor of shape [ndims, nnz]
    values = sparse_tensor.values()    # Tensor of shape [nnz]

    # We want to sum along dimension 1 and 2, so we group by dimension 0
    dim0_indices = indices[0]  # Get the indices corresponding to dimension 0

    # Perform the summation for each unique index in dimension 0
    unique_dim0_indices = torch.unique(dim0_indices)

    sum_result = torch.zeros(len(unique_dim0_indices))

    for i, idx in enumerate(unique_dim0_indices):
        # Select values where dimension 0 equals idx
        mask = (dim0_indices == idx)
        
        # Sum values corresponding to that index
        sum_result[i] = torch.sum(values[mask])

    # The result will be the sum over dimensions 1 and 2 for each unique index in dimension 0
    return sum_result


def old_sparse_path_matrix(sparse_matrix, remove_self_loops, identical_T):
    '''
    Compute the transitive closure of the sparse adjacency matrix
    '''
    # Get dimensions of the 3D sparse tensor
    og_batch_size, n, _ = sparse_matrix.size()

    batch_size = og_batch_size
    # We only need to compute P once if all Ts along the batch dimension are identical
    if identical_T:
        batch_size = 1

    for i in range(batch_size):
        T = sparse_matrix[i].coalesce()
        path_matrix(T, remove_self_loops=remove_self_loops, identical_T=identical_T)

    # List to hold the transitive closures for each graph
    closures = []

    # TODO: Any way to do this without a for loop?
    for i in range(batch_size):
        # Get the sparse matrix for the batch, coalesce to handle duplicates
        closure = sparse_matrix[i].coalesce()
        closure_indices = closure.indices()
        
        # Create a boolean mask for known edges
        known_edges_mask = torch.zeros((n, n), dtype=torch.bool)
        known_edges_mask[closure_indices[0], closure_indices[1]] = True

        # Perform n iterations of transitive closure (if necessary)
        for _ in range(n):
            # Perform sparse matrix multiplication
            product = torch.sparse.mm(closure, closure).coalesce()
            product_indices = product.indices()

            # Create a mask for the product edges
            product_edges_mask = known_edges_mask[product_indices[0], product_indices[1]]

            # Identify new edges that are not in the known edges
            new_edges_mask = ~product_edges_mask
            
            if new_edges_mask.any():
                new_indices = product_indices[:, new_edges_mask]
                combined_indices = torch.cat([closure.indices(), new_indices], dim=1)
                combined_values = torch.ones(combined_indices.shape[1], dtype=torch.float32)
                closure = torch.sparse_coo_tensor(combined_indices, combined_values, (n, n)).coalesce()

                # Update known edges
                known_edges_mask[new_indices[0], new_indices[1]] = True

        closures.append(closure)
    
    if remove_self_loops:
        for i in range(batch_size):
            closure = closures[i]
            indices = closure.indices()
            values = closure.values()

            # Identify self-loops (where row index == column index)
            mask = indices[0] != indices[1]
            closures[i] = torch.sparse_coo_tensor(
                indices[:, mask],
                values[mask],
                closure.size()
            ).coalesce()

    if identical_T:
        stacked_closure = repeat_n(closures[0], og_batch_size)
    else:
        stacked_closure = torch.stack(closures)

    return stacked_closure 


def path_matrix_matmul(T, remove_self_loops=False, identical_T=False):
    '''
    T is a numpy ndarray or tensor adjacency matrix (where Tij = 1 if there is a path from i to j)
    remove_self_loops: bool, whether to retain 1s on the diagonal
    identical_T: every T along the batch_size dimension is identical (allows us to optimize)

    Returns path matrix that tells us if path exists from node i to node j    
    '''

    bs = 1 if len(T.shape) == 2 else T.shape[0]
    if T.is_sparse:
        return sparse_path_matrix(T, remove_self_loops, identical_T)

    I = torch.eye(T.shape[1]).repeat(bs, 1, 1)  # Repeat identity matrix along batch dimension

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
        # Early stopping if there have been no changes
        if torch.equal(P_old, torch.nonzero(P)) and k != 0:
            break
        
    if remove_self_loops:
        P = torch.logical_xor(P, I.int())
    return P.squeeze(0) if len(T.shape) == 2 else P


# Old get best final solutions

tracemalloc.start()
solution_set, has_pss_solution = create_solution_set(best_Vs[soln_idx], best_soft_Vs[soln_idx], best_Ts[soln_idx], 
                                                        G, O, p, node_collection, weights, solve_polytomies)
current, peak = tracemalloc.get_traced_memory()
print(f"[After creating soln set] Current memory usage: {current / 10**6:.2f} MB; Peak was {peak / 10**6:.2f} MB")
tracemalloc.stop()
multiresult_has_pss_solution = multiresult_has_pss_solution or has_pss_solution

# 1. Make solutions unique before we do all this additional post-processing work which is time intensive
results_unique_solution_set = []
for soln in solution_set:
    tree = vutil.MigrationHistory(soln.T, soln.V)
    if tree not in unique_labelings:
        results_unique_solution_set.append(soln)
        unique_labelings.add(tree)
del solution_set

# Don't need to recover a primary-only single source solution if we have already found one
if not multiresult_has_pss_solution:
    recover_prim_ss_solutions(results_unique_solution_set, unique_labelings, 
                                weights, O, p, solve_polytomies, 
                                fixed_indices, num_internal_nodes)
    multiresult_has_pss_solution = True

#  Remove any extra resolver nodes that don't actually help
results_unique_solution_set = prutil.remove_extra_resolver_nodes(results_unique_solution_set, poly_res, weights, O, p)
full_solution_set.extend(results_unique_solution_set)

#del results[i]


# This was in find_optimal_subtree_nodes
# Don't fix when it's just a node and its witness node (the labeling of the node could be
# multiple possibilities and still be optimal)
# Caveat in the case of lineage tracing, where every node with just a witness node is truly
# originated from the site of the witness node
if len(descendants) == 1 and not is_lineage_tracing:
    continue



def reindex_dense_matrix(matrix, removal_indices):
    # Attach children of the node to remove to their original parent
    for remove_idx in removal_indices:
        parent_idx = torch.where(matrix[:,remove_idx] > 0)[0][0]
        child_indices = vutil.get_child_indices(matrix, [remove_idx])
        for child_idx in child_indices:
            matrix[parent_idx,child_idx] = 1.0

    # Remove rows
    adj_matrix = adj_matrix[torch.tensor([i for i in range(adj_matrix.size(0)) if i not in removal_indices])]
    # Remove columns of T
    T = T[:, torch.tensor([i for i in range(T.size(1)) if i not in removal_indices])]

def shrink_tensor(tensor, index_to_remove):
    """
    Shrinks a 3D tensor by removing the slice at the specified index.
    Handles both dense and sparse tensors.
    """
    if tensor.is_sparse:
        # Handle the case for sparse tensors (3D)
        indices = tensor._indices()
        values = tensor._values()

        # Filter out the indices where the first dimension equals index_to_remove
        mask = indices[0] != 0
        new_indices = indices[:, mask]
        new_indices[0] = new_indices[0]-1
        new_values = values[mask]
        
        # The new shape of the tensor is one less in the first dimension (batch dimension)
        new_shape = list(tensor.shape)
        new_shape[0] -= 1

        # Create a new sparse tensor with the filtered indices and values
        new_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        # For a 2D dense tensor, we just slice off the 2D tensor at index 0 each time 
        index_to_remove = 0
        # Handle the case for dense tensors (same as before)
        new_size = list(tensor.shape)
        new_size[0] -= 1  # Remove one slice along the batch dimension

        new_tensor = torch.empty(new_size, dtype=tensor.dtype, device=tensor.device)
        new_tensor[:index_to_remove] = tensor[:index_to_remove]
        new_tensor[index_to_remove:] = tensor[index_to_remove + 1:]
    del tensor
    return new_tensor

# @profile
def create_solution_set(best_Vs, best_soft_Vs, best_Ts, 
                        G, O, p, node_collection, weights, solve_polytomies):
    # Make a solution set
    losses, new_values = vutil.clone_tree_labeling_objective(
        vutil.to_tensor(best_Vs),
        vutil.to_tensor(best_soft_Vs),
        vutil.to_tensor(best_Ts),
        torch.stack([G for _ in range(len(best_Vs))]) if G != None else None,
        O, p, weights, update_path_matrix=solve_polytomies
    )

    solution_set = []
    _, _, ss, _, _, _ = new_values
    has_pss_solution = any(s == 1 for s in ss)
    print("before", best_Vs.shape, best_soft_Vs.shape, best_Ts.shape)
    for i, (loss, m, c, s, g, o, e) in enumerate(zip(losses, *new_values)):
        # Node info (node index to label) is the same for every solution if we're not resolving for
        # polytomies, but can be different solution to solution if not
        # TODO: this deepcopy can cause memory issues when tree inputs are large
        soln_node_collection = node_collection if not solve_polytomies else copy.deepcopy(node_collection)
        
        V = best_Vs[0].clone()
        soft_V = best_soft_Vs[0].clone()
        T = best_Ts[0].clone()
        print(loss, m, c, s, g, o, e)
        soln = vutil.VertexLabelingSolution(loss, m, c, s, g, o, e, V, soft_V, T, G, soln_node_collection)
        solution_set.append(soln)
        
        # Shrink best_Vs, best_soft_Vs, and best_Ts tensors after processing
        best_Vs = shrink_tensor(best_Vs, i)
        best_soft_Vs = shrink_tensor(best_soft_Vs, i)
        best_Ts = shrink_tensor(best_Ts, i)
        print("during", best_Vs.shape, best_soft_Vs.shape, best_Ts.shape)
        
        # Optionally, free memory from the processed tensors
        del V, soft_V, T  # Remove references to the processed tensors

    print("after", best_Vs.shape, best_soft_Vs.shape, best_Ts.shape)
    return solution_set, has_pss_solution



def compute_weighted_gradients(v_optimizer, losses, prev_losses, scaler, alpha=0.9):
    """
    Computes gradients with a weighted scheme, giving more emphasis to samples with improved loss.

    Args:
    - v_optimizer: Optimizer object.
    - losses: Current loss values for each sample.
    - prev_losses: Previous loss values for each sample.
    - scaler: GradScaler object to handle gradient scaling.
    - alpha: Weight factor for improved samples (0 < alpha <= 1). Higher values give more weight to improved samples.

    Returns:
    - updated_prev_losses: Updated previous losses for the next iteration.
    """
    # Identify which samples have an improved loss
    improved_loss_mask = losses < prev_losses  # Mask where loss has improved

    # Compute weights for the loss
    weights = torch.where(improved_loss_mask, alpha, 1 - alpha)
    weighted_losses = weights * losses

    # Reset the gradients
    v_optimizer.zero_grad()

    # Compute the weighted loss and perform the backward pass
    scaled_loss = scaler.scale(weighted_losses.mean())
    scaled_loss.backward()

    # Update the previous losses for the next iteration
    updated_prev_losses = losses.clone()

    return updated_prev_losses

def compute_weighted_loss(losses, weighting_factor=2.0):
    """
    Computes a weighted mean loss, emphasizing low-loss samples.
    
    Args:
    - losses: Tensor of individual sample losses. Should be non-negative.
    - weighting_factor: Exponential weight to emphasize low-loss samples.
    
    Returns:
    - weighted_loss: Scalar weighted loss.
    """
    if losses.numel() == 0:
        raise ValueError("Losses tensor is empty. Cannot compute weighted loss.")

    # Safeguard against NaNs or negative values in losses
    if torch.any(losses < 0):
        raise ValueError("Losses tensor contains negative values, which are not supported.")

    # Apply power-based scaling to make the weighting more aggressive
    scaled_losses = weighting_factor * losses

    # Use a power function to emphasize lower losses more strongly
    weights = 1 / (1 + scaled_losses ** 2)  # Squared loss scaling
    # print(losses)
    # print(weights)
    # Weighted mean loss, without normalizing the weights
    weighted_loss = torch.sum(weights * losses)

    return weighted_loss

def update_t_with_polytomy_resolver(poly_res: prutil.PolytomyResolver, 
                                    t_temp: float, 
                                    v_solver: VertexLabelingSolver) -> torch.Tensor:
    """
    Updates a sparse adjacency matrix T based on children-to-parent assignments 
    resolved using the polytomy resolver (poly_res) and Gumbel-softmax sampling.

    Removes old connections between child nodes and their previous parents, 
    replacing them with the new connections resolved by the polytomy resolver.
    Additionally, removes connections to nodes in `resolver_indices` if they have no children.

    Args:
        poly_res (Any): The polytomy resolver containing:
            - latent_var: A batch_size x n x m matrix, where n is the number of nodes 
              and m is the number of children of polytomies.
            - children_of_polys: The order of child nodes of polytomies in latent_var's 2nd dim.
        t_temp (float): Temperature parameter for Gumbel-softmax sampling.
        v_solver (Any): A solver object with an adjacency matrix T (a sparse tensor).

    Returns:
        torch.Tensor: Updated sparse adjacency matrix T for the batch, 
        with old parent-child connections replaced as per the resolved decisions.
    """
    # Perform Gumbel-softmax to resolve parent assignments for children of polytomies
    # softmax_pol_res is sample_size x number of nodes x children of polytomies
    softmax_pol_res, _ = gumbel_softmax(poly_res.latent_var, t_temp)
    non_zero_indices = torch.nonzero(softmax_pol_res, as_tuple=False).T

    # Extract relevant indices from the non-zero entries
    batch_indices = non_zero_indices[0]
    parent_indices = non_zero_indices[1] 
    child_indices = non_zero_indices[2]
    global_child_indices = torch.tensor(poly_res.children_of_polys, device=softmax_pol_res.device)[child_indices]

    # Find rows that are all zeros in softmax_pol_res
    all_zeros_mask = torch.all(softmax_pol_res == 0, dim=2)  # Shape: (sample_size, num_nodes)
    zero_rows_per_batch = torch.nonzero(all_zeros_mask, as_tuple=False)  # Shape: (num_zero_rows, 2)
    # Filter the row indices to only include those in resolver_indices
    resolver_indices = torch.tensor(poly_res.resolver_indices, device=zero_rows_per_batch.device)
    valid_rows_mask = torch.isin(zero_rows_per_batch[:, 1], resolver_indices)
    resolver_indices_no_children = zero_rows_per_batch[valid_rows_mask]
    #print("resolver_indices_no_children", resolver_indices_no_children)

    # Create new connections
    new_indices = torch.stack([batch_indices, parent_indices, global_child_indices])
    new_values = torch.ones(new_indices.size(1), dtype=torch.float32, device=softmax_pol_res.device)

    # Repeat and coalesce the adjacency matrix for the batch
    bs = poly_res.latent_var.shape[0]
    T = vutil.repeat_n(v_solver.T, bs).coalesce()

    # Step 1: Find the parent-child connections in T where the child is one of resolver_indices_no_children
    resolver_children_indices = resolver_indices_no_children[:, 1]  # Extract child indices
    # Find all parent-child connections in T
    parent_child_mask = torch.isin(T.indices()[2], resolver_children_indices)

    # Filter out old connections for the updated children
    existing_indices = T.indices()
    existing_values = T.values()
    
    # Perform comparison using broadcasting
    batch_child_pairs = (
    existing_indices[0] * T.shape[2] + existing_indices[2]
        )  # Hash batch-child pairs
    new_batch_child_pairs = (
        batch_indices * T.shape[2] + global_child_indices
    )  # Hash new batch-child pairs

    mask = ~torch.isin(batch_child_pairs, new_batch_child_pairs)

    # Mask out connections from parent to children in resolver_indices_no_children
    resolver_batch_mask = torch.isin(existing_indices[0], resolver_indices_no_children[:, 0])
    masked_parent_child_mask = parent_child_mask & resolver_batch_mask

    # Mask out connections from parent to children in resolver_indices_no_children
    # filtered_indices = existing_indices[:, mask & ~masked_parent_child_mask]  # Mask out parent-child relations
    # filtered_values = existing_values[mask & ~masked_parent_child_mask]  # Apply the same mask to values
    filtered_indices = existing_indices[:, mask]  # Mask out parent-child relations
    filtered_values = existing_values[mask]  # Apply the same mask to values
    # Concatenate filtered existing data with new connections
    updated_indices = torch.cat([filtered_indices, new_indices], dim=1)
    updated_values = torch.cat([filtered_values, new_values])

    # Create updated sparse tensor
    updated_T = torch.sparse_coo_tensor(updated_indices, updated_values, T.shape).coalesce()

    return updated_T

def stack_vertex_labeling(L, X, p, poly_res, fixed_labeling):
    '''
    Use leaf labeling L and X (both of size sample_size x num_sites X num_internal_nodes)
    to get the anatomical sites of the leaf nodes and the internal nodes (respectively). 
    Stack the root labeling to get the full vertex labeling V. 
    '''
    # Expand leaf node labeling L to be repeated sample_size times
    bs = X.shape[0]
    L = vutil.repeat_n(L, bs)

    if fixed_labeling != None:
        full_X = torch.zeros((bs, X.shape[1], len(fixed_labeling.known_indices)+len(fixed_labeling.unknown_indices)))
        known_labelings = vutil.repeat_n(fixed_labeling.known_labelings, bs)
        full_X[:,:,fixed_labeling.unknown_indices] = X
        full_X[:,:,fixed_labeling.known_indices] = known_labelings
    else:
        full_X = X

    # if poly_res != None:
    #     # Order is: internal nodes, new poly nodes, leaf nodes from U
    #     full_vert_labeling = torch.cat((full_X, vutil.repeat_n(poly_res.resolver_labeling, bs), L), dim=2)
    # else:
    #     full_vert_labeling = torch.cat((full_X, L), dim=2)

    full_vert_labeling = torch.cat((full_X, L), dim=2)
    p = vutil.repeat_n(p, bs)
    # Concatenate the left part, new column, and right part along the second dimension
    full_vert_labeling = torch.cat((p, full_vert_labeling), dim=2)

    # Set to store unique labels where labels in columns 2 and 7 are equal across nodes and batches
    # unique_labels = set()

    # # Iterate over batches and nodes to check for equality in columns 2 and 7
    # for batch_idx in range(full_vert_labeling.shape[0]):  # Iterate over batches
    #     if torch.equal(full_vert_labeling[batch_idx, :, 9], full_vert_labeling[batch_idx, :, 1]):
    #         unique_labels.add((int(torch.nonzero(full_vert_labeling[batch_idx, :, 9]).squeeze()))) 
    #     if torch.equal(full_vert_labeling[batch_idx, :, 10], full_vert_labeling[batch_idx, :, 1]):
    #         unique_labels.add((int(torch.nonzero(full_vert_labeling[batch_idx, :, 10]).squeeze()))) 

    #Print unique labels where columns 2 and 7 have the same label
    # print(f"Unique labels where columns 9/10 and 1 have the same label: {sorted(unique_labels)}")

    return full_vert_labeling

class SolutionTracker:
    """Tracks and manages diverse solutions using soft assignments"""
    def __init__(self, temperature=0.1, diversity_threshold=0.1, max_solutions=100):
        self.temperature = temperature
        self.diversity_threshold = diversity_threshold
        self.max_solutions = max_solutions
        self.solutions = []  # List of (V, soft_V, T, metrics, signature)
        
    def compute_signature(self, soft_V):
        """Convert soft assignments to a comparable signature"""
        # Use softmax to get probabilities
        probs = torch.nn.functional.softmax(soft_V / self.temperature, dim=1)
        # Get most likely assignments and their probabilities
        top_probs, top_indices = torch.topk(probs, k=2, dim=1)
        # Create signature combining top assignments and their probabilities
        return torch.cat([top_indices.float(), top_probs], dim=1)
        
    def is_diverse(self, signature):
        """Check if solution is sufficiently different from existing ones"""
        if not self.solutions:
            return True
            
        existing_signatures = torch.stack([s[4] for s in self.solutions])
        distances = torch.cdist(signature.unsqueeze(0), existing_signatures)
        return distances.min() > self.diversity_threshold
        
    def add_solution(self, V, soft_V, T, metrics):
        """Add new solution if it's diverse or dominates existing solutions"""
        signature = self.compute_signature(soft_V)
        
        # Check if solution is diverse
        if self.is_diverse(signature):
            self.solutions.append((V, soft_V, T, metrics, signature))
            return True
            
        # Even if not diverse, check if it dominates any existing solutions
        new_metrics = torch.tensor(metrics)
        dominated_indices = []
        
        for i, (_, _, _, existing_metrics, _) in enumerate(self.solutions):
            existing_metrics = torch.tensor(existing_metrics)
            if torch.all(new_metrics <= existing_metrics) and torch.any(new_metrics < existing_metrics):
                dominated_indices.append(i)
                
        if dominated_indices:
            # Remove dominated solutions
            self.solutions = [s for i, s in enumerate(self.solutions) if i not in dominated_indices]
            self.solutions.append((V, soft_V, T, metrics, signature))
            return True
            
        return False
        
    def prune_solutions(self):
        """Maintain only non-dominated solutions"""
        non_dominated = []
        metrics_list = torch.stack([torch.tensor(s[3]) for s in self.solutions])
        
        for i, metrics in enumerate(metrics_list):
            if not torch.any(torch.all(metrics_list < metrics, dim=1)):
                non_dominated.append(self.solutions[i])
                
        self.solutions = non_dominated[:self.max_solutions]

def batch_solutions(solutions):
    """Helper function to batch solutions into tensors for efficient processing.
    
    Args:
        solutions: List of (V, soft_V, T, metrics) tuples
        
    Returns:
        Tuple of (batched_V, batched_soft_V, batched_T, batched_metrics)
    """
    if not solutions:
        return tuple()
        
    # Stack all solutions along batch dimension
    batched_V = torch.stack([s[0] for s in solutions])
    batched_soft_V = torch.stack([s[1] for s in solutions]) 
    batched_T = torch.stack([s[2] for s in solutions])
    batched_metrics = tuple(torch.stack([s[3][i] for s in solutions]) 
                          for i in range(len(solutions[0][3])))
                          
    return batched_V, batched_soft_V, batched_T, batched_metrics

def fit_u_map(u_solver):
    
    # We're learning eta, which is the mixture matrix U (U = softmax(eta)), and tells us the existence
    # and anatomical locations of the extant clones (U > U_CUTOFF)
    #eta = -1 * torch.rand(num_sites, num_internal_nodes + 1) # an extra column for normal cells
    eta = torch.ones(u_solver.num_sites, u_solver.num_internal_nodes + 1) # an extra column for normal cells
    eta.requires_grad = True 
    u_optimizer = torch.optim.Adam([eta], lr=u_solver.config['lr'])

    B = vutil.mutation_matrix_with_normal_cells(u_solver.input_T)
    print("B\n", B)
    i = 0
    u_prev = eta
    u_diff = 1e9
    while u_diff > 1e-6 and i < 300:
        u_optimizer.zero_grad()
        U, u_loss, nll, reg = compute_u_loss(eta, u_solver.ref, u_solver.var, u_solver.omega, B, u_solver.weights)
        u_loss.backward()
        u_optimizer.step()
        u_diff = torch.abs(torch.norm(u_prev - U))
        u_prev = U
        i += 1

    print_U(U, B, u_solver.node_collection, u_solver.ordered_sites, u_solver.ref, u_solver.var)

    return build_tree_with_witness_nodes(U, u_solver)

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

def compute_u_loss(eta, ref, var, omega, B, weights):
    '''
    Args:
        eta: raw values we are estimating of matrix U (num_sites x num_internal_nodes)
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
    U = torch.softmax(eta, dim=1)
    # print("eta", eta)
    #print("U", U)

    # 1. Data fit
    F_hat = (U @ B)
    nlglh = calc_llh(F_hat, ref, var, omega)
    # 2. Regularization to make some values of U -> 0
    reg = torch.sum(eta) # l1 norm 
    clone_proportion_loss = (weights.data_fit*nlglh + weights.reg*reg)
    
    return U, clone_proportion_loss, weights.data_fit*nlglh, weights.reg*reg

def print_U(U, B, node_collection, ordered_sites, ref, var):
    cols = ["GL"]+[";".join([str(i)]+node_collection.get_node(i).label[:2]) for i in range(len(node_collection.get_nodes())) if not node_collection.get_node(i).is_witness]
    U_df = pd.DataFrame(U.detach().numpy(), index=ordered_sites, columns=cols)

    print("U\n", U_df)
    F_df = pd.DataFrame((var/(ref+var)).numpy(), index=ordered_sites, columns=cols[1:])
    print("F\n", F_df)
    Fhat_df = pd.DataFrame(0.5*(U @ B).detach().numpy()[:,1:], index=ordered_sites, columns=cols[1:])
    print("F hat\n", Fhat_df)

# def get_best_final_solutions(results, G, O, p, weights, print_config, 
#                            node_collection, solve_polytomies, 
#                            v_solver, num_internal_nodes, keep_pareto_only=True):
#     """Modified to use solution diversity tracking"""
#     solution_tracker = SolutionTracker(
#         temperature=0.1,
#         diversity_threshold=0.1 * len(node_collection.idx_to_label()),
#         max_solutions=print_config.k_best_trees
#     )
    
#     # Process results and add to tracker
#     for result_idx, result in enumerate(results):
#         best_Vs, soft_Vs, best_Ts, _, metrics = result
#         for soln_idx, (m,c,s,g,o,e) in enumerate(zip(*metrics)):
#             V = best_Vs[soln_idx].clone().cpu()
#             soft_V = soft_Vs[soln_idx].clone().cpu()
#             T = best_Ts[soln_idx].clone().cpu()
            
#             # Add back removed nodes if necessary
#             if v_solver.fixed_labeling is not None and v_solver.poly_res is None:
#                 V = add_back_removed_nodes(V, v_solver, p)
#                 T = add_back_removed_nodes_to_tree(T, v_solver)
                
#             solution_tracker.add_solution(V, soft_V, T, (m,c,s,g,o,e))
    
#     # Get final solutions
#     solution_tracker.prune_solutions()
#     final_solutions = []
#     for V, soft_V, T, metrics in solution_tracker.solutions:
#         loss = vutil.clone_tree_labeling_loss_with_computed_metrics(*metrics, weights, bs=1)
#         soln = vutil.VertexLabelingSolution(loss, *metrics, V, soft_V, T, G, node_collection)
#         final_solutions.append(soln)
    
#     return rank_solutions(final_solutions, print_config)

    def _calculate_valid_sites(self):
        """Calculate valid sites for each node based on its rooted subtree's leaves using BFS."""
        from collections import deque
        
        n_nodes = self.input_T.shape[0]
        valid_sites = {}  # Changed from list to dict
        primary_site = torch.nonzero(self.p)[0,0].item()

        # Initialize leaf nodes
        for idx, sites in self.idx_to_observed_sites.items():
            valid_sites[idx] = set(sites)
        
        # Build children dictionary
        children = {}
        for i in range(n_nodes):
            children[i] = [j for j in range(n_nodes) if self.input_T[i, j] == 1]

        leaf_nodes = list(self.idx_to_observed_sites.keys())
        # BFS from leaves up to root
        queue = deque(leaf_nodes)
        processed = set(leaf_nodes)
        bottom_up_order = []
        
        while queue:
            node = queue.popleft()
            bottom_up_order.append(node)
            
            # Find parent (there should be exactly one, except for root)
            for potential_parent in range(n_nodes):
                if self.input_T[potential_parent, node] == 1:
                    # Check if all children of the parent have been processed
                    if potential_parent not in processed and all(child in processed for child in children[potential_parent]):
                        queue.append(potential_parent)
                        processed.add(potential_parent)
                    break
        
        # Process nodes in bottom-up order
        for node in bottom_up_order:
            if node not in valid_sites:  # if not already initialized
                valid_sites[node] = {primary_site}  # Add primary site
                for child in children[node]:
                    valid_sites[node].update(valid_sites[child])
        
        return valid_sites

# Add these diagnostic functions
def analyze_solution_diversity(X):
    """Analyze how diverse the current solutions are"""
    # Get the most likely assignment for each node
    max_assignments = torch.argmax(X, dim=1)  # shape: [sample_size, num_nodes_to_label]
    
    # Count unique solutions
    unique_solutions = torch.unique(max_assignments, dim=0).shape[0]
    
    # Calculate entropy of assignments for each node
    probs = torch.softmax(X, dim=1)  # shape: [sample_size, num_sites, num_nodes_to_label]
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean(dim=0)
    
    print(f"Number of unique solutions: {unique_solutions}")
    print(f"Average entropy per node: {entropy.tolist()}")

def analyze_loss_landscape(v_solver, X, v_temp, t_temp, poly_res, exploration_weights):
    """Analyze loss landscape to detect optimization issues like local minima or plateaus.
    
    Returns:
        dict: Diagnostic metrics about the loss landscape
    """
    with torch.no_grad():
        # Get current loss
        _, current_losses, _, _, _ = compute_v_t_loss(
            X, v_solver, poly_res, exploration_weights,
            update_path_matrix=False, v_temp=v_temp,
            t_temp=t_temp, compute_full_c=False, identical_T=True
        )
        current_loss = torch.mean(current_losses).item()
        
        # Sample multiple perturbations at each noise level
        noise_levels = [0.01, 0.1, 0.5, 1.0]
        samples_per_noise = 5
        diagnostics = {
            'current_loss': current_loss,
            'noise_stats': {},
            'warnings': []
        }
        
        print(f"\nLoss Landscape Analysis (current loss: {current_loss:.4f}):")
        
        for noise in noise_levels:
            losses = []
            for _ in range(samples_per_noise):
                perturbed = X + torch.randn_like(X) * noise
                _, perturbed_losses, _, _, _ = compute_v_t_loss(
                    perturbed, v_solver, poly_res, exploration_weights,
                    update_path_matrix=False, v_temp=v_temp,
                    t_temp=t_temp, compute_full_c=False, identical_T=True
                )
                losses.append(torch.mean(perturbed_losses).item())
            
            # Compute statistics for this noise level
            losses = np.array(losses)
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            min_loss = np.min(losses)
            
            diagnostics['noise_stats'][noise] = {
                'mean_loss': mean_loss,
                'std_loss': std_loss,
                'min_loss': min_loss,
                'mean_delta': mean_loss - current_loss,
                'min_delta': min_loss - current_loss
            }
            
            print(f"\nNoise level {noise:.3f}:")
            print(f"  Mean loss: {mean_loss:.4f} (Δ={mean_loss-current_loss:+.4f})")
            print(f"  Std dev:   {std_loss:.4f}")
            print(f"  Min loss:  {min_loss:.4f} (Δ={min_loss-current_loss:+.4f})")
        
        # Analyze landscape characteristics
        small_noise = diagnostics['noise_stats'][0.01]
        large_noise = diagnostics['noise_stats'][1.0]
        
        # Check for local minimum
        if small_noise['min_loss'] > current_loss:
            diagnostics['warnings'].append("Likely in local minimum - all nearby points have higher loss")
        
        # Check for plateau
        if small_noise['std_loss'] < 1e-4:
            diagnostics['warnings'].append("Possible plateau detected - very small loss variation in local neighborhood")
        
        # Check for optimization potential
        if large_noise['min_loss'] < current_loss:
            delta_percent = ((current_loss - large_noise['min_loss']) / current_loss) * 100
        
            diagnostics['warnings'].append(
                f"Better solutions may exist - found {delta_percent:.1f}% improvement with large perturbation"
            )
        
        # Print interpretation
        print("\nInterpretation:")
        if not diagnostics['warnings']:
            print("✓ Loss landscape appears well-behaved")
        else:
            for warning in diagnostics['warnings']:
                print(f"! {warning}")
        
        return diagnostics

# In init_optimal_x_polyres
# fixed_labeling = None
    # if known_indices:
    #     print("T", T[0])
    #     unknown_indices = [x for x in range(v_solver.num_nodes_to_label+1) if x not in known_indices and x != 0]
    #     known_labelings = torch.stack(known_labelings, dim=1)
    #     X = X[:,:,[x-1 for x in unknown_indices]]
    #     fixed_labeling = vutil.FixedVertexLabeling(known_indices, unknown_indices, known_labelings, optimal_root_nodes)

    #     print("known_indices", known_indices, "unknown_indices", unknown_indices)
    #     print("optimal_root_nodes", optimal_root_nodes)

    #     # Identify nodes to remove: descendants that are in optimal subtrees
    #     nodes_to_remove = set()
    #     for known_idx in known_indices:
    #         if known_idx not in optimal_root_nodes:
    #             nodes_to_remove.add(known_idx)
    #     print("nodes_to_remove", nodes_to_remove)
    #     _T = _remove_nodes_from_T(T, nodes_to_remove)
    #     print("_T", _T[0])

def stack_closures(closures, batch_size, n):
    # Stack the closures into a single sparse tensor
    stacked_indices = []
    stacked_values = []

    for i in range(batch_size):
        closure = closures[i]
        indices = closure.indices().clone()
        values = closure.values()

        # Adjust row indices for stacking
        indices[0] += i * n

        stacked_indices.append(indices)
        stacked_values.append(values)

    # Concatenate indices and values
    final_indices = torch.cat(stacked_indices, dim=1)
    final_values = torch.cat(stacked_values)

    # Create the final stacked sparse tensor
    stacked_closure = torch.sparse_coo_tensor(final_indices, final_values, (batch_size, n, n))

    return stacked_closure

# in eval_util.get_max_cross_ent_thetas

if use_min_tau:
    gen_dist_scores = torch.zeros((len(loss_dicts)))
    for i, loss_dict in enumerate(loss_dicts):
        gen_dist_scores[i] = loss_dict[GEN_DIST_KEY]
    unique_gs,_ = torch.sort(torch.unique(gen_dist_scores))
    if len(unique_gs) >= 2:
        print("gen dist diff", (unique_gs[1]-unique_gs[0]))
        min_tau = min(min_tau, (unique_gs[1]-unique_gs[0]))

def run_fitch_hartigan(v_solver, results):
    """
    Recover possible ancestral states using Fitch's algorithm and select one optimal solution

    Parameters:
    - adj_matrix: Sparse COO adjacency matrix representing the tree structure (n x n).
    - node_idx_to_observed_sites: Dictionary mapping leaf node indices to their labels.
    - root_label: The known label of the root node.

    Returns:
    - ancestral_matrix: A k x n matrix, where k is the number of labels, and n is the internal node indices.
    """
    adj_matrix = v_solver.input_T
    node_idx_to_observed_sites = v_solver.idx_to_observed_sites
    root_label = torch.argmax(v_solver.p, dim=0).item()
    
    n = adj_matrix.shape[0]
    k = v_solver.num_sites

    # Step 1: Initialize a k x n matrix to store possible labels for each node (True/False for each label)
    ancestral_matrix = torch.zeros((k, n), dtype=torch.float32)

    # Step 2: Fill in the matrix for leaf nodes and root node based on the node_idx_to_observed_sites dictionary
    for node, leaf_labels in node_idx_to_observed_sites.items():
        ancestral_matrix[:, node] = 0
        for leaf_label in leaf_labels:
            ancestral_matrix[leaf_label, node] = 1  # Set the corresponding label to True

    # For leaf nodes without any observed sites, initialize as possibly belonging to all sites
    leaf_nodes = torch.nonzero(adj_matrix.sum(dim=1) == 0).squeeze(dim=1).tolist()

    for leaf in leaf_nodes:
        if leaf not in node_idx_to_observed_sites:
            ancestral_matrix[:, leaf] = 1

    root = get_root_index(adj_matrix)
    ancestral_matrix[:, root] = 0
    ancestral_matrix[root_label, root] = 1

    # Step 3: Perform the downward pass (Fitch algorithm) for internal nodes
    def fitch_down(node):
        node = int(node)

        # Check if the node has observed sites
        has_observed_sites = node in node_idx_to_observed_sites

        # Find children of the node using the adjacency matrix
        children = torch.nonzero(adj_matrix[node]).squeeze(dim=1)

        # If the node is a leaf, return its known label set
        if len(children) == 0 and has_observed_sites:
            return ancestral_matrix[:, node]

        # Collect label sets from all children
        child_label_sets = [fitch_down(child) for child in children]

        # Perform set intersection or union based on children's labels
        intersection = torch.stack(child_label_sets).all(dim=0)
        if torch.any(intersection):  # If intersection is non-empty, use it as the set
            ancestral_matrix[:, node] = intersection
        else:  # Otherwise, use the union of the children's sets
            union = torch.stack(child_label_sets).any(dim=0)
            ancestral_matrix[:, node] = union

        # If the node has observed sites, enforce them on the computed labels
        if has_observed_sites:
            observed_sites = ancestral_matrix[:, node]
            ancestral_matrix[:, node] = ancestral_matrix[:, node] * observed_sites

        return ancestral_matrix[:, node]
    
    # Step 4: Perform Fitch's downward pass starting from the root
    fitch_down(root)
    # Reinforce root label
    ancestral_matrix[:, root] = 0
    ancestral_matrix[root_label, root] = 1
    single_soln_matrix = ancestral_matrix.clone()

    # Step 5: Perform the upward pass to choose specific states for internal nodes,
    # starting with the root label.
    def fitch_up(node, parent_label):
        children = torch.nonzero(adj_matrix[node]).squeeze(dim=1)

        # If there are multiple possible labels, choose one (use the parent's label if possible)
        possible_labels = torch.where(single_soln_matrix[:, node])[0]
        if parent_label in possible_labels:
            chosen_label = parent_label
        else:
            chosen_label = possible_labels[0]

        # Set the chosen label to True and others to False for the node
        single_soln_matrix[:, node] = 0
        single_soln_matrix[chosen_label, node] = 1

        # Pass the chosen label to the children
        for child in children:
            fitch_up(child, chosen_label)

    # Perform the upward pass starting from the root with the enforced root label
    fitch_up(root, root_label)

    # We needed to include the root labeling for Fitch-Hartigan, but we restack 
    # it using stack_vertex_labeling, so remove it momentarily
    single_soln_matrix = torch.cat((single_soln_matrix[:, :root], single_soln_matrix[:, root+1:]), dim=1)
    V = stack_vertex_labeling(v_solver.L, add_batch_dim(single_soln_matrix), v_solver.p, None, None)
    T = repeat_n(v_solver.full_T,1)
    if v_solver.config['use_sparse_T']:
        T = T.to_sparse()
    metrics = ancestral_labeling_metrics(V, T, v_solver.full_G, v_solver.O, v_solver.p, 
                                         update_path_matrix=True, compute_full_c=True, identical_T=True)
    V = V.cpu().detach()
    T = T.cpu().detach()
    metrics = tuple(metric.cpu().detach() for metric in metrics)
    print("Fitch-hartigan result:", metrics)
    results.append((V, torch.zeros(V.shape, device=V.device), T, None, (*metrics,torch.zeros(size=(1,),device=V.device))))

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
    
    if not A.is_sparse:
        return comigration_number_dense(base_c, V, VA, VT, P)
    # Get node colors (anatomical site labels) - shape: [batch_size, num_nodes]
    node_colors = torch.argmax(V, dim=1)
    
    # Compute parent colors efficiently using sparse matrix multiplication
    # Shape: [batch_size, num_nodes]
    parent_site_probs = VA.transpose(1, 2)
    parent_colors = torch.argmax(parent_site_probs, dim=2)

    # Find nodes that differ from their parents
    diff_nodes = torch.where(node_colors != parent_colors)  # Returns (batch_indices, node_indices)
    
    temporal_migrations = torch.zeros(node_colors.shape[0], device=V.device)
    print("P.is_sparse", P.is_sparse)
    # Process each batch
    for batch in range(node_colors.shape[0]):
        # Get the differing nodes for this batch
        batch_diff_nodes = diff_nodes[1][diff_nodes[0] == batch]
        print("batch_diff_nodes\n",batch_diff_nodes)
        root_node = get_root_index(A[batch])
        print("root_node\n",root_node)
        
        if len(batch_diff_nodes) > 1:  # Need at least 2 nodes for temporal migrations
            # Create combined key for just these nodes
            diff_node_colors = node_colors[batch, batch_diff_nodes]
            diff_parent_colors = parent_colors[batch, batch_diff_nodes]
            combined_colors = diff_node_colors * V.shape[1] + diff_parent_colors

            # Find groups of nodes with same combination
            unique_combos, inverse_indices, counts = torch.unique(combined_colors, return_inverse=True, return_counts=True)
            
            # Process combinations that appear more than once
            repeated_mask = counts > 1
            repeated_combos = unique_combos[repeated_mask]

            # For each group of matching nodes with same color combination
            for combo in repeated_combos:
                matching_indices = batch_diff_nodes[combined_colors == combo]
                print("matching_indices\n",matching_indices)
                # Handle sparse tensor properly
                if P.is_sparse:
                    P_batch = P[batch].coalesce()
                    # Create a mask for the matching indices
                    matching_mask = torch.zeros(P_batch.size(0), dtype=torch.bool, device=P_batch.device)
                    matching_mask[matching_indices] = True
                    
                    # Get all reachable pairs at once
                    source_mask = P_batch.indices()[0][:, None] == matching_indices
                    target_mask = matching_mask[P_batch.indices()[1]]
                    
                    # Count valid paths (avoiding self-paths and double counting)
                    valid_paths = (source_mask & target_mask[:, None]).sum()
                    print("valid_paths\n",valid_paths)
                    temporal_migrations[batch] += valid_paths
                else:
                    # For dense tensors, use matrix operations
                    reachability_submatrix = P[batch][matching_indices][:, matching_indices]
                    # Create upper triangular mask to avoid double counting
                    upper_tri_mask = torch.triu(torch.ones_like(reachability_submatrix), diagonal=1, device=reachability_submatrix.device)
                    temporal_migrations[batch] += (reachability_submatrix * upper_tri_mask).sum()

    print("base_c", base_c, "temporal_migrations", temporal_migrations)
    return base_c + temporal_migrations


def comigration_number_dense(base_c, V, VA, VT, P):
    """
    Handles the case where the adjacency matrix is dense, in which case we can use faster matrix operations
    to compute the comigration number.

    Args:
        - base_c: base comigration number (without temporal repeats, calculated using comigration_number_approximation)
        - V: Vertex labeling one-hot matrix (sample_size x num_sites x num_nodes)
        - VA: V*A (will be converted to sparse)
        - VT: transpose of V 
        - P: path matrix
     Returns:
        - comigration number: a subset of the migration edges between two anatomical sites, such that 
        the migration edges occur on distinct branches of the clone tree
        
    """
    X = VT @ V # 1 if two nodes are the same color
    VAT = torch.transpose(VA, 2, 1)
    W = VAT @ VA # 1 if two nodes' parents are the same color
    Y = torch.sum(torch.mul(VAT, 1-VT), axis=2) # Y has a 1 for every node where its parent has a diff color
    shared_par_and_self_color = torch.mul(W, X) # 1 if two nodes' parents are same color AND nodes are same color

    shared_path_and_par_and_self_color = torch.sum(torch.mul(P, shared_par_and_self_color), axis=2)
    repeated_temporal_migrations = torch.sum(torch.mul(shared_path_and_par_and_self_color, Y), axis=1)
    return base_c + repeated_temporal_migrations
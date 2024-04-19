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
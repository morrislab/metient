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
    
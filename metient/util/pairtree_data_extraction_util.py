
import numpy as np
import torch
import os
import pandas as pd
import json

from metient.util import vertex_labeling_util as vutil
from metient.util import data_extraction_util as dutil

def write_pairtree_inputs(patient_data, patient_id, output_dir):
    '''
    Args:
        patient_data: pandas DataFrame with at least columns: [character_label, character_index
        sample_label, var, ref, major_cn, minor_cn]. character_label = name of mutation (e.g. "KCTD15:19:34584166:C"),
        character_index = unique index for each character label, sample_label = name of the sample,
        (e.g. "tumor1_region1"), var = variant allele read count, ref = reference allele read count
        
        patient_id: used to name the output files
        
    Outputs:
        Saves ssm and params.json files to output_dir as needed to run PairTree and/or Orchard. 
        See "Input Files" section here: https://github.com/morrislab/pairtree
    
    '''
    header = ["id", "name", "var_reads", "total_reads", "var_read_prob"]

    mutation_names = list(patient_data['character_label'].unique())
    mutation_ids  = list(patient_data['character_index'].unique())
    sample_names = list(patient_data['sample_label'].unique())

    mut_name_to_mut_id = {}
    with open(os.path.join(output_dir, f"{patient_id}.ssm"), "w") as f:

        f.write("\t".join(header))
        f.write("\n")
        for i, mut in zip(mutation_ids, mutation_names):
            mut_name_to_mut_id[mut] = f"m{i}"
            row = [f"m{i}", mut]
            mut_patient_subset = patient_data[patient_data['character_label'] == mut]
            var_reads = []
            total_reads = []
            var_read_probs = []
            for sample in sample_names:
                mut_patient_sample = mut_patient_subset[mut_patient_subset['sample_label'] == sample]
                var = int(mut_patient_sample['var'].values[0])
                ref = int(mut_patient_sample['ref'].values[0])
                var_reads.append(str(var))
                total_reads.append(str(var+ref))
                major_cn = int(mut_patient_sample['major_cn'].values[0])
                minor_cn = int(mut_patient_sample['minor_cn'].values[0])
                p = float(mut_patient_sample['purity'].values[0])
                x = (p*(major_cn+minor_cn)+2*(1-p))
                var_read_prob = (p*major_cn)/x
                var_read_probs.append(str(var_read_prob))

            row += [",".join(var_reads), ",".join(total_reads), ",".join(var_read_probs)]
            f.write("\t".join(row))
            f.write("\n")
    json_data = {"samples": sample_names, "clusters": [[mut_name_to_mut_id[name]] for name in mutation_names], "garbage": []}

    with open(os.path.join(output_dir, f"{patient_id}.params.json"), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


def _get_cluster_id(mut_id, clusters):
    '''
    Given mut_id = 'm1' and clusters= [['m1'], ['m0', 'm2']], returns 1 
    '''
    for i, cluster in enumerate(clusters):
        if mut_id in cluster:
            return i
    
    print(f"mut id {mut_id} not found")
    return None

def write_clustered_params_json_from_pyclone_clusters(clusters_tsv_fn, pairtree_ssm_fn, pairtree_params_fn, 
                                                      output_dir, patient_id):
    '''
    After clustering with PyClone (see: https://github.com/Roth-Lab/pyclone),
    prepares clustered params.json files needed for PairTree/Orchard tree building

    Args:

        clusters_tsv_fn: PyClone results tsv that maps each mutation to a cluster id

        pairtree_ssm_fn: PairTree .ssm file which maps mutation ids (e.g. m0) to mutation
        names (e.g. SETD2:3:47103798:G)

        output_dir: where to save clustered params.json

        patient_id: name of patient used for saving

    Outputs:

        Saves clustered params.json at {output_dir}/{patient_id}_pyclone_clustered.params.json

    '''

    pyclone_df = pd.read_csv(clusters_tsv_fn, delimiter="\t")
    pt_ssm_df = pd.read_csv(pairtree_ssm_fn, delimiter="\t")
    cluster_id_to_mut_ids = dict() # 3 -> [m0, m5 ...]

    with open(pairtree_params_fn) as f:
        params = json.load(f)
        
    # 1. Get mapping between mutation ids (from .ssm file) and PyClone cluster ids
    for _, row in pt_ssm_df.iterrows():
        mut_id = row['id']
        mut_items = row['name'].split(":")
        cluster_id = pyclone_df[(pyclone_df['CHR']==int(mut_items[1]))&(pyclone_df['POS']==int(mut_items[2]))&(pyclone_df['REF']==mut_items[3])]['treeCLUSTER'].unique()
        assert(len(cluster_id) <= 1)
        if len(cluster_id) == 1:
            cluster_id = cluster_id.item()
            if cluster_id not in cluster_id_to_mut_ids:
                cluster_id_to_mut_ids[cluster_id] = []
            cluster_id_to_mut_ids[cluster_id].append(mut_id)
            
    # 2. Update params.json with clusters
    clusters = [cluster_id_to_mut_ids[x] for x in range(0, len(cluster_id_to_mut_ids))]
    params['clusters'] = clusters
    with open(os.path.join(output_dir, f"{patient_id}_pyclone_clustered.params.json"), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False)

def write_pooled_tsv_from_pairtree_clusters(tsv_fn, ssm_fn, clustered_params_json_fn, 
                                            aggregation_rules, output_dir, patient_id,
                                            cluster_sep=";"):

    '''
    After clustering with PairTree (see: https://github.com/morrislab/pairtree#clustering-mutations),
    prepares tsvs by pooling mutations belonging to the same cluster

    Args:
        tsv_fn: path to tsv with reqiured columns: [#sample_index, sample_label, 
        character_index, character_label, anatomical_site_index, anatomical_site_label, 
        ref, var]

        ssm_fn: SSM file used for input into pairtree (see: https://github.com/morrislab/pairtree#input-files)

        clustered_params_json_fn: output params.json file from running PairTree clustervars
        (see: https://github.com/morrislab/pairtree#clustering-mutations)

        aggregation_rules: dictionary indicating how to aggregate any extra columns that are not in 
        the required columns (specificied above). e.g. "first" aggregation rules can be used for columns 
        shared by rows with the same anatomical_site_label, since we are aggregating within an
        anatomical_site_label. 

        output_dir: where to save clustered tsv

        patient_id: name of patient used in tsv filename

        cluster_sep: string that separates names of mutations when creating a cluster name.
        e.g. cluster name for mutations ABC:4:3 and DEF:1:2 with cluster_sep=";" will be 
        "ABC:4:3;DEF:1:2"
    
    Outputs:

        Saves pooled tsv at {output_dir}/{patient_id}_clustered_SNVs.tsv

    '''

    df = pd.read_csv(tsv_fn, delimiter="\t", index_col=0)
    
    # 1. Get mapping between mutation names and pairtree mutation_ids
    mut_name_to_mut_id = dict()
    with open(ssm_fn) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            items = line.split("\t")
            if items[1] not in mut_name_to_mut_id:
                mut_name_to_mut_id[items[1]] = items[0]
                
    # 2. Get pairtree cluster assignments
    with open(clustered_params_json_fn) as f:
        cluster_json = json.loads(f.read())
    mut_name_to_cluster_id = dict()
    for mut_name in df['character_label']:
        cluster_id = _get_cluster_id(mut_name_to_mut_id[mut_name], cluster_json['clusters'])
        mut_name_to_cluster_id[mut_name] = cluster_id

    # 3. Get new names for clustered mutations
    mut_id_to_name = {v:k for k,v in mut_name_to_mut_id.items()}
    cluster_id_to_cluster_name = dict()
    for i, cluster in enumerate(cluster_json['clusters']):
        cluster_comps = []
        for mut in cluster:
            cluster_comps.append(mut_id_to_name[mut])
        cluster_id_to_cluster_name[i] = cluster_sep.join(cluster_comps)

    # 4. Pool mutations and write to file
    dutil.write_pooled_tsv_from_clusters(df, mut_name_to_cluster_id, cluster_id_to_cluster_name,
                                         aggregation_rules, output_dir, patient_id,)

# Adapted from pairtree 
def get_adj_matrix_from_parents(parents):
    K = len(parents) + 1
    T = np.eye(K)
    T[parents,np.arange(1, K)] = 1
    I = np.identity(T.shape[0])
    T = np.logical_xor(T,I).astype(int) # remove self-loops
    # remove the normal subclone
    normal_clone_idx = vutil.get_root_index(T)
    # Assert that this is a monoprimary tree (normal subclone has one cancerous child)
    if len(T[normal_clone_idx].nonzero()[0]) > 1:
        raise ValueError("Normal subclone has more than one child node. Rerun Orchard with monoprimary flag (-p).")

    T = np.delete(T, normal_clone_idx, 0)
    T = np.delete(T, normal_clone_idx, 1)
    return T

def get_adj_matrices_from_pairtree_results(pairtee_results_fns):
    if not isinstance(pairtee_results_fns, list):
        return get_adj_matrix_from_pairtree_results(pairtee_results_fns)

    assert(len(pairtee_results_fns) >= 1)

    return [get_adj_matrix_from_pairtree_results(fn) for fn in pairtee_results_fns]

def get_adj_matrix_from_pairtree_results(pairtee_results_fn):
    results = np.load(pairtee_results_fn)
    parent_vectors = results['struct']
    llhs = results['llh']

    data = []
    for parents_vector, llh in zip(parent_vectors, llhs):
        adj_matrix = get_adj_matrix_from_parents(parents_vector)
        data.append(torch.tensor(adj_matrix, dtype = torch.float32))
    return data


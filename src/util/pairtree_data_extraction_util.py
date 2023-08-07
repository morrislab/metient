
import numpy as np
import torch
from src.util import plotting_util as plt_util
import os

# TODO: make more assertions on uniqueness and completeness of input csvs

import json

# TODO: rename pairtree_data_extraction_util -> pairtree_util


# TODO: write unit tests for this
def write_pairtree_inputs(patient_data, patient_id, output_dir):
    '''
    Args:
        patient_data: pandas DataFrame with at least columns: [character_label, character_index
        sample_label, var, ref]. character_label = name of mutation (e.g. "KCTD15:19:34584166:C"),
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
                # TODO: Add CNA and incorporate ploidy, for now assume no CNA and diploid cells
                # var_read_prob = M/N, where N = 2 + (K - 2) * p (with K being the copy number 
                # at that locus, and p being the fraction of cells which have a copy number of 
                # K at the locus containing M. p is the purity of the sample if the copy number 
                # abberation is clonal)
                var_read_probs.append(str(0.5))

            row += [",".join(var_reads), ",".join(total_reads), ",".join(var_read_probs)]
            f.write("\t".join(row))
            f.write("\n")
    json_data = {"samples": sample_names, "clusters": [[mut_name_to_mut_id[name]] for name in mutation_names], "garbage": []}

    with open(os.path.join(output_dir, f"{patient_id}.params.json"), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)

# Adapted from pairtree 
def get_adj_matrix_from_parents(parents):
    K = len(parents) + 1
    T = np.eye(K)
    T[parents,np.arange(1, K)] = 1
    I = np.identity(T.shape[0])
    T = np.logical_xor(T,I).astype(int) # remove self-loops
    # remove the normal subclone
    root_idx = plt_util.get_root_index(T)
    T = np.delete(T, root_idx, 0)
    T = np.delete(T, root_idx, 1)
    return T

def get_adj_matrices_from_pairtree_results(pairtee_results_fn):
    results = np.load(pairtee_results_fn)
    parent_vectors = results['struct']
    llhs = results['llh']
    adj_matrices = []

    data = []
    for parents_vector, llh in zip(parent_vectors, llhs):
        adj_matrix = get_adj_matrix_from_parents(parents_vector)
        data.append((torch.tensor(adj_matrix, dtype = torch.float32), llh))
    return data

# Adapted from pairtree lib/vaf_plotter.py
def _get_F_matrix(clustered_vars, num_variants, num_samples, should_correct_vaf):
    nclusters = len(clustered_vars)
    # variant x sample matrix (we'll transpose at the end)
    F = np.zeros((num_variants, num_samples))
    vidx = 0 # variant index
    ordered_variant_ids = []
    for cidx, cluster in enumerate(clustered_vars):
        assert len(cluster) > 0

        K = lambda V: int(V['id'][1:])
        cluster = sorted(cluster, key=K)

        for V in cluster:
            V = dict(V)
            if should_correct_vaf:
                V['vaf'] = V['vaf'] / V['omega_v']
            for sidx in range(len(V['vaf'])):
                F[vidx][sidx] = V['vaf'][sidx]
            ordered_variant_ids.append(V['id'])
            vidx += 1
    return F, ordered_variant_ids

def get_F_matrix(clusters, variants, sampnames, should_correct_vaf):
    # Only get the number of variants that are present in clusters
    num_variants = sum([len(sublist) for sublist in params['clusters']])
    variants = {vid: dict(variants[vid]) for vid in variants.keys()}
    clustered_vars = [[variants[vid] for vid in C] for C in clusters]
    for cidx, cluster in enumerate(clustered_vars):
        if supervars is not None:
            supervars[cidx]['cluster'] = cidx
        for var in cluster:
            var['cluster'] = cidx
    F, ordered_variant_ids = _get_F_matrix(clustered_vars, num_variants, len(sampnames), should_correct_vaf)
    return F, ordered_variant_ids, cluster

def filter_F_matrix(samples, F):
    unique_sites = []
    indices_to_keep = []
    for i, sample in enumerate(samples):
        alpha_sample_name = sample.translate(remove_digits)
        if (alpha_sample_name not in unique_sites):
            indices_to_keep.append(i)
            unique_sites.append(alpha_sample_name)
    return F[indices_to_keep,:], unique_sites


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def get_B(clusters, parents, ordered_variant_ids):
    '''
    clusters: list of lists where each inner list has the variants in that cluster e.g. [['s0', 's1'], ['s3'], ['s2', 's4']]
    parents: K-1 length vector, where parents[i] provides the parent of node i+1
    ordered_variant_ids: the variant ids in the order that they are in the columns of the F matrix e.g. ['s0', 's1', 's4', 's2', 's3']
    '''
    parents = [p-1 for p in parents]
    def _get_ancestors(node_idx):
        parent_node = parents[node_idx]
        if parent_node == -1:
            return []
        return clusters[parent_node] + _get_ancestors(parent_node)

    num_variants = len(ordered_variant_ids)
    num_subclones = len(clusters)
    B = np.zeros((num_subclones, num_variants))
    for cidx, cluster_variants in enumerate(clusters):
        # get all ancestral variants
        all_variants = _get_ancestors(cidx)
        all_variants.extend(cluster_variants)
        for variant in all_variants:
            vidx = ordered_variant_ids.index(variant)
            B[cidx][vidx] = 1
    return B



import numpy as np

def load_input_data(ssm_filename, params_filename):
    variants = inputparser.load_ssms(ssm_filename)
    params = inputparser.load_params(params_filename)
    if 'garbage' not in params:
        params['garbage'] = []
    if 'clusters' not in params or len(params['clusters']) == 0:
        params['clusters'] = [[vid] for vid in variants.keys() if vid not in params['garbage']]
    supervars = clustermaker.make_cluster_supervars(params['clusters'], variants)
    supervars = [supervars[vid] for vid in common.sort_vids(supervars.keys())]
    return variants, params, supervars

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

def get_adj_matrix_from_pairtree_tree(parents):
    K = len(parents) + 1
    T = np.eye(K)
    T[parents,np.arange(1, K)] = 1
    I = np.identity(T.shape[0])
    T = np.logical_xor(T,I).astype(int) # remove self-loops
    # remove the normal subclone
    T = np.delete(T, 0, 0)
    T = np.delete(T, 0, 1)
    return T


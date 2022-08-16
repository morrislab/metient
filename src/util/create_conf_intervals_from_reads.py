'''
This module takes simulated read data and observed clusters from Machina
(El Kebir et.al.; see: https://github.com/raphael-group/machina/tree/master/data/sims)
and creates pooled confidence intervals from this data (pooled by mutation cluster)

Confidence intervals can  be obtained by first pooling for each sample
the read counts of the mutations that belong to the same cluster followed by
using a beta distribution (see MACHINA supplement A.1.2 for more info).

Author: Divya Koyyalagunta
Jul 21, 2022

'''
import sys
import os
import pandas as pd

import numpy
from scipy.stats import beta
from scipy.stats import norm

CONFIDENCE = 0.95

# Adapted from Machina
def binomial_hpdr(n, N, pct, a=1, b=1, n_pbins=1e3):
    """
    Function computes the posterior mode along with the upper and lower bounds of the
    **Highest Posterior Density Region**.

    Parameters
    ----------
    n: number of successes
    N: sample size
    pct: the size of the confidence interval (between 0 and 1)
    a: the alpha hyper-parameter for the Beta distribution used as a prior (Default=1)
    b: the beta hyper-parameter for the Beta distribution used as a prior (Default=1)
    n_pbins: the number of bins to segment the p_range into (Default=1e3)

    Returns
    -------
    A tuple that contains the mode as well as the lower and upper bounds of the interval
    (mode, lower, upper)

    """
    # fixed random variable object for posterior Beta distribution
    rv = beta(n+a, N-n+b)
    # determine the mode and standard deviation of the posterior
    stdev = rv.stats('v')**0.5
    mode = (n+a-1.)/(N+a+b-2.)
    # compute the number of sigma that corresponds to this confidence
    # this is used to set the rough range of possible success probabilities
    n_sigma = numpy.ceil(norm.ppf( (1+pct)/2. ))+1
    # set the min and max values for success probability
    max_p = mode + n_sigma * stdev
    if max_p > 1:
        max_p = 1.
    min_p = mode - n_sigma * stdev
    if min_p > 1:
        min_p = 1.
    # make the range of success probabilities
    p_range = numpy.linspace(min_p, max_p, int(n_pbins+1))
    # construct the probability mass function over the given range
    if mode > 0.5:
        sf = rv.sf(p_range)
        pmf = sf[:-1] - sf[1:]
    else:
        cdf = rv.cdf(p_range)
        pmf = cdf[1:] - cdf[:-1]
    # find the upper and lower bounds of the interval
    sorted_idxs = numpy.argsort( pmf )[::-1]
    cumsum = numpy.cumsum( numpy.sort(pmf)[::-1] )
    j = numpy.argmin( numpy.abs(cumsum - pct) )
    upper = p_range[ (sorted_idxs[:j+1]).max()+1 ]
    lower = p_range[ (sorted_idxs[:j+1]).min() ]

    return (mode, lower, upper)

def get_ub(row, sam, corrected_confidence):
    v=binomial_hpdr(row['var-'+sam], row['var-'+sam]+row['ref-'+sam], corrected_confidence)
    return v[2]


def get_lb(row, sam, corrected_confidence, cutoff=False,):
    v=binomial_hpdr(row['var-'+sam], row['var-'+sam]+row['ref-'+sam], corrected_confidence)
    mval = v[1]
    if cutoff and mval < 0.01:
        mval = 0
    return mval

def get_mean(row, sam, corrected_confidence):
    v=binomial_hpdr(row['var-'+sam], row['var-'+sam]+row['ref-'+sam], corrected_confidence)
    mval = v[0]
    return mval

def print_char(row, i, sam, cluster_index_to_variant_indices, use_char_idx_as_char_label):
    char_label = row.name if use_char_idx_as_char_label else cluster_index_to_variant_indices[row.name]
    return "\t".join(map(str,[i, sam, i, sam, row.name, char_label, max(row['lb-'+sam] * 2, 0), min(1, 2 * row['ub-'+sam]), int(row['ref-'+sam]), int(row['var-'+sam])]))+"\n"

def write(reads_filename, cluster_filename, out_directory, cluster_split_function=None, use_char_idx_as_char_label=False):
    '''
    To obtain mutation clusters and their frequencies, we use the clustering
    procedure of AncesTree

    To infer a confidence interval on the frequency of mutation cluster C in each sample p,
    we combine the read counts for all mutations in the same mutation cluster C,
    yielding a combined variant read count and combined total read count (see MACHINA supplement A.1.2 for more info)

    cluster_split_function: lambda function for how to get the variant names from the cluster name. e.g.
    some clusters may be mutation names separated by ";" or "_". "MEM13;GHG24" -> ['MEM13', 'GHG24']

    use_char_idx_as_char_label: if True, uses the cluster index as the cluster label in the output tsv.
    useful in cases where the cluster labels get very long
    '''

    read_data = pd.read_table(reads_filename, skiprows=3)
    # Map variant IDs to observed clusters IDs
    variant_id_to_cluster_idx_map = {}
    cluster_index_to_variant_indices = {} # this will be saved as the character label
    with open(cluster_filename) as f:
        for i,line in enumerate(f):
            if cluster_split_function != None:
                var_indices = cluster_split_function(line.strip())
            else:
                var_indices = [x for x in line.strip().split(';') if x.isnumeric()]
            for l in var_indices:
                variant_id_to_cluster_idx_map[l] = i
            cluster_index_to_variant_indices[i] = line.strip()

    variants = read_data['character_label'].unique()
    # TODO: should this be per sample or per anatomical site
    sample_labels = read_data['anatomical_site_label'].unique()

    print("num variants:", len(variants))
    print("anatomical site labels:", sample_labels)

    # Reformat the read data into a dataframe with the reference and variant reads for each variant ID
    # with clusters annotated so that we can pool the read data by cluster
    cols = ["cluster"] + ['ref-'+c for c in sample_labels] + ['var-'+c for c in sample_labels]
    pooled_data = []
    indices = []
    for v in variants:
        if v in variant_id_to_cluster_idx_map:
            cluster_idx = variant_id_to_cluster_idx_map[v]
            row = [cluster_idx]
            variant_subset = read_data[read_data['character_label']==v]
            for col in ["ref", "var"]:
                for sample_label in sample_labels:
                    # TODO: if we are using both primary samples, we need to somehow incorporate both
                    # but right now we're just taking the first primary sample... (.iloc[0])
                    var_sample_sub = variant_subset[variant_subset['anatomical_site_label'] == sample_label].iloc[0]
                    row.append(int(var_sample_sub[col]))
            pooled_data.append(row)
            indices.append(v)

    pooled_df = pd.DataFrame(pooled_data, index=indices, columns=cols)

    ctable = pooled_df.groupby('cluster').sum()

    nsamples = len([c for c in ctable.columns if c.startswith('ref')])
    nclusters = len(ctable)
    corrected_confidence = 1-((1.-CONFIDENCE)/(nsamples*nclusters))

    assert(corrected_confidence > CONFIDENCE)
    assert(corrected_confidence < 1.0)

    ctable = pooled_df.groupby('cluster').sum()
    for sample in sample_labels:
        ctable['ub-'+sample]= ctable.apply(get_ub, args=[sample, corrected_confidence], axis=1)
        ctable['lb-'+sample]= ctable.apply(get_lb, args=[sample, corrected_confidence, False], axis=1)
        ctable[sample]= ctable.apply(get_mean, args=[sample, corrected_confidence], axis=1)

    ctable_cutoff = pooled_df.groupby('cluster').sum()
    for sample in sample_labels:
        ctable_cutoff['ub-'+sample]= ctable.apply(get_ub, args=[sample, corrected_confidence], axis=1)
        ctable_cutoff['lb-'+sample]= ctable.apply(get_lb, args=[sample, corrected_confidence, True], axis=1)
        ctable_cutoff[sample]= ctable.apply(get_mean, args=[sample, corrected_confidence], axis=1)

    rows = ["5 #anatomical sites\n5 #samples\n9 #mutations\n#sample_index\tsample_label\tanatomical_site_index\tanatomical_site_label\tcharacter_index\tcharacter_label\tf_lb\tf_ub\tref\tvar\n",]
    for i, sam in enumerate(sample_labels):
        rows += list(ctable_cutoff.apply(print_char, args=[i, sam, cluster_index_to_variant_indices, use_char_idx_as_char_label], axis=1))

    basename = os.path.basename(reads_filename).replace("reads_", "").replace(".tsv", "")
    with open(os.path.join(out_directory, basename+"_"+str(CONFIDENCE)+".tsv"), 'w') as f:
        for line in rows:
            f.write(line)

if __name__=="__main__":
    if len(sys.argv) != 4:
        print("USAGE: python create_conf_intervals_from_reads.py [reads_file] [cluster_file] [out_directory]")
        quit()

    reads_filename = sys.argv[1]
    cluster_filename = sys.argv[2]
    out_directory = sys.argv[3]

    write(reads_filename, cluster_filename, out_directory)

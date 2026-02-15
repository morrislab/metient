import pyreadr
import torch 
from metient import *
from metient.util.globals import *
import pandas as pd
from metient.util import plotting_util as putil
from metient.util import vertex_labeling_util as vutil
import glob
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def split_pattern_clonality(full_pattern):
    clonality = full_pattern.split(" ")[0]
    pattern = " ".join(full_pattern.split(" ")[1:]).replace(" seeding", "")
    return pattern, clonality

def is_ln(met_site):
    return met_site.startswith("LN") or "lymph" in met_site or "Lymph" in met_site

pts = set()
def get_root_clone_presence_in_met(idx_to_label, primary, A):
    root_idx = vutil.get_root_index(A)
    root_node_muts = ";".join(idx_to_label[root_idx][0])
    root_clone_site = "None"
    root_clone_sites = []
    for i in idx_to_label:
        is_observed_clone = idx_to_label[i][1]
        muts = idx_to_label[i][0]
        # -1 because the last label includes the site label
        if i != root_idx and is_observed_clone and ";".join(muts[:-1])==root_node_muts and muts[-1] != primary and 'primary' not in muts[-1]:
            root_clone_sites.append(muts[-1])
    is_in_ln, is_in_distant = False, False
    for site in root_clone_sites:
        if is_ln(site):
            is_in_ln = True
        else:
            is_in_distant = True
    if is_in_ln and is_in_distant:
        root_clone_site = "Both"
    elif is_in_ln:
        root_clone_site = "Lymph node"
    elif is_in_distant:
        root_clone_site = "Distant"
   
    return root_clone_site

def get_seeding_info(index_order, pkl):
    V = torch.tensor(pkl[OUT_LABElING_KEY][index_order[0][0]]) 
    A = adjacency_matrix_from_parents(pkl[OUT_PARENTS_KEY][index_order[0][0]])

    idx_to_label = pkl[OUT_IDX_LABEL_KEY][index_order[0][0]]
    pattern = putil.seeding_pattern(V,A)
    G = putil.migration_graph(V,A)
    num_migs_polyclonal = (G>1).sum().item() / (G>0).sum().item() *100
    num_ss = (G!=0).any(dim=1).sum().item()
    seeding_clusters = putil.seeding_clusters(V, A, idx_to_label)
    return pattern, num_migs_polyclonal, num_ss, seeding_clusters

def get_pattern_clonality_phyleticity(V,A,idx_to_label):
    pattern = putil.seeding_pattern(V,A)
    st_clonality = putil.site_clonality(V,A)
    gen_clonality = putil.genetic_clonality(V,A,idx_to_label)
    phyl = putil.phyleticity(V,A,idx_to_label)
    tracerx_phyl = putil.tracerx_phyleticity(V,A,idx_to_label)
    return pattern, st_clonality, gen_clonality, phyl, tracerx_phyl

def get_num_internal_nodes(idx_to_label):
    num_internal_nodes = 0
    for idx in idx_to_label:
        if idx_to_label[idx][1] == False:
            num_internal_nodes += 1
    return num_internal_nodes

def get_all_migration_percentages(pkl, sites):
    """Calculate migration percentages for all trees in a pickle file.
    
    Args:
        pkl (dict): Pickle file containing tree data with OUT_LABElING_KEY and OUT_PARENTS_KEY
        sites (list): List of anatomical site names
        
    Returns:
        tuple: Contains:
            - all_pct_migs_polyclonal (list): Percentage of polyclonal migrations for each solution
            - all_num_ss (list): Number of seeding sites for each solution  
            - all_pct_met_sites (list): Percentage of metastatic sites receiving seeding for each solution
    """
    Vs = pkl[OUT_LABElING_KEY]
    parents = pkl[OUT_PARENTS_KEY]

    all_pct_migs_polyclonal = []
    all_num_ss = []
    all_pct_met_sites = []
    for V,p in zip(Vs,parents):
        A = adjacency_matrix_from_parents(p)
        G = putil.migration_graph(torch.tensor(V),torch.tensor(A))
        x,y,z = get_migration_percentages(G, sites)
        all_pct_migs_polyclonal.append(x)
        all_num_ss.append(y)
        all_pct_met_sites.append(z)

    return all_pct_migs_polyclonal, all_num_ss, all_pct_met_sites

def get_migration_percentages(G, sites):
    """Calculate migration percentages from migration graph G and sites list.
    
    Args:
        G (torch.Tensor): Migration graph tensor showing number of migrations between sites
        sites (list): List of anatomical site names
        
    Returns:
        tuple: Contains:
            - percent_migrations_polyclonal (float): Percentage of migrations that are polyclonal
            - num_seeding_sites (int): Number of sites that receive seeding events
            - percent_met_sites (float): Percentage of metastatic sites that receive seeding events
    """
    percent_migrations_polyclonal = (G>1).sum().item() / (G>0).sum().item() *100
    num_seeding_sites = (G!=0).any(dim=1).sum().item()
    percent_met_sites = (num_seeding_sites-1)/(len(sites)-1) * 100
    return percent_migrations_polyclonal, num_seeding_sites, percent_met_sites

def build_top_calibrate_tree_df(result_dirs, dataset_names, bootstrap_fn):

    bootstrap_df = pd.read_csv(bootstrap_fn).drop(columns=['Unnamed: 0'])

    data = []
    avg_num_nodes = []
    for calibrate_dir,dataset in zip(result_dirs, dataset_names):

        # Use glob to get the list of matching files
        matching_files = glob.glob(f'{calibrate_dir}/*pkl.gz')
        patients = [m.split("/")[-1].split("_")[0] for m in matching_files]
        print(dataset, len(patients))

        for fn in matching_files:
            pid = fn.split("/")[-1].split("_")[0]
            with gzip.open(fn, 'rb') as f:
                pkl = pickle.load(f)
            # Best calibrated tree
            V = torch.tensor(pkl[OUT_LABElING_KEY][0])
            A = adjacency_matrix_from_parents(pkl[OUT_PARENTS_KEY][0])
        
            losses = [l.item() for l in pkl[OUT_LOSSES_KEY]]
            sites = pkl[OUT_SITES_KEY]
            idx_to_label = pkl[OUT_IDX_LABEL_KEY][0]
            avg_num_nodes.append(get_num_internal_nodes(idx_to_label))
            cal_pattern,cal_st_clonality,cal_gen_clonality,cal_phyletic,cal_tracerx_phyletic =  get_pattern_clonality_phyleticity(V, A, idx_to_label)
            G = putil.migration_graph(V,A)
            cal_pct_migs_polyclonal, cal_num_ss, cal_pct_met_sites = get_migration_percentages(G, sites)
            
            cal_seeding_clusters = putil.seeding_clusters(V, A, idx_to_label)
            primary = pkl[OUT_PRIMARY_KEY]
            
            # How many patients have root clone observed in a met 
            root_clone_in_met = get_root_clone_presence_in_met(idx_to_label, primary, A)
            
            # Gather the solution with lowest migration number, comigration number, or seeding site
            loss_dicts = pkl[OUT_LOSS_DICT_KEY]
            pars_metrics = []
            for l in loss_dicts:
                pars_metrics.append((int(l[MIG_KEY]), int(l[COMIG_KEY]), int(l[SEEDING_KEY])))
            
            mig_sorted_indices = sorted(enumerate(pars_metrics), key=lambda x: (x[1][0], x[1][1], x[1][2]))
            mig_num_pattern, mig_num_migs_polyclonal, mig_num_ss, _ = get_seeding_info(mig_sorted_indices, pkl)
            
            comig_sorted_indices = sorted(enumerate(pars_metrics), key=lambda x: (x[1][1], x[1][0], x[1][2]))
            comig_num_pattern, comig_num_migs_polyclonal, comig_num_ss, _ = get_seeding_info(comig_sorted_indices, pkl)

            ss_sorted_indices = sorted(enumerate(pars_metrics), key=lambda x: (x[1][2], x[1][0], x[1][1]))
            ss_num_pattern, ss_num_migs_polyclonal, ss_num_ss, ss_seeding_clusters = get_seeding_info(ss_sorted_indices, pkl)
            
            num_trees_on_pareto = len(pars_metrics)
            mult_trees_w_same_pars_metrics = True if len(set(pars_metrics)) == 1 and len(pars_metrics) > 1 else False
            mult_trees_w_diff_pars_metrics = True if len(set(pars_metrics)) != 1 else False

            major_vote_classification = get_majority_vote_classification(bootstrap_df, dataset, pkl)
            major_vote_pattern, major_vote_st_clonality, major_vote_gen_clonality, major_vote_phyletic, all_bootstrap_patterns = major_vote_classification
            
            all_pct_migs_polyclonal, all_num_ss, all_pct_met_sites = get_all_migration_percentages(pkl, sites)

            subtype = "N/A"

            data.append([dataset, pid, fn, losses[0], subtype, len(sites), num_trees_on_pareto,mult_trees_w_same_pars_metrics,mult_trees_w_diff_pars_metrics, root_clone_in_met,
                        cal_pattern, cal_st_clonality, cal_gen_clonality, cal_phyletic, cal_tracerx_phyletic, cal_pct_migs_polyclonal,cal_num_ss,cal_seeding_clusters,cal_pct_met_sites,
                        mig_num_pattern, mig_num_migs_polyclonal,mig_num_ss,
                        comig_num_pattern, comig_num_migs_polyclonal,comig_num_ss,
                        ss_num_pattern,ss_num_migs_polyclonal,ss_num_ss, ss_seeding_clusters, pars_metrics,
                        major_vote_pattern, major_vote_st_clonality, major_vote_gen_clonality, major_vote_phyletic,
                        all_pct_migs_polyclonal, all_num_ss, all_pct_met_sites, all_bootstrap_patterns])
    

    trees_df = pd.DataFrame(data, columns=['Dataset', "Patient id", "Filename", "Lowest cal loss", "Subtype", 'Num sites', 'Num trees on Pareto', 'Multiple trees w/ same pars metrics','Multiple trees w/ diff pars metrics', "Root clone observed site",
                                        'Top cal seeding pattern', 'Top cal site clonality', 'Top cal genetic clonality', 'Top cal phyleticity', 'Top cal tracerx phyleticity', 'Top cal % migs poly','Top cal num ss', 'Top cal seeding clusters', 'Top cal % metastatic sites',
                                        'Lowest mig seeding pattern', 'Lowest mig % migs poly','Lowest mig num ss',
                                        'Lowest comig seeding pattern', 'Lowest comig % migs poly','Lowest comig num ss',
                                        'Lowest ss seeding pattern', 'Lowest ss % migs poly','Lowest ss num ss', 'Lowest ss seeding clusters', 'Pars metrics',
                                        'Majority vote seeding pattern', 'Majority vote site clonality', 'Majority vote genetic clonality', 'Majority vote phyleticity',
                                        'All % migs poly', 'All num ss', 'All % met sites', 'All bootstrap patterns'])

    # When there are patients with multiple possible primaries, only keep the solution with the lowest loss 
    idx = trees_df.groupby(['Dataset', 'Patient id'])['Lowest cal loss'].idxmin()
    trees_df = trees_df.loc[idx].reset_index(drop=True)

    print("Average number of nodes", np.mean(avg_num_nodes))
    return trees_df

def get_polyclonally_seeded_sites_and_total_seeded_sites(trees_df):
    '''
    Get the number of polyclonally seeded sites and the total number of seeded sites
    using the best calibrated tree for each patient
    '''

    total_seeded_sites, polyclonally_seeded_sites = 0, 0
    for _,row in trees_df.iterrows():
        fn = row['Filename']
        with gzip.open(fn, 'rb') as f:
            pkl = pickle.load(f)
        # Best calibrated tree
        V = torch.tensor(pkl[OUT_LABElING_KEY][0])
        A = adjacency_matrix_from_parents(pkl[OUT_PARENTS_KEY][0])
        G = putil.migration_graph(V,A)
        
        # Count number of columns (sites) that are seeded
        num_sites_with_migs = (G!=0).any(dim=0).sum().item()
        total_seeded_sites += num_sites_with_migs
        # Count number of columns (sites) that are seeded by MORE than one clone
        polyclonally_seeded_sites += (G>1).any(dim=0).sum().item()

    return polyclonally_seeded_sites, total_seeded_sites

def get_soln_idx_to_probability(loss_dicts, thetas, prnt=False):
    # Calculate losses for each solution
    losses = []
    for loss_dict in loss_dicts:
        # Reconstruct loss using the bootstrapped weights
        weights = Weights(mig=thetas[0]*PARS_METRIC_MULTIPLIER, comig=thetas[1]*PARS_METRIC_MULTIPLIER, seed_site=thetas[2]*PARS_METRIC_MULTIPLIER,
                              gen_dist=1.0, organotrop=1.0)
        m, c, s, g, o, e = loss_dict[MIG_KEY], loss_dict[COMIG_KEY], loss_dict[SEEDING_KEY], loss_dict[GEN_DIST_KEY], loss_dict[ORGANOTROP_KEY], loss_dict[ENTROPY_KEY]
        loss = vutil.clone_tree_labeling_loss_with_computed_metrics(m, c, s, g, o, e, weights)
        losses.append(loss)
    
    probabilities = putil.losses_to_probabilities(losses, temperature=0.05)
    soln_idx_to_prob = {idx: prob for idx, prob in enumerate(probabilities)}

    if prnt:
        print("Losses:",losses)
        print("Probabilities:", soln_idx_to_prob)

    return soln_idx_to_prob


def get_metric_probabilities(soln_idx_to_data, soln_idx_to_prob):
    """
    Calculate weighted probabilities for each metric value.
    
    Args:
        soln_idx_to_data (dict): Maps solution indices to (pattern, st_clonality, gen_clonality, phyl)
        soln_idx_to_prob (dict): Maps solution indices to probabilities
        
    Returns:
        tuple: Dictionaries mapping each metric value to its weighted probability
    """
    # Initialize probability dictionaries
    pattern_probs = {}
    st_clonality_probs = {}
    gen_clonality_probs = {} 
    phyleticity_probs = {}
    
    # Calculate weighted probabilities
    for soln_idx, prob in soln_idx_to_prob.items():
        pattern, st_clonality, gen_clonality, phyl = soln_idx_to_data[soln_idx]
        
        pattern_probs[pattern] = pattern_probs.get(pattern, 0) + prob
        st_clonality_probs[st_clonality] = st_clonality_probs.get(st_clonality, 0) + prob
        gen_clonality_probs[gen_clonality] = gen_clonality_probs.get(gen_clonality, 0) + prob
        phyleticity_probs[phyl] = phyleticity_probs.get(phyl, 0) + prob
        
    return pattern_probs, st_clonality_probs, gen_clonality_probs, phyleticity_probs

def get_most_probable_metrics(metric_probabilities):
    """
    Get the most probable value for each metric.
    
    Args:
        metric_probabilities (tuple): Probability dictionaries for each metric
        
    Returns:
        tuple: Most probable value for each metric
    """
    pattern_probs, st_clonality_probs, gen_clonality_probs, phyleticity_probs = metric_probabilities
    
    most_probable_pattern = max(pattern_probs, key=pattern_probs.get)
    most_probable_st_clonality = max(st_clonality_probs, key=st_clonality_probs.get)
    most_probable_gen_clonality = max(gen_clonality_probs, key=gen_clonality_probs.get)
    most_probable_phyleticity = max(phyleticity_probs, key=phyleticity_probs.get)
    
    return most_probable_pattern, most_probable_st_clonality, most_probable_gen_clonality, most_probable_phyleticity

def _get_weighted_classifications(soln_idx_to_data, soln_idx_to_prob):
    """
    Perform a weighted classification based on the probability of each solution.
    
    Args:

    Returns:
        tuple: Most probable value for each metric
    """
    # Get weighted probabilities for each metric
    metric_probabilities = get_metric_probabilities(soln_idx_to_data, soln_idx_to_prob)
    
    # Get most probable values
    most_probable_metrics = get_most_probable_metrics(metric_probabilities)

    return most_probable_metrics

def get_majority_vote_classification(bootstrap_df, dataset, pckl, prnt=False):
    
    # Get bootstrapped thetas for this cohort
    dataset_thetas = list(bootstrap_df[bootstrap_df['dataset']==dataset]['Fit theta'])

    # Get data from pickle
    loss_dicts = pckl[OUT_LOSS_DICT_KEY]
    parents = pckl[OUT_PARENTS_KEY]
    As = [adjacency_matrix_from_parents(p) for p in parents]
    Vs = pckl[OUT_LABElING_KEY]
    node_infos = pckl[OUT_IDX_LABEL_KEY]

    # Map data from each solution
    soln_idx_to_data = {}
    for i, (V,A,node_info) in enumerate(zip(Vs,As,node_infos)):
        pattern, st_clonality, gen_clonality, phyl, _ = get_pattern_clonality_phyleticity(V, A, node_info)
        soln_idx_to_data[i] = pattern, st_clonality, gen_clonality, phyl

    # Get weighted classifications for each bootstrapped theta
    weighted_classifications = []
    last_weighted_classification = None
    for i in range(0, len(dataset_thetas), 3):
        one_samp_thetas = dataset_thetas[i:i+3]
        soln_idx_to_probability = get_soln_idx_to_probability(loss_dicts, one_samp_thetas)
        weighted_classification = _get_weighted_classifications(soln_idx_to_data, soln_idx_to_probability)
        weighted_classifications.append(weighted_classification)

        if prnt and last_weighted_classification != weighted_classification and last_weighted_classification != None:
            print("Thetas", one_samp_thetas)
            get_soln_idx_to_probability(loss_dicts, one_samp_thetas, prnt=True)
            print("Last weighted classification: ", last_weighted_classification)
            print("Weighted classification: ", weighted_classification)
            print("----------------------------------------------------------------")
        last_weighted_classification = weighted_classification
    

    # Convert list of tuples into separate lists for each metric
    patterns = [x[0] for x in weighted_classifications]
    site_clonalities = [x[1] for x in weighted_classifications]
    genetic_clonalities = [x[2] for x in weighted_classifications]
    phyleticities = [x[3] for x in weighted_classifications]
    
    # Get most common value for each metric
    majority_pattern = max(set(patterns), key=patterns.count)
    majority_site_clonality = max(set(site_clonalities), key=site_clonalities.count)
    majority_genetic_clonality = max(set(genetic_clonalities), key=genetic_clonalities.count)
    majority_phyleticity = max(set(phyleticities), key=phyleticities.count)

    
    majority_vote_classification = majority_pattern, majority_site_clonality, majority_genetic_clonality, majority_phyleticity, patterns
    return majority_vote_classification


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def get_patient_pars_metrics(trees_df, pids):
    
    data = [['Migrations', 'Comigrations', 'Seeding sites']]
    for pid in pids:
        patient_df = trees_df[trees_df['Patient id']==pid]
        unique_metrics = []
        for m in list(patient_df['Pars metrics'])[0]:
            if m not in unique_metrics:
                unique_metrics.append(m)
        print(unique_metrics)
        reversed_metrics = unique_metrics[::-1]
        processed = [[x[0],x[1]*2,x[2]*3] for x in reversed_metrics]
        data.append((pid,processed))

    return data
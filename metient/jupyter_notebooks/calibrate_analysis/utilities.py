import pyreadr
import torch 
from metient.metient import *
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

def split_pattern_clonality(full_pattern):
    clonality = full_pattern.split(" ")[0]
    pattern = " ".join(full_pattern.split(" ")[1:]).replace(" seeding", "")
    return pattern, clonality

def get_tracerx_subtype(pid):
    # Get NSCLC subtype (LUAD and LUSC)
    tracerx_patient_info = pyreadr.read_r(os.path.join('/data/morrisq/divyak/data/tracerx_nsclc_2023/20221109_TRACERx421_all_patient_df.rds'))[None]
    tracerx_patient_info['histology_multi_full_genomically.confirmed'].value_counts()

    subtype = tracerx_patient_info[tracerx_patient_info['cruk_id']==pid]['histology_multi_full_genomically.confirmed'].item()
    subtype = "LUAD" if "LUAD" in subtype else subtype
    if subtype == "Other":
        print("Not LUAD or LUSC subtype", pid)
        return "N/A"
    return subtype

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
    indices, values, size = pkl[OUT_ADJ_KEY][index_order[0][0]]
    A = torch.sparse_coo_tensor(indices.cpu(), values.cpu(), tuple(size))
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

def build_top_calibrate_tree_df(result_dirs, dataset_names):

    data = []
    avg_num_nodes = []
    for calibrate_dir,dataset in zip(result_dirs, dataset_names):
        # Use glob to get the list of matching files
        matching_files = glob.glob(f'{calibrate_dir}/*pkl.gz')
        patients = [m.split("/")[-1].split("_")[0] for m in matching_files]
        print(dataset, len(patients))
        for fn in matching_files:
            with gzip.open(fn, 'rb') as f:
                
                pid = fn.split("/")[-1].split("_")[0]
                pkl = pickle.load(f)
                # Best calibrated tree
                V = torch.tensor(pkl[OUT_LABElING_KEY][0])
                indices, values, size = pkl[OUT_ADJ_KEY][0]
                A = torch.sparse_coo_tensor(indices.cpu(), values.cpu(), tuple(size))
                losses = [l.item() for l in pkl[OUT_LOSSES_KEY]]
                sites = pkl[OUT_SITES_KEY]
                idx_to_label = pkl[OUT_IDX_LABEL_KEY][0]
                avg_num_nodes.append(get_num_internal_nodes(idx_to_label))
                cal_pattern,cal_st_clonality,cal_gen_clonality,cal_phyletic,cal_tracerx_phyletic =  get_pattern_clonality_phyleticity(V, A, idx_to_label)
                G = putil.migration_graph(V,A)
                cal_num_migs_polyclonal = (G>1).sum().item() / (G>0).sum().item() *100
                cal_num_ss = (G!=0).any(dim=1).sum().item()
                cal_prop_met_sites = (cal_num_ss-1)/(len(sites)-1) * 100
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
                
                subtype = "N/A"
                if dataset == 'NSCLC':
                    subtype = get_tracerx_subtype(pid)

                data.append([dataset, pid, fn, losses[0], subtype, len(sites), num_trees_on_pareto,mult_trees_w_same_pars_metrics,mult_trees_w_diff_pars_metrics, root_clone_in_met,
                            cal_pattern, cal_st_clonality, cal_gen_clonality, cal_phyletic, cal_tracerx_phyletic, cal_num_migs_polyclonal,cal_num_ss,cal_seeding_clusters,cal_prop_met_sites,
                            mig_num_pattern, mig_num_migs_polyclonal,mig_num_ss,
                            comig_num_pattern, comig_num_migs_polyclonal,comig_num_ss,
                            ss_num_pattern,ss_num_migs_polyclonal,ss_num_ss, ss_seeding_clusters, pars_metrics])
        

    trees_df = pd.DataFrame(data, columns=['Dataset', "Patient id", "Filename", "Lowest cal loss", "Subtype", 'Num sites', 'Num trees on Pareto', 'Multiple trees w/ same pars metrics','Multiple trees w/ diff pars metrics', "Root clone observed site",
                                        'Top cal seeding pattern', 'Top cal site clonality', 'Top cal genetic clonality', 'Top cal phyleticity', 'Top cal tracerx phyleticity', 'Top cal % migs poly','Top cal num ss', 'Top cal seeding clusters', 'Top cal % metastatic sites',
                                        'Lowest mig seeding pattern', 'Lowest mig % migs poly','Lowest mig num ss',
                                        'Lowest comig seeding pattern', 'Lowest comig % migs poly','Lowest comig num ss',
                                        'Lowest ss seeding pattern', 'Lowest ss % migs poly','Lowest ss num ss', 'Lowest ss seeding clusters', 'Pars metrics'])

    # When there are patients with multiple possible primaries, only keep the solution with the lowest loss 
    idx = trees_df.groupby(['Dataset', 'Patient id'])['Lowest cal loss'].idxmin()
    trees_df = trees_df.loc[idx].reset_index(drop=True)

    print("Average number of nodes", np.mean(avg_num_nodes))
    return trees_df



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
    
    data = [['Migration #', 'Comigration #', 'Seeding sites']]
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
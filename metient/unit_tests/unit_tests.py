import unittest
import torch
import os

from src.util import vertex_labeling_util as vert_util
from src.util import data_extraction_util as data_util
from src.util import plotting_util as plot_util
from src.util import pairtree_data_extraction_util as pt_util


class TestLabeledTree(unittest.TestCase):

	def test_init(self):
		bad_tree = torch.tensor([[0,1]])
		good_tree = torch.tensor([[0,1,0], [1,0,0], [1,0,0]])

		mismatched_labeling = torch.tensor([[0,1],[0,1], [1,0]])
		matched_labeling = torch.tensor([[0,1,0],[1,0,1]])

		U = torch.tensor([[0,0], [0,0]])
		branch_lengths = torch.tensor([[0,0], [0,0]])

		with self.assertRaises(ValueError):
			vert_util.LabeledTree(bad_tree, matched_labeling, U, branch_lengths)

		with self.assertRaises(ValueError):
			vert_util.LabeledTree(good_tree, mismatched_labeling, U, branch_lengths)

		vert_util.LabeledTree(good_tree, matched_labeling, U, branch_lengths)

	def test_equal_and_hash(self):
		# Test uniqueness of trees, which is determined by
		# 2 attributes: tree (adjacency matrix) and vertex labeling
		tree1 = torch.tensor([[1,0], [0,1]])
		labeling1 = torch.tensor([[0,1],[0,1]])

		U = torch.tensor([[0,0], [0,0]])
		branch_lengths = torch.tensor([[0,0], [0,0]])

		labeled_tree1 = vert_util.LabeledTree(tree1, labeling1, U, branch_lengths)
		labeled_tree2 = vert_util.LabeledTree(tree1, labeling1, U, branch_lengths)

		tree_set = set()
		tree_set.add(labeled_tree1)
		tree_set.add(labeled_tree2)

		self.assertEqual(labeled_tree1, labeled_tree2)
		self.assertEqual(hash(labeled_tree1), hash(labeled_tree2))

		self.assertEqual(len(tree_set), 1)

		tree2 = torch.tensor([[0,0],[0,1]])
		labeling2 = torch.tensor([[1,0],[0,1]])

		labeled_tree3 = vert_util.LabeledTree(tree1, labeling2, U, branch_lengths)
		self.assertNotEqual(labeled_tree1, labeled_tree3)
		self.assertNotEqual(hash(labeled_tree1), hash(labeled_tree3))

		labeled_tree4 = vert_util.LabeledTree(tree2, labeling1, U, branch_lengths)
		self.assertNotEqual(labeled_tree1, labeled_tree4)
		self.assertNotEqual(hash(labeled_tree1), hash(labeled_tree4))

		labeled_tree5 = vert_util.LabeledTree(tree2, labeling2, U, branch_lengths)
		self.assertNotEqual(labeled_tree1, labeled_tree5)
		self.assertNotEqual(hash(labeled_tree1), hash(labeled_tree5))

		tree_set.add(labeled_tree3)
		tree_set.add(labeled_tree4)
		tree_set.add(labeled_tree5)
		self.assertEqual(len(tree_set), 4)


class TestOrganotropismDataExtraction(unittest.TestCase):
	def test_msk_met_extraction(self):
		csv_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/msk_met/msk_met_freq_by_cancer_type.csv')
		bad_site_map = dict()
		bad_site_map_2 = {
						  "breast": "Breast",
				    	  "kidney": "Kidney",
				    	 }
		site_map = {
				    "liver": "Liver",
				    "brain": "CNS/Brain",
				    "rib": "Bone",
				    "breast": "Breast",
				    "kidney": "Kidney",
				    "lung": "Lung",
				    "adrenal": "Adrenal Gland",
				    "spinal": "CNS/Brain"
					}

		sites = ["breast", "liver", "kidney", "brain"]

		# no map provided
		with self.assertRaises(ValueError):
			data_util.get_organotropism_matrix_from_msk_met(sites, "Breast Cancer", csv_fn)

		# empty map provided
		with self.assertRaises(ValueError):
			data_util.get_organotropism_matrix_from_msk_met(sites, "Breast Cancer", csv_fn, bad_site_map)

		# incomplete map provided
		with self.assertRaises(ValueError):
			data_util.get_organotropism_matrix_from_msk_met(sites, "Breast Cancer", csv_fn, bad_site_map_2)

		# invalid primary cancer type provided
		with self.assertRaises(ValueError):
			data_util.get_organotropism_matrix_from_msk_met(sites, "Fake Cancer", csv_fn, bad_site_map)

		organo_1 = data_util.get_organotropism_matrix_from_msk_met(sites, "Breast Cancer", csv_fn, site_map)
		self.assertEqual(organo_1.shape[0], 4)
		correct_vals = torch.tensor([0.0223734135810546, 0.123642548737407, 0.00641109511971739, 0.0569148240219809], dtype = torch.float32)
		self.assertTrue((organo_1 == correct_vals).all())

# TODO test migration, comigration, seeding-site formulations

class TestSeedingPatternFromMigrationGraph(unittest.TestCase):
	def test_seeding_pattern_from_mig_graph(self):
		G = torch.tensor([[0,1,1], [0,0,0], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "monoclonal single-source seeding")
		
		G = torch.tensor([[0,0,0], [0,2,3], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "polyclonal single-source seeding")

		G = torch.tensor([[0,1,0], [0,0,1], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "monoclonal multi-source seeding")

		G = torch.tensor([[0,1,0], [0,0,2], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "polyclonal multi-source seeding")

		G = torch.tensor([[0,1,1], [1,0,0], [1,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "monoclonal reseeding")

		G = torch.tensor([[0,1,0], [2,0,0], [0,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern(G), "polyclonal reseeding")


import glob
import pandas as pd
import pyreadr
import numpy as np
import json

class TestTRACERxPreprocessing(unittest.TestCase):

	def _get_sample_to_ref_var_count(self, region_sum):
		'''
		Args:
			region_sum: from TRACERx input mutTableAll.cloneInfo.20220726.rda RegionSum 
			column, e.g. R_LN1:40/168;BR_LN2:64/276;BR_LN3:47/235;SU_FLN1:96/518

		Returns:
			dictionary mapping sample name to tuple of (ref, total) counts
		'''
		d = dict()
		region_sum_values = region_sum.split(";")
		for sample_info in region_sum_values:
			items = sample_info.split(":")
			counts = items[1].split("/")
			d[items[0]] = int(counts[0]), int(counts[1])
		return d

	def _get_sample_to_cn_count(self, cn):
		'''
		Args:
			region_sum: from TRACERx input mutTableAll.cloneInfo.20220726.rda MajorCPN_SC
			or MinoRCPN_SC column, e.g. SU_T1.R1:3;SU_T1.R2:2;SU_FLN1:2;

		Returns:
			dictionary mapping sample name to tuple of copy number counts
		'''
		d = dict()
		if isinstance(cn, float):
			return d
		cn_values = cn.split(";")
		for sample_info in cn_values:
			items = sample_info.split(":")
			cn_count = int(items[1])
			d[items[0]] = cn_count
		return d

	def _test_mig_hist_pt_orchard_tsv(self, patient_id, patient_tsv_dir, true_patient_data):
		'''
		Tests tsvs used for MACHINA and Metient input and ssm/params.json files
		used for PairTree and Orchard input against ground truth data
		'''
		# Load preprocessed data
		patient_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}_SNVs.tsv"), sep="\t")
		ssm_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}.ssm"), sep="\t")
		with open(os.path.join(patient_tsv_dir, f"{patient_id}.params.json")) as f:
			params_json = json.load(f)
		json_sample_names = params_json['samples']
		#print(params_json['samples'])
		#print(ssm_df['var_reads'])
		sample_names = patient_df['sample_label'].unique()
		mut_names = patient_df['character_label'].unique()
		anatomical_site_names = patient_df['anatomical_site_label'].unique()

		self.assertEqual(len(sample_names), len(json_sample_names))
		self.assertEqual(len(patient_df), len(sample_names)*len(mut_names))

		true_patient_data = true_patient_data[true_patient_data['patient_id']==patient_id]
		for _, row in patient_df.iterrows():
			mut = (":").join(row['character_label'].split(":")[1:])
			sample = ("_").join(row['sample_label'].split("_")[1:])

			ssm_row = ssm_df[ssm_df['name'] == row['character_label']]
			json_sample_index = json_sample_names.index(row['sample_label'])

			# e.g. R_LN1:40/168;BR_LN2:64/276;BR_LN3:47/235;SU_FLN1:96/518
			true_mut_subset = true_patient_data[(true_patient_data['mutation_id']==f"{patient_id}:{mut}")]['RegionSum'].iloc[0]
			true_sample_to_counts = self._get_sample_to_ref_var_count(true_mut_subset)
			
			# Check tsvs
			self.assertEqual(true_sample_to_counts[sample][0], row['var'])
			self.assertEqual(true_sample_to_counts[sample][1], row['var']+row['ref'])

			# Check ssm
			ssm_var_count = int(ssm_row['var_reads'].item().split(",")[json_sample_index])
			ssm_total_count = int(ssm_row['total_reads'].item().split(",")[json_sample_index])
			self.assertEqual(true_sample_to_counts[sample][0], ssm_var_count)
			self.assertEqual(true_sample_to_counts[sample][1], ssm_total_count)

	def _test_conipher_tsv(self, patient_id, conipher_tsv_dir, true_patient_data, true_sample_data, true_purity_ploidy):
		'''
		Tests tsvs used for input into CONIPHER (which does clustering via pyclone
		+ tree building). See: https://github.com/McGranahanLab/CONIPHER-wrapper
		'''
		patient_df = pd.read_csv(os.path.join(conipher_tsv_dir, f"{patient_id}_conipher_SNVs.tsv"), sep="\t")
		sample_names = patient_df['SAMPLE'].unique()
		case_id = patient_df['CASE_ID'].unique()
		self.assertEqual(case_id, patient_id)

		true_patient_data = true_patient_data[true_patient_data['patient_id']==patient_id]
		true_sample_data = true_sample_data[true_sample_data['patient_id']==patient_id]

		num_expected_rows = 0
		for _, row in true_patient_data.iterrows():
			samples_w_cn = set(self._get_sample_to_cn_count(row['MajorCPN_SC']).keys()).intersection(self._get_sample_to_cn_count(row['MinorCPN_SC']).keys())
			samples_w_cn_and_mut_data = samples_w_cn.intersection(set(self._get_sample_to_ref_var_count(row['RegionSum']).keys()))
			samples_w_sample_info = set([region.replace(f"{patient_id}_", "") for region in true_sample_data['region']])
			samples_w_full_info = samples_w_cn_and_mut_data.intersection(samples_w_sample_info)
			num_expected_rows += len(samples_w_full_info)

		
		self.assertEqual(len(patient_df), num_expected_rows)

		for _, row in patient_df.iterrows():
			mut_id = f"{patient_id}:{row['CHR']}:{row['POS']}:{row['REF']}"
			sample = row['SAMPLE']
			truncated_sample = ("_").join(sample.split("_")[1:])
			true_mut_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['RegionSum'].iloc[0]
			true_major_cn_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['MajorCPN_SC'].iloc[0]
			true_minor_cn_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['MinorCPN_SC'].iloc[0]
			true_sample_to_counts = self._get_sample_to_ref_var_count(true_mut_subset)
			true_sample_to_major_cn = self._get_sample_to_cn_count(true_major_cn_subset)
			true_sample_to_minor_cn = self._get_sample_to_cn_count(true_minor_cn_subset)

			self.assertEqual(true_sample_to_counts[truncated_sample][0], row['VAR_COUNT'])
			self.assertEqual(true_sample_to_counts[truncated_sample][1], row['DEPTH'])
			self.assertEqual(row['DEPTH'] - row['VAR_COUNT'], row['REF_COUNT'])
			self.assertEqual(true_sample_to_major_cn[truncated_sample], row['COPY_NUMBER_A'])
			self.assertEqual(true_sample_to_minor_cn[truncated_sample], row['COPY_NUMBER_B'])

			# Arghhh this indexing is not working
			# true_purity_ploidy = true_purity_ploidy[true_purity_ploidy['Sample']==sample]
			# self.assertEqual(true_purity_ploidy['final.purity'].iloc[0], np.array(row['ACF']))
			# self.assertEqual(true_purity_ploidy['final.ploidy'].iloc[0], row['PLOIDY'])


	def test_tsvs(self):
		tracerx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tracerx_nsclc")
		patient_tsv_dir = os.path.join(tracerx_dir, "patient_data")
		conipher_tsv_dir = os.path.join(tracerx_dir, "conipher_inputs")
		# Groudn truth (original data)
		true_patient_data = pyreadr.read_r(os.path.join(tracerx_dir, 'mutTableAll.cloneInfo.20220726.rda'))['mutTableAll']
		true_sample_info_df = pd.read_csv(os.path.join(tracerx_dir,"sample_overview_original.txt"), sep="\t")
		true_purity_ploidy_df = pd.read_csv(os.path.join(tracerx_dir, "20220808_purityandploidy.txt"), sep="\t")

		patients = true_patient_data['patient_id'].unique()
		print(len(patients), "patients")
		
		for patient in patients:
			self._test_mig_hist_pt_orchard_tsv(patient, patient_tsv_dir, true_patient_data)
			self._test_conipher_tsv(patient, conipher_tsv_dir, true_patient_data, true_sample_info_df, true_purity_ploidy_df)

	# TODO: test write_pooled_tsv_from_pairtree_clusters

# Test cases for transitive closure
import unittest
class TestTransitiveClosure(unittest.TestCase):
    
    def test_basic_functionality(self):
        T = torch.tensor([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        expected_result = torch.tensor([[0, 1, 1],
                                         [0, 0, 1],
                                         [0, 0, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_no_edges(self):
        T = torch.tensor([[0, 0],
                           [0, 0]])
        expected_result = torch.tensor([[0, 0],
                                         [0, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_complex_binary_tree(self):
        # Create a complex binary tree adjacency matrix
        T = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 0 to Node 1 and 2
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Node 1 to Node 3 and 4
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Node 2 to Node 5
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # Node 3 to Node 6 and 7
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 4 has no children
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Node 5 to Node 8 and 9
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 6 has no children
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 7 has no children
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 8 has no children
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)

        expected_result = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Node 0 can reach all nodes
                                        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],  # Node 1 can reach Nodes 3, 4, 6, 7
                                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # Node 2 can reach Nodes 5, 8, 9
                                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # Node 3 can reach Nodes 6 and 7
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 4 has no outgoing edges
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Node 5 can reach Nodes 8 and 9
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 6 has no outgoing edges
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 7 has no outgoing edges
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 8 has no outgoing edges
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # Node 9 has no outgoing edges
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_fully_connected(self):
        T = torch.tensor([[0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0]])
        expected_result = torch.tensor([[0, 1, 1],
                                         [0, 0, 1],
                                         [0, 0, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_disconnected_graph(self):
        T = torch.tensor([[0, 1, 0],
                           [0, 0, 0],
                           [0, 1, 0]])
        expected_result = torch.tensor([[0, 1, 0],
                                         [0, 0, 0],
                                         [0, 1, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_leaf_nodes(self):
        T = torch.tensor([[0, 1],
                           [0, 0]])
        expected_result = torch.tensor([[0, 1],
                                         [0, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

    def test_large_sparse_graph(self):
        T = torch.zeros((10, 10), dtype=torch.int)
        T[0, 1] = 1
        T[1, 2] = 1
        T[3, 4] = 1
        T[4, 5] = 1
        expected_result = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        result = vert_util.path_matrix(T)
        self.assertTrue(torch.equal(result, expected_result))

class TestMigrationHistory(unittest.TestCase):
    def test_migration_history_equality(self):
        # Create test trees and labelings
        tree1 = torch.sparse_coo_tensor(indices=torch.tensor([[0,1], 
                                                             [1,2]]).t(), 
                                      values=torch.ones(2),
                                      size=(3,3))
        tree2 = torch.zeros(3,3)
        tree2[0,1] = tree2[1,2] = 1
        
        labeling1 = torch.tensor([[1,0,0], [0,1,1]])
        labeling2 = torch.tensor([[1,0,0], [0,1,1]])
        
        # Create MigrationHistory objects
        mh1 = vert_util.MigrationHistory(tree1, labeling1)
        mh2 = vert_util.MigrationHistory(tree2, labeling2)
        
        # Test equality between sparse and dense trees
        self.assertEqual(mh1, mh2)
        
        # Test inequality with different labelings
        labeling3 = torch.tensor([[1,0,1], [0,1,0]])
        mh3 = vert_util.MigrationHistory(tree1, labeling3)
        self.assertNotEqual(mh1, mh3)
        
        # Test inequality with different trees
        tree3 = torch.sparse_coo_tensor(indices=torch.tensor([[0,2], [1,2]]).t(),
                                      values=torch.ones(2), 
                                      size=(3,3))
        mh4 = vert_util.MigrationHistory(tree3, labeling1)
        self.assertNotEqual(mh1, mh4)

class TestMigrationEdges(unittest.TestCase):
    def test_migration_edges(self):
        # Create test case
        num_sites = 3
        num_nodes = 4
        
        # Create vertex labeling matrix V where nodes 0,1 are from site 0, 
        # node 2 from site 1, and node 3 from site 2
        V = torch.zeros(num_sites, num_nodes)
        V[0,0] = 1
        V[0,1] = 1 
        V[1,2] = 1
        V[2,3] = 1

        # Test with dense adjacency matrix
        A_dense = torch.zeros(num_nodes, num_nodes)
        A_dense[0,1] = 1
        A_dense[1,2] = 1
        A_dense[1,3] = 1

        # Test without sites restriction - dense case
        Y_dense = plot_util.migration_edges(V, A_dense)
        
        # Expected: Y should have 1s at positions (1,2) and (1,3) since these
        # represent migrations between different sites
        expected = torch.zeros(num_nodes, num_nodes)
        expected[1,2] = 1
        expected[1,3] = 1
        
        self.assertTrue(torch.equal(Y_dense, expected), "Basic migration edges test failed for dense matrix")
        
        # Test with sites restriction - dense case
        sites = {1} # Only keep migrations to site 1
        Y_restricted_dense = plot_util.migration_edges(V, A_dense, sites)
        
        # Expected: Y should only have 1 at position (1,2) since that's the only
        # migration edge going to site 1
        expected_restricted = torch.zeros(num_nodes, num_nodes) 
        expected_restricted[1,2] = 1
        
        self.assertTrue(torch.equal(Y_restricted_dense, expected_restricted), "Sites restriction test failed for dense matrix")

        # Test with sparse adjacency matrix
        indices = torch.tensor([[0,1,1], [1,2,3]])
        values = torch.ones(3)
        A_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        # Test without sites restriction - sparse A case
        Y_from_sparse_A = plot_util.migration_edges(V, A_sparse)
        self.assertTrue(torch.equal(Y_from_sparse_A.to_dense(), expected), "Basic migration edges test failed for sparse A matrix")

        # Test with sites restriction - sparse A case
        Y_restricted_sparse_A = plot_util.migration_edges(V, A_sparse, sites)
        self.assertTrue(torch.equal(Y_restricted_sparse_A.to_dense(), expected_restricted), "Sites restriction test failed for sparse A matrix")

unittest.main()
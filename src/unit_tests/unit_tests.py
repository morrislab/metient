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
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "monoclonal single-source seeding")
		
		G = torch.tensor([[0,0,0], [0,2,3], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "polyclonal single-source seeding")

		G = torch.tensor([[0,1,0], [0,0,1], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "monoclonal multi-source seeding")

		G = torch.tensor([[0,1,0], [0,0,2], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "polyclonal multi-source seeding")

		G = torch.tensor([[0,1,1], [1,0,0], [1,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "monoclonal reseeding")

		G = torch.tensor([[0,1,0], [2,0,0], [0,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.get_seeding_pattern_from_migration_graph(G), "polyclonal reseeding")


import glob
import pandas as pd
import pyreadr
import json

class TestTRACERxPreprocessing(unittest.TestCase):

	def _test_patient_tsv(self, patient_id, patient_tsv_dir, true_patient_data):

		def _region_sum_to_ref_var_count(region_sum):
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

		# Load preprocessed data
		patient_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}_SNVs.tsv"), sep="\t")
		ssm_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}.ssm"), sep="\t")
		params_json = json.load(open(os.path.join(patient_tsv_dir, f"{patient_id}.params.json")))
		json_sample_names = params_json['samples']
		#print(params_json['samples'])
		#print(ssm_df['var_reads'])
		sample_names = patient_df['sample_label'].unique()
		mut_names = patient_df['character_label'].unique()
		anatomical_site_names = patient_df['anatomical_site_label'].unique()

		self.assertEqual(len(sample_names), len(json_sample_names))
		self.assertEqual(len(sample_names), len(anatomical_site_names))
		self.assertEqual(len(patient_df), len(sample_names)*len(mut_names))

		true_patient_data = true_patient_data[true_patient_data['patient_id']==patient_id]
		for _, row in patient_df.iterrows():
			mut = (":").join(row['character_label'].split(":")[1:])
			sample = ("_").join(row['sample_label'].split("_")[1:])

			ssm_row = ssm_df[ssm_df['name'] == row['character_label']]
			json_sample_index = json_sample_names.index(row['sample_label'])

			# e.g. R_LN1:40/168;BR_LN2:64/276;BR_LN3:47/235;SU_FLN1:96/518
			true_mut_subset = true_patient_data[(true_patient_data['mutation_id']==f"{patient_id}:{mut}")]['RegionSum'].iloc[0]
			true_sample_to_counts = _region_sum_to_ref_var_count(true_mut_subset)
			
			# Check tsvs
			self.assertEqual(true_sample_to_counts[sample][0], row['var'])
			self.assertEqual(true_sample_to_counts[sample][1], row['var']+row['ref'])

			# Check ssm
			ssm_var_count = int(ssm_row['var_reads'].item().split(",")[json_sample_index])
			ssm_total_count = int(ssm_row['total_reads'].item().split(",")[json_sample_index])
			self.assertEqual(true_sample_to_counts[sample][0], ssm_var_count)
			self.assertEqual(true_sample_to_counts[sample][1], ssm_total_count)


	def test_tsvs(self):
		tracerx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tracerx_nsclc")
		patient_tsv_dir = os.path.join(tracerx_dir, "patient_tsvs")
		true_patient_data = pyreadr.read_r(os.path.join(tracerx_dir, 'mutTableAll.cloneInfo.20220726.rda'))['mutTableAll']
		sample_info_df = pd.read_csv(os.path.join(tracerx_dir,"sample_overview_original.txt"), sep="\t")
		patients = true_patient_data['patient_id'].unique()
		print(len(patients), "patients")
		for patient in patients:
			print(patient)
			self._test_patient_tsv(patient, patient_tsv_dir, true_patient_data)

unittest.main()
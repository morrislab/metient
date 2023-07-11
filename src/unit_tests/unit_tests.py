import unittest
import torch

from src.util import vertex_labeling_util as vert_util
from src.util import data_extraction_util as data_util
from src.util import plotting_util as plot_util

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
		csv_fn = '../data/msk_met/msk_met_freq_by_cancer_type.csv'
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


unittest.main()
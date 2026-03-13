import unittest
import torch
import os

from metient.util import vertex_labeling_util as vert_util
from metient.util import data_extraction_util as data_util
from metient.util import plotting_util as plot_util

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


class TestSeedingPatternFromMigrationGraph(unittest.TestCase):
	def test_seeding_pattern_from_mig_graph(self):
		G = torch.tensor([[0,1,1], [0,0,0], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "monoclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "primary single-source")
		
		G = torch.tensor([[0,0,0], [0,2,3], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "polyclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "primary single-source")

		G = torch.tensor([[0,1,0], [0,0,1], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "monoclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "multi-source")

		G = torch.tensor([[0,1,0], [0,0,2], [0,0,0]])
		self.assertFalse(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "polyclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "multi-source")

		G = torch.tensor([[0,1,1], [1,0,0], [1,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "monoclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "reseeding")

		G = torch.tensor([[0,1,0], [2,0,0], [0,0,0]])
		self.assertTrue(plot_util.is_cyclic(G))
		self.assertEqual(plot_util.site_clonality_with_G(G), "polyclonal")
		self.assertEqual(plot_util.seeding_pattern_with_G(G), "reseeding")


# Test cases for transitive closure
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
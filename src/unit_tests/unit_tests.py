import unittest
import torch

from src.util import vertex_labeling_util as vert_util

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


unittest.main()
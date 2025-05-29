import unittest
import torch
import os

from metient.util import vertex_labeling_util as vert_util

class TestBasicTreeFunctions(unittest.TestCase):

    def test_root_index(self):
        T = torch.tensor([
            [0,1,1], 
            [0,0,0], 
            [0,0,0]])
        root_idx = vert_util.get_root_index(T)
        self.assertEqual(root_idx, 0)
        root_idx = vert_util.get_root_index(T.to_sparse())
        self.assertEqual(root_idx, 0)

        T = torch.tensor([
            [0,0,0], 
            [1,0,1], 
            [0,0,0]])
        root_idx = vert_util.get_root_index(T)
        self.assertEqual(root_idx, 1)
        root_idx = vert_util.get_root_index(T.to_sparse())
        self.assertEqual(root_idx, 1)
        
        T = torch.tensor([
            [0,0,0], 
            [0,0,1], 
            [0,0,0]])
        with self.assertRaises(AssertionError):
            vert_util.get_root_index(T)
        with self.assertRaises(AssertionError):
            vert_util.get_root_index(T.to_sparse())
    
    def test_get_child_indices(self):
        T = torch.tensor([
            [0,1,1], 
            [0,0,0], 
            [0,0,0]])
        child_indices = vert_util.get_child_indices(T, [0])
        self.assertEqual(child_indices, [1,2])

        child_indices = vert_util.get_child_indices(T.to_sparse(), [0])
        self.assertEqual(child_indices, [1,2])

        T = torch.tensor([
            [0,1,0], 
            [0,0,1], 
            [0,0,0]])
        child_indices = vert_util.get_child_indices(T, [0])
        self.assertEqual(child_indices, [1])
        child_indices = vert_util.get_child_indices(T.to_sparse(), [0])
        self.assertEqual(child_indices, [1])

        child_indices = vert_util.get_child_indices(T, [1])
        self.assertEqual(child_indices, [2])
        child_indices = vert_util.get_child_indices(T.to_sparse(), [1])
        self.assertEqual(child_indices, [2])
unittest.main()
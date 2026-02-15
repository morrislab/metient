import unittest
import torch
import numpy as np
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


class TestGeneticDistance(unittest.TestCase):

    def _test_inputs_1(self, sparse_A: bool):

        device = "cpu"
        
        # migration counts
        m = torch.tensor([0, 1, 1], device=device)

        V = torch.tensor(
            [   # no migrations
                [[1,1,1],
                 [0,0,0]
                ],
                # one migration edge
                [[1,1,0],
                 [0,0,1]
                ],
                # one migration edge
                [[1,0,1],
                 [0,1,0]
                ],
            ]
        )
        VT = V.transpose(1, 2)

        A_dense = torch.tensor(
            [
                [[0,1,1],
                [0,0,0],
                [0,0,0]],
                [[0,1,1],
                [0,0,0],
                [0,0,0]],
                [[0,1,1],
                [0,0,0],
                [0,0,0]],
            ]
        )

        if sparse_A:
            A = A_dense.to_sparse()
        else:
            A = A_dense

        G = torch.tensor(
            [[0,0.3,0.7],
            [0,0,0],
            [0,0,0]],
            device=device,
            requires_grad=True
        )

        return G, m, A, V, VT

    def _test_inputs_2(self, sparse_A: bool):

        device = "cpu"
        
        # migration counts
        m = torch.tensor([2, 1, 1], device=device)

        V = torch.tensor(
            [
                [[1,1,0,1],
                 [0,0,1,0]
                ],
                [[1,1,1,0],
                 [0,0,0,1]
                ],
                [[1,1,0,0],
                 [0,0,1,1]]
            ]
        )
        VT = V.transpose(1, 2)

        A_dense = torch.tensor(
            [
                [[0,1,1,0],
                [0,0,0,0],
                [0,0,0,1],
                [0,0,0,0]],
                [[0,1,1,1],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]],
                [[0,1,1,0],
                [0,0,0,0],
                [0,0,0,1],
                [0,0,0,0]],
            ]
        )

        if sparse_A:
            A = A_dense.to_sparse()
        else:
            A = A_dense

        G = torch.tensor(
            [[0,0.1,0.2,0.3],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]],
            device=device,
            requires_grad=True
        )

        return G, m, A, V, VT
    
    def test_basic_inputs(self):
        G, m, A, V, VT = self._test_inputs_1(sparse_A=False)

        g = vert_util.genetic_distance_score(G, m, A, V, VT)

        val1 = -np.log(0.7+0.01) / 2
        val2 = -np.log(0.3+0.01) / 2
        expected = torch.tensor([0.0, val1, val2], dtype=torch.float32)
        assert torch.allclose(g.detach().cpu(), expected, atol=1e-6)

        G, m, A, V, VT = self._test_inputs_1(sparse_A=True)

        g = vert_util.genetic_distance_score(G, m, A, V, VT)
        assert torch.allclose(g.detach().cpu(), expected, atol=1e-6)
    
    def test_multimigration_inputs(self):
        G, m, A, V, VT = self._test_inputs_2(sparse_A=False)

        g = vert_util.genetic_distance_score(G, m, A, V, VT)

        val1 = (-np.log(0.2+0.01)+-np.log(0.0+0.01)) / 3
        val2 = -np.log(0.3+0.01) / 2
        val3 = -np.log(0.2+0.01) / 2
        expected = torch.tensor([val1, val2, val3], dtype=torch.float32)
        print(expected)
        assert torch.allclose(g.detach().cpu(), expected, atol=1e-6)

        G, m, A, V, VT = self._test_inputs_2(sparse_A=True)

        g = vert_util.genetic_distance_score(G, m, A, V, VT)
        assert torch.allclose(g.detach().cpu(), expected, atol=1e-6)


unittest.main()
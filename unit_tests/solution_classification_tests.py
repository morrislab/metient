import unittest
import torch
import numpy as np
import metient as met
from metient.util.globals import *
from metient.util import plotting_util as plutil
from metient.util import vertex_labeling_util as vert_util

import metient
print(metient.__file__)

class TestSolutionClassification(unittest.TestCase):
    
    def test_weighted_classification(self):
        # Test case 1: Simple majority with equal weights
        losses = [0.4, 0.3, 0.3]
        classifications = ['A', 'B', 'B']
        self.assertEqual(plutil.weighted_classification(losses, classifications), 'B')
        
        # Test case 2: Strong weight for minority class
        losses = [0.6, 0.2, 0.2]
        classifications = ['A', 'B', 'B']
        self.assertEqual(plutil.weighted_classification(losses, classifications), 'A')
        
        # Test case 3: Equal weights resulting in first class
        losses = [0.5, 0.5]
        classifications = ['A', 'B']
        self.assertEqual(plutil.weighted_classification(losses, classifications), 'A')

    def _tree1(self):
        '''
        Tree:
             0
            / \
           1   2
          /     \
         3       4
        
        1,2,3,4 are same met site
        site polyclonal, genetically polyclonal
        polyphyletic
        '''
        parents = [-1,0,0,1,1]
        V = torch.tensor(
            [
                [1,0,0,0,0],
                [0,1,1,1,1]
            ]
        )
        return parents, V
    
    def _tree2(self):
        '''
        Tree:
             0
             |
             1
            / \
           2   3
        
        1,2,3 are same met site
        site monoclonal, genetically monoclonal
        monophyletic
        '''
        parents = [-1,0,1,1]
        V = torch.tensor(
            [
                [1,0,0,0],
                [0,1,1,1]
            ]
        )
        return parents, V

    def _tree3(self):
        '''
        Tree:
                0
                |
                1
               / \
              2   3

        0,1 are site 0; 2 is site 1; and 3 is site 2
        site monoclonal, genetically polyclonal
        polyphyletic
        '''
        parents = [-1,0,1,1]
        V = torch.tensor(
            [
                [1,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]
        )
        return parents, V
    
    def _tree4(self):
        '''
        Tree:
                0
                |
                1
               / \
              2   3
                   \
                    4
        0,1 are site 0; 2 is site 1; 3,4 are site 2
        site monoclonal, genetically polyclonal
        polyphyletic, primary single-source
        '''
        parents = [-1,0,1,1,3]
        V = torch.tensor(
            [
                [1,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,1,1]
            ]
        )
        return parents, V
    
    def _tree5(self):
        '''
        Tree:
                0
                |
                1
               / \
              2   3
                   \
                    4
        0 is site 0; 1,2 is site 1; 3,4 are site 2
        site monoclonal, genetically polyclonal
        monophyletic, single-source
        '''
        parents = [-1,0,1,1,3]
        V = torch.tensor(
            [
                [1,0,0,0,0],
                [0,1,1,0,0],
                [0,0,0,1,1]
            ]
        )
        return parents, V

    def _tree6(self):
        '''
        Tree:
             4
            / \
           1   2
          /     \
         3       0
        
        0,1,2,3 are same met site
        site polyclonal, genetically polyclonal
        polyphyletic
        '''
        parents = [2,4,4,1,-1]
        V = torch.tensor(
            [
                [1,0,0,0,0],
                [0,1,1,1,1]
            ]
        )
        return parents, V

    def test_weighted_classification_tree_1(self):
        parents, V = self._tree1()
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V],
            OUT_PARENTS_KEY: [parents],
            OUT_PROBABILITIES_KEY: [1.0],
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents))}]
        }
        
        # Test basic functionality
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')
    
    def test_weighted_classification_tree_2(self):
        parents, V = self._tree2()
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V],
            OUT_PARENTS_KEY: [parents],
            OUT_PROBABILITIES_KEY: [1.0],
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents))}]
        }
        
        # Test basic functionality
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'monophyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')
    
    def test_weighted_classification_both_trees(self):
        parents1, V1 = self._tree1()
        parents2, V2 = self._tree2()
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V1, V2],
            OUT_PARENTS_KEY: [parents1, parents2],
            OUT_PROBABILITIES_KEY: [0.9,0.1],
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents1))}, 
                                {x:([f'x'], False, False) for x in range(len(parents2))}]
        }
        
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')
    
    def test_weighted_classification_same_adj_diff_labels(self):
        parents2, V2 = self._tree2()
        parents3, V3 = self._tree3()
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V2, V3],
            OUT_PARENTS_KEY: [parents2, parents3],
            OUT_PROBABILITIES_KEY: [0.1,0.9], # Tree 3 is a lot better than tree 2
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents2))}, 
                                {x:([f'x'], False, False) for x in range(len(parents3))}]
        }
        
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')

        # Now do Tree 2 is a lot better than tree 3
        mock_pkl = {
            OUT_LABElING_KEY: [V2, V3],
            OUT_PARENTS_KEY: [parents2, parents3],
            OUT_PROBABILITIES_KEY: [0.9,0.1],
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents2))}, 
                                {x:([f'x'], False, False) for x in range(len(parents3))}]
        }
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'monophyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')

    
    def test_weighted_classification_both_trees3_4_5(self):
        parents3, V3 = self._tree3()
        parents4, V4 = self._tree4()
        parents5, V5 = self._tree5()
        # Tree 3 the best
        mock_pkl = {
            OUT_LABElING_KEY: [V3, V4, V5],
            OUT_PARENTS_KEY: [parents3, parents4, parents5],
            OUT_PROBABILITIES_KEY: [0.8,0.1,0.1], 
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents3))}, 
                                {x:([f'x'], False, False) for x in range(len(parents4))},
                                {x:([f'x'], False, False) for x in range(len(parents5))}]
        }
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')

        # Tree 4 the best
        mock_pkl = {
            OUT_LABElING_KEY: [V3, V4, V5],
            OUT_PARENTS_KEY: [parents3, parents4, parents5],
            OUT_PROBABILITIES_KEY: [0.1,0.8,0.1], 
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents3))}, 
                                {x:([f'x'], False, False) for x in range(len(parents4))},
                                {x:([f'x'], False, False) for x in range(len(parents5))}]
        }
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')

        # Tree 5 the best
        mock_pkl = {
            OUT_LABElING_KEY: [V3, V4, V5],
            OUT_PARENTS_KEY: [parents3, parents4, parents5],
            OUT_PROBABILITIES_KEY: [0.1,0.1,0.8], 
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents3))}, 
                                {x:([f'x'], False, False) for x in range(len(parents4))},
                                {x:([f'x'], False, False) for x in range(len(parents5))}]
        }
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'monophyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'single-source')
    
    def _tree7_mixed_site_clonality(self):
        '''
        Two metastatic sites:
        site1 = monoclonal
        site2 = polyclonal

        Global site clonality should be polyclonal.
        '''
        parents = [-1,0,1,1,1]

        V = torch.tensor(
            [
                [1,1,0,0,0],  # primary
                [0,0,1,0,0],  # site A (monoclonal)
                [0,0,0,1,1],  # site B (polyclonal)
            ]
        )
        return parents, V

    def test_mixed_site_clonality(self):
        parents, V = self._tree7_mixed_site_clonality()

        node_info = [{x:([f'x'], False, False) for x in range(len(parents))}]
        mock_pkl = {
            OUT_LABElING_KEY: [V],
            OUT_PARENTS_KEY: [parents],
            OUT_PROBABILITIES_KEY: [1.0],
            OUT_IDX_LABEL_KEY: node_info
        }

        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'polyclonal')
        A = met.adjacency_matrix_from_parents(parents)
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0]), [2,3,4])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[1]), [2])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[2]), [3,4])

        node_info[0][3] = (['x'], True, False) # make node 3 a polytomy resolver node
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0]), [1,2,4])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[1]), [2])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[2]), [1,4])

        A = A.to_dense()
        node_info = [{x:([f'x'], False, False) for x in range(len(parents))}]
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0]), [2,3,4])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[1]), [2])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[2]), [3,4])

        node_info[0][3] = (['x'], True, False) # make node 3 a polytomy resolver node
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0]), [1,2,4])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[1]), [2])
        self.assertEqual(plutil.seeding_clusters(V, A, node_info[0], sites=[2]), [1,4])

    def test_phyleticity(self):
        parents = [ 2,  2, -1,  1,  0]
        V = torch.tensor(
            [[0, 0, 1, 0, 0],
             [1, 1, 0, 1, 1]]
        )
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V],
            OUT_PARENTS_KEY: [parents],
            OUT_PROBABILITIES_KEY: [1.0],
            OUT_IDX_LABEL_KEY: [{0: (['0'], False, False), 4: (['0', 'LN'], True, False), 1: (['1'], False, False), 3: (['1', 'LN'], True, False), 2: (['2'], False, False)}]
        }
        

        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'polyclonal')
    
    def test_seeding_pattern_from_mig_graph(self):
        G = torch.tensor([[0,1,1], [0,0,0], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "monoclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "primary single-source")
        
        G = torch.tensor([[0,0,0], [2,0,3], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "polyclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "primary single-source")

        G = torch.tensor([[0,1,0], [0,0,1], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "monoclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "single-source")

        G = torch.tensor([[0,1,0], [0,0,2], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "polyclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "single-source")

        G = torch.tensor([[0,1,1], [0,0,1], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "polyclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "multi-source")

        G = torch.tensor([[0,1,1], [1,0,0], [1,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "polyclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "reseeding")

        G = torch.tensor([[0,1,0], [2,0,0], [0,0,0]])
        self.assertEqual(plutil.site_clonality_with_G(G), "polyclonal")
        self.assertEqual(plutil.seeding_pattern_with_G(G), "reseeding")

class TestTransitiveClosure(unittest.TestCase):
    
    def test_basic_functionality(self):
        T = torch.tensor([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        expected_result = torch.tensor([[0, 1, 1],
                                         [0, 0, 1],
                                         [0, 0, 0]])
        result = vert_util.path_matrix(T, remove_self_loops=True)
        self.assertTrue(torch.equal(result, expected_result))

    def test_no_edges(self):
        T = torch.tensor([[0, 0],
                           [0, 0]])
        expected_result = torch.tensor([[0, 0],
                                         [0, 0]])
        result = vert_util.path_matrix(T, remove_self_loops=True)
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
        result = vert_util.path_matrix(T, remove_self_loops=True)
        self.assertTrue(torch.equal(result, expected_result))

    def test_fully_connected(self):
        T = torch.tensor([[0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0]])
        expected_result = torch.tensor([[0, 1, 1],
                                         [0, 0, 1],
                                         [0, 0, 0]])
        result = vert_util.path_matrix(T, remove_self_loops=True)
        self.assertTrue(torch.equal(result, expected_result))

    def test_disconnected_graph(self):
        T = torch.tensor([[0, 1, 0],
                           [0, 0, 0],
                           [0, 1, 0]])
        expected_result = torch.tensor([[0, 1, 0],
                                         [0, 0, 0],
                                         [0, 1, 0]])
        result = vert_util.path_matrix(T, remove_self_loops=True)
        self.assertTrue(torch.equal(result, expected_result))

    def test_leaf_nodes(self):
        T = torch.tensor([[0, 1],
                           [0, 0]])
        expected_result = torch.tensor([[0, 1],
                                         [0, 0]])
        result = vert_util.path_matrix(T, remove_self_loops=True)
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
        result = vert_util.path_matrix(T, remove_self_loops=True)
        self.assertTrue(torch.equal(result, expected_result))

if __name__ == '__main__':
    unittest.main()

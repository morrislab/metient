import unittest
import torch
import numpy as np
import metient as met
from metient.util.globals import *
from metient.util import plotting_util as plutil

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

    def test_get_soln_probabilities(self):
        # Create mock loss dictionaries
        loss_dict1 = {
            MIG_KEY: 1.0,
            COMIG_KEY: 1.0,
            SEEDING_KEY: 1.0,
            GEN_DIST_KEY: 1.0,
            ORGANOTROP_KEY: 1.0,
            ENTROPY_KEY: 1.0,
            FULL_LOSS_KEY: 1.0
        }
        loss_dict2 = {
            MIG_KEY: 2.0,
            COMIG_KEY: 2.0,
            SEEDING_KEY: 2.0,
            GEN_DIST_KEY: 2.0,
            ORGANOTROP_KEY: 2.0,
            ENTROPY_KEY: 2.0,
            FULL_LOSS_KEY: 2.0
        }
        
        # Test case 1
        probs = plutil.get_soln_probabilities([loss_dict1, loss_dict2])
        self.assertTrue(isinstance(probs, np.ndarray))
        self.assertEqual(len(probs), 2)
        self.assertGreater(probs[0], probs[1])  # First solution should have higher probability


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

        0,1 are same site, 2 and 3 are diff site
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

    def test_weighted_classification_tree_1(self):
        parents, V = self._tree1()
        # Create mock pickle data
        mock_pkl = {
            OUT_LABElING_KEY: [V],
            OUT_ADJ_KEY: [parents],
            OUT_LOSS_DICT_KEY: [
                {FULL_LOSS_KEY: 1.0},
            ],
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
            OUT_ADJ_KEY: [parents],
            OUT_LOSS_DICT_KEY: [
                {FULL_LOSS_KEY: 1.0},
            ],
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
            OUT_ADJ_KEY: [parents1, parents2],
            OUT_LOSS_DICT_KEY: [
                {FULL_LOSS_KEY: 1.0}, # Tree 1 is a lot better than tree 2
                {FULL_LOSS_KEY: 10.0},
            ],
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
            OUT_ADJ_KEY: [parents2, parents3],
            OUT_LOSS_DICT_KEY: [
                {FULL_LOSS_KEY: 10.0}, 
                {FULL_LOSS_KEY: 1.0}, # Tree 3 is a lot better than tree 2
            ],
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
            OUT_ADJ_KEY: [parents2, parents3],
            OUT_LOSS_DICT_KEY: [
                {FULL_LOSS_KEY: 1.0}, 
                {FULL_LOSS_KEY: 10.0}, # Tree 3 is a lot better than tree 2
            ],
            OUT_IDX_LABEL_KEY: [{x:([f'x'], False, False) for x in range(len(parents2))}, 
                                {x:([f'x'], False, False) for x in range(len(parents3))}]
        }
        self.assertEqual(met.weighted_phyleticity(mock_pkl), 'polyphyletic')
        self.assertEqual(met.weighted_genetic_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_site_clonality(mock_pkl), 'monoclonal')
        self.assertEqual(met.weighted_seeding_pattern(mock_pkl), 'primary single-source')

    

if __name__ == '__main__':
    unittest.main()

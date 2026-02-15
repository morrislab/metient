import unittest
import torch
from metient.lib import polytomy_resolver as pr
from metient.util import vertex_labeling_util as vutil

class TestRemoveNodes(unittest.TestCase):
    
    def input_1(self):
        '''
            A - B - C
        '''
        self.V = torch.tensor([[0, 0, 1], 
                               [0, 1, 0], 
                               [1, 0, 0]])
        self.T = torch.tensor([[0, 1, 0], 
                               [0, 0, 1], 
                               [0, 0, 0]]).to_sparse()
        self.G = torch.tensor([[0, 0.1, 0.5], 
                               [0, 0, 0.5], 
                               [0, 0, 0]])
        self.removal_indices = [1]  # Index to remove
        nodes = [vutil.MigrationHistoryNode(0,["A"],False,False),
                 vutil.MigrationHistoryNode(1,["B"],False,True),
                 vutil.MigrationHistoryNode(2,["C"],True,False)]
        self.node_collection = vutil.MigrationHistoryNodeCollection(nodes)

        # Expected results
        self.expected_V = torch.tensor([[0, 1], 
                                   [0, 0], 
                                   [1, 0]])
        self.expected_T = torch.tensor([[0, 1], 
                                   [0, 0]])
        self.expected_G = torch.tensor([[0, 0.5], 
                                   [0, 0]])
        expected_nodes = [vutil.MigrationHistoryNode(0,["A"],False,False),
                          vutil.MigrationHistoryNode(1,["C"],True,False)]
        self.expected_node_collection = vutil.MigrationHistoryNodeCollection(expected_nodes)
    
    def input_2(self):
        '''
            A
           / \
           B  C
          / \  \
          D  E  F
        '''
        self.V = torch.tensor([[0, 0, 1, 0, 0, 1], 
                               [0, 1, 0, 1, 0, 0], 
                               [1, 0, 0, 0, 1, 0]])
        self.T = torch.tensor([[0, 1, 1, 0, 0, 0], 
                               [0, 0, 0, 1, 1, 0], 
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0], ]).to_sparse()
        print('self T shape', self.T.size())
        self.G = torch.tensor([[0, 0.1, 0.1, 0.2, 0.3, 0.4], 
                               [0, 0, 0, 0.2, 0.3, 0], 
                               [0, 0, 0, 0, 0, 0.4],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],])
        self.removal_indices = [1, 2]  # Index to remove
        nodes = [vutil.MigrationHistoryNode(0,["A"],False,False),
                 vutil.MigrationHistoryNode(1,["B"],False,True),
                 vutil.MigrationHistoryNode(2,["C"],False,True),
                 vutil.MigrationHistoryNode(3,["D"],True,False),
                 vutil.MigrationHistoryNode(4,["E"],True,False),
                 vutil.MigrationHistoryNode(5,["F"],True,False)]
        self.node_collection = vutil.MigrationHistoryNodeCollection(nodes)

        # Expected results
        self.expected_V = torch.tensor([[0, 0, 0, 1], 
                                        [0, 1, 0, 0], 
                                        [1, 0, 1, 0]])
        self.expected_T = torch.tensor([[0, 1, 1, 1], 
                                        [0, 0, 0, 0], 
                                        [0, 0, 0, 0], 
                                        [0, 0, 0, 0], ]).to_sparse()
        self.expected_G = torch.tensor([[0, 0.2, 0.3, 0.4], 
                                        [0, 0, 0, 0], 
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],])
        expected_nodes = [vutil.MigrationHistoryNode(0,["A"],False,False),
                          vutil.MigrationHistoryNode(1,["D"],True,False),
                          vutil.MigrationHistoryNode(2,["E"],True,False),
                          vutil.MigrationHistoryNode(3,["F"],True,False)]
        self.expected_node_collection = vutil.MigrationHistoryNodeCollection(expected_nodes)
    
    def check_results(self):
        # Call the function
        V, T, G, node_collection = pr.remove_nodes(self.removal_indices, self.V, self.T, self.G, self.node_collection)
        # Assertions
        self.assertTrue(torch.equal(V, self.expected_V), "V matrix is incorrect after removal")
        self.assertTrue(torch.equal(T.to_dense(), self.expected_T.to_dense()), "T matrix is incorrect after removal")
        self.assertTrue(torch.equal(G, self.expected_G), "G matrix is incorrect after removal")
        self.assertTrue(len(node_collection.idx_to_mig_hist_node)==T.size()[0])
        self.assertTrue(T.size()[0]==T.size()[1])
        self.assertTrue(G.size()[0]==G.size()[1])
        self.assertTrue(T.size()[0]==V.shape[1])
        self.assertTrue(V.shape[1]==G.shape[0])
        self.assertTrue(len(node_collection.idx_to_mig_hist_node)==len(self.expected_node_collection.idx_to_mig_hist_node))
        for idx in node_collection.idx_to_mig_hist_node:
            self.assertTrue(idx in self.expected_node_collection.idx_to_mig_hist_node)
            node = node_collection.idx_to_mig_hist_node[idx]
            expected_node = self.expected_node_collection.idx_to_mig_hist_node[idx]
            self.assertTrue(node.label == expected_node.label)
            self.assertTrue(node.is_witness == expected_node.is_witness)
            self.assertTrue(node.is_polytomy_resolver_node == expected_node.is_polytomy_resolver_node)


    def test_remove_nodes_1(self):
        self.input_1()
        self.check_results()        

    def test_remove_nodes_2(self):
        self.input_2()
        self.check_results()  

    def test_child_indices(self):
        

        T = torch.tensor([[0, 1, 1, 0, 0, 0], 
                          [0, 0, 0, 1, 1, 0], 
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0], ]).to_sparse()
        child_indices = vutil.get_child_indices_sparse_t(T, 0)
        self.assertEqual(child_indices, [1,2])

        child_indices = vutil.get_child_indices_sparse_t(T, 1)
        self.assertEqual(child_indices, [3,4])

        child_indices = vutil.get_child_indices_sparse_t(T, 4)
        self.assertEqual(child_indices, [])

if __name__ == "__main__":
    unittest.main()

import unittest
import torch
import metient as met
from metient.util import vertex_labeling_util as vutil
import os
import gzip
import pickle

class TestAncestralLabelingMetrics(unittest.TestCase):
    
    def input_1(self, sparse_T=False):
        '''
            A - B - C
        '''
        self.V = torch.tensor([[[0, 0, 1], 
                               [0, 1, 0], 
                               [1, 0, 0]]])

        self.T = torch.tensor([[[0, 1, 0], 
                               [0, 0, 1], 
                               [0, 0, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()
        
    def input_2(self, sparse_T=False):
        '''
            A
           / \
           B  C
          / \  \
          D  E  F
        '''
        self.V = torch.tensor([[[0, 0, 1, 0, 0, 1], 
                               [0, 1, 0, 1, 0, 0], 
                               [1, 0, 0, 0, 1, 0]]])
        self.T = torch.tensor([[[0, 1, 1, 0, 0, 0], 
                               [0, 0, 0, 1, 1, 0], 
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0],]])
        if sparse_T:
            self.T = self.T.to_sparse()
    
    def input_3(self, sparse_T=False):
        '''
            A
           / \
           B  C
          / \  \
          D  E  F
        '''
        self.V = torch.tensor([[[0, 0, 1, 1, 1, 1], 
                               [0, 1, 0, 0, 0, 0], 
                               [1, 0, 0, 0, 0, 0]]])
        self.T = torch.tensor([[[0, 1, 1, 0, 0, 0], 
                               [0, 0, 0, 1, 1, 0], 
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0],]])
        if sparse_T:
            self.T = self.T.to_sparse()

    def input_no_seeding(self, sparse_T=False):
        '''
            A
           / \
           B  C
          / \  \
          D  E  F
        '''
        self.V = torch.tensor([[[0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0], 
                               [1, 1, 1, 1, 1, 1]]])
        self.T = torch.tensor([[[0, 1, 1, 0, 0, 0], 
                               [0, 0, 0, 1, 1, 0], 
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0],]])
        if sparse_T:
            self.T = self.T.to_sparse()
    
    def input_reseeding(self, sparse_T=False):
        '''
            A - B - C - D
        '''
        self.V = torch.tensor([[[1, 0, 1, 0], 
                               [0, 1, 0, 1]]])

        self.T = torch.tensor([[[0, 1, 0, 0], 
                               [0, 0, 1, 0], 
                               [0, 0, 0, 1],
                               [0, 0, 0, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()
    
    def input_reseeding2(self, sparse_T=False):
        '''
              A
             / 
            B  
           / \
          C   F
         / \   \
        D   E   G
        '''
        self.V = torch.tensor([[[1, 0, 0, 1, 1, 1, 1], 
                                [0, 1, 1, 0, 0, 0, 0]]])

        self.T = torch.tensor([[[0, 1, 0, 0, 0, 0, 0], 
                                [0, 0, 1, 0, 0, 1, 0], 
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0],]])
        if sparse_T:
            self.T = self.T.to_sparse()
        

    def input_reseeding3(self, sparse_T=False):
        '''
              A
             / 
            B  
           /
          C
         /|\
        D E F
        '''
        self.V = torch.tensor([[[1, 0, 1, 0, 0, 0], 
                                [0, 1, 0, 1, 1, 1]]])

        self.T = torch.tensor([[[0, 1, 0, 0, 0, 0], 
                                [0, 0, 1, 0, 0, 0], 
                                [0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()
        
    def input_reseeding4(self, sparse_T=False):
        '''
            A  
            |  
            B  
           / \
          C   D
         /   / \
        E   F   G
        '''
        self.V = torch.tensor([[[1, 0, 0, 0, 1, 1, 1], 
                                [0, 1, 1, 1, 0, 0, 0]]])

        self.T = torch.tensor([[[0, 1, 0, 0, 0, 0, 0], 
                                [0, 0, 1, 1, 0, 0, 0], 
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()
    
    def input_reseeding5(self, sparse_T=False):
        '''
        0 - 1 - 2 - 3 - 4 
        '''
        self.T = torch.tensor([[[0, 1, 0, 0, 0], 
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0]]])

        self.V = torch.tensor([[[1, 0, 1, 0, 1], 
                               [0, 1, 0, 1, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()
    
    def input_reseeding6(self, sparse_T=False):
        '''
                A  
                |  
                B  
               /
              C 
             / \ 
            D   F   
           / \
          E   G
        '''
        self.T = torch.tensor([[[0, 1, 0, 0, 0, 0, 0], 
                               [0, 0, 1, 0, 0, 0, 0], 
                               [0, 0, 0, 1, 0, 1, 0],
                               [0, 0, 0, 0, 1, 0, 1],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]]])

        self.V = torch.tensor([[[1, 0, 1, 0, 1, 0, 1], 
                                [0, 1, 0, 1, 0, 1, 0]]])
        if sparse_T:
            self.T = self.T.to_sparse()

    def assert_metrics(self,out,true_m,true_c,true_s,true_g,true_o):
        self.assertEqual(int(out[0]),true_m)
        self.assertEqual(int(out[1]),true_c)
        self.assertEqual(int(out[2]),true_s)
        self.assertEqual(float(out[3]),true_g)
        self.assertEqual(float(out[4]),true_o)

    def test_basic(self):
        self.input_1(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 2, 2, 2, 0, 0)

        self.input_1(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 2, 2, 2, 0, 0)

        self.input_2(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 3, 3, 2, 0, 0)
        
        self.input_2(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 3, 3, 2, 0, 0)

        self.input_3(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 3, 2, 0, 0)
        
        self.input_3(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 3, 2, 0, 0)

        self.input_no_seeding(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 0, 0, 0, 0, 0)
        
        self.input_no_seeding(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 0, 0, 0, 0, 0)

    def test_reseeding(self):
        self.input_reseeding(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 3, 3, 2, 0, 0)
        
        self.input_reseeding(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 3, 3, 2, 0, 0)

        self.input_reseeding2(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 2, 2, 0, 0)
        
        self.input_reseeding2(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 2, 2, 0, 0)

        self.input_reseeding3(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 5, 3, 2, 0, 0)
        
        self.input_reseeding3(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 5, 3, 2, 0, 0)

        self.input_reseeding4(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 2, 2, 0, 0)
        
        self.input_reseeding4(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 2, 2, 0, 0)

        self.input_reseeding5(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 4, 2, 0, 0)
        
        self.input_reseeding5(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 4, 4, 2, 0, 0)

        self.input_reseeding6(sparse_T=False)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 6, 4, 2, 0, 0)
        
        self.input_reseeding6(sparse_T=True)
        out = vutil.ancestral_labeling_metrics(self.V, self.T, None, None, None, True, True, True)
        self.assert_metrics(out, 6, 4, 2, 0, 0)
    
    def test_lt_clone_43(self):
        data_dir = "/data1/morrisq/divyak/projects/metient/metient/unit_tests/data"
        with gzip.open(os.path.join(data_dir, "43_LL.pkl.gz") ,"rb") as f:
            pckl = pickle.load(f)
        parents = pckl['full_adjacency_matrices'][0]
        A = met.adjacency_matrix_from_parents(parents)
        A = A.to_dense()
        V = torch.tensor(pckl['clone_tree_labeling_matrices'][0]).unsqueeze(0)
        out = vutil.ancestral_labeling_metrics(V, A, None, None, None, True, True, True)
        self.assert_metrics(out, 27, 7, 3, 0, 0)

        A = A.to_sparse()
        out = vutil.ancestral_labeling_metrics(V, A, None, None, None, True, True, True)
        self.assert_metrics(out, 27, 7, 3, 0, 0)

    def test_lt_clone_15(self):
        data_dir = "/data1/morrisq/divyak/projects/metient/metient/unit_tests/data"
        with gzip.open(os.path.join(data_dir, "15_LL.pkl.gz") ,"rb") as f:
            pckl = pickle.load(f)
        parents = pckl['full_adjacency_matrices'][0]
        A = met.adjacency_matrix_from_parents(parents)
        A = A.to_dense()
        V = torch.tensor(pckl['clone_tree_labeling_matrices'][0]).unsqueeze(0)
        out = vutil.ancestral_labeling_metrics(V, A, None, None, None, True, True, True)
        self.assert_metrics(out, 124, 14, 3, 0, 0)

        A = A.to_sparse()
        out = vutil.ancestral_labeling_metrics(V, A, None, None, None, True, True, True)
        self.assert_metrics(out, 124, 14, 3, 0, 0)

if __name__ == "__main__":
    unittest.main()

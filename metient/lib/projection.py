###########################################################################
# projection.py
#
# Contains all of the necessary functions to call the projection algorithm
# from the projectppm library. This is a condensed version of the
# cellular prevalence fitting source from Pairtree that has added documentation.
###########################################################################

import sys, os
import numpy as np
from scipy.stats import binom
import numpy.ctypeslib as npct
import ctypes
from omicsdata.tree.parents import parents_to_adj
from omicsdata.tree.adj import adj_to_anc

from metient.util.globals import MIN_VARIANCE

def _adj_to_adjppm(adj):
    """Convert an adjacency matrix to a format projectppm is expecting
    
    Parameters
    ----------
    adj : ndarray
        an adjaceny matrix for the partial tree

    Returns
    -------
    ndarray 
        an adjacency list used by the project algorithm
    ndarray
        an array containing the degree of each vertex in the tree
    """
    assert np.all(np.diag(adj) == 1) # ensure we have a valid format for an adjacency matrix
    adj = np.copy(adj)
    np.fill_diagonal(adj, 0)
    adjlist = [np.flatnonzero(row) for row in adj + adj.T]
    degrees = np.array([len(row) for row in adjlist], dtype=np.short)
    adjppm = np.zeros_like(adj, dtype=np.short)
    for i, row in enumerate(adjlist):
        adjppm[i, :len(row)] = row
    return adjppm, degrees

def fit_F(parents, V, R, omega, W):
    """Uses the projection algorithm to compute the F matrix
    
    Parameters
    ----------
    parents : ndarray
        a numpy array where each index represents a node, and the value at that index represents
        that nodes direct ancestor (parent) in the tree
    V : ndarray
        a 2D array of variant read counts across all samples for each node. Each row i, V_i, are the 
        variant read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read counts for node i in sample s.
    R : ndarray
        a 2D array of reference read counts across all samples for each node. Each row i, V_i, are the 
        reference read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the reference read counts for node i in sample s.
    omega : ndarray
        a 2D array of variant read probabilities across all samples for each node. Each row i, V_i, are the 
        variant read probabilities for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read probabilities for node i in sample s.
    W : ndarray
        a 2D array of weights used to account for variance in sequencing coverage 
    Returns
    -------
    ndarray
        the cellular prevalence matrix F fit to the (partial) tree
    ndarray
        the subpopulation frequency matrix eta fit to the (partial) tree
    float
        the log binomial likelihood of the cellular prevalence matrix given the read count data
    """
    F, eta = _fit_F(parents, V, R, omega, W)
    F_llh, _, _ = calc_llh(F, V, V + R, omega)
    return F, eta, np.sum(F_llh)

def calc_llh(F, V, N, omega_v, epsilon=1e-5):
    """Computes the log-likelihood under a binomial likelihood model for the cellular prevalence
    matrix F given the read count data
    
    Parameters
    ----------
    F : ndarray
        the mutation frequency matrix F fit to the (partial) tree
    V : ndarray
        a 2D array of variant read counts across all samples for each node. Each row i, V_i, are the 
        variant read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read counts for node i in sample s.
    R : ndarray
        a 2D array of reference read counts across all samples for each node. Each row i, V_i, are the 
        reference read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the reference read counts for node i in sample s.
    omega : ndarray
        a 2D array of variant read probabilities across all samples for each node. Each row i, V_i, are the 
        variant read probabilities for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read probabilities for node i in sample s.
    epsilon : float
        the smallest VAF that a node i can have in a sample s

    Returns
    -------
    ndarray
        a matrix where each entry (i,s) is the log binomial likelihood of the cellluar prevalence of node i in sample s
    ndarray
        an array where each entry (s) is the sum of log binomial likelihoods of the cellluar prevalence all nodes sample s
    float
        the negative log-likelihood of the cellular prevalence matrix
    """
    K, S = F.shape
    for arr in V, N, omega_v:
        assert arr.shape == (K-1, S), "F, V, or N do not have that same input shape (mutation x sample)"

    assert np.allclose(1, F[0]), "The clonal population cellular prevalence is not equal to 1."
    P = omega_v * F[1:]
    P = np.maximum(P, epsilon)
    P = np.minimum(P, 1 - epsilon)

    F_llh = binom.logpmf(V, N, P) 
    assert not np.any(np.isnan(F_llh)) and not np.any(np.isinf(F_llh)), \
           "F log-likelihood is incorrect. Please check inputs to calc_llh."

    llh_per_sample = -np.sum(F_llh, axis=0) / K
    nlglh = np.sum(llh_per_sample) / S
    return (F_llh, llh_per_sample, nlglh)

def _fit_F(parents, V, R, omega, W):
    """Fits the cellular prevalence matrix F matrix one sample at a time
    
    Parameters
    ----------
    parents : ndarray
        a numpy array where each index represents a node, and the value at that index represents
        that nodes direct ancestor (parent) in the tree
    V : ndarray
        a 2D array of variant read counts across all samples for each node. Each row i, V_i, are the 
        variant read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read counts for node i in sample s.
    R : ndarray
        a 2D array of reference read counts across all samples for each node. Each row i, V_i, are the 
        reference read counts for the node at index i in the parents array, and each column
        s, V_{is}, are the reference read counts for node i in sample s.
    omega : ndarray
        a 2D array of variant read probabilities across all samples for each node. Each row i, V_i, are the 
        variant read probabilities for the node at index i in the parents array, and each column
        s, V_{is}, are the variant read probabilities for node i in sample s.
    W : ndarray
        a 2D array of weights used to account for variance in sequencing coverage 
        
    Returns
    -------
    ndarray
        the cellular prevalence matrix F fit to the (partial) tree
    ndarray
        the subpopulation frequency matrix eta fit to the (partial) tree
    """
    # compute total reads (T)
    T = R + V
    M, S = T.shape

    # make an ancestry matrix and adjacency matrix from the parents vector
    adj = parents_to_adj(parents)
    anc = adj_to_anc(adj)
    
    # prevent dividing by zero
    F_hat = np.divide(V, omega * T, out=np.zeros_like(V).astype(np.float64), where=(omega * T)!=0)
    F_hat = np.maximum(0, F_hat)
    F_hat = np.minimum(1, F_hat)
    F_hat = np.insert(F_hat, 0, 1, axis=0)

    V_hat = V + 1
    T_hat = V + R + 2
    W = V_hat*(1 - V_hat/T_hat) / (T_hat*omega)**2
    W = np.maximum(MIN_VARIANCE, W)
    W = np.insert(W, 0, MIN_VARIANCE, axis=0)

    # initialize eta 
    eta = np.zeros((M+1, S))

    # fit eta for each sample
    for sidx in range(S):
        eta[:,sidx] = _fit_eta_S(adj, F_hat[:,sidx], W[:,sidx])
    
    # perform checks
    assert not np.any(np.isnan(eta)), "eta contains np.nans"
    assert np.allclose(0, eta[eta < 0])
    eta[eta < 0] = 0
    assert np.allclose(1, np.sum(eta, axis=0))

    # compute F
    F = np.dot(anc, eta)
    return (F, eta)

def _fit_eta_S(adj, F_hat, var_F_hat, max_retries=3):
    """Fits the eta values for a particular sample using the projection algorithm
    
    Parameters
    ----------
    adj : ndarray
        the adjacency matrix of the (partial) tree
    F_hat : ndarray
        an array of data-implied cellular prevalences for the nodes in the (partial) tree in sample s
    var_F_hat : ndarray
        an array of cellular prevalence variances for each node in sample s
    max_retries : int
        the number of retries to perform if the projection algorithm doesn't provide a valid output

    Returns
    -------
    ndarray
        the array of subpopulation frequency eta for sample s
    """
    M = len(F_hat)
    assert F_hat.ndim == var_F_hat.ndim == 1 and var_F_hat.shape == (M,) and M >=1, \
           "Invalid input"

    # There is an issue with projectppm for certain inputs, see Pairtree source for more information
    for _ in range(max_retries):
        eta = _project_ppm(adj, F_hat, var_F_hat)
        if not np.any(np.isnan(eta)):
            return eta
        print("Projectppm failed to return a valid eta, retrying ...")
    raise Exception('eta still contains NaN after %s attempt(s)' % max_retries)

def _project_ppm(adj, F_hat, var_F_hat, root=0):
    """Wrapper around the C implementation of the projection algorithm. Runs the projection algorithm
    on the data for a single sample (which is how the algorithm is designed).
    
    Parameters
    ----------
    adj : ndarray
        the adjacency matrix of the (partial) tree
    F_hat : ndarray
        an array of data-implied cellular prevalences for the nodes in the (partial) tree in sample s
    var_F_hat : ndarray
        an array of cellular prevalence variances for each node in sample s
    root : int
        the index that designates the root of the tree in the adjacency matrix

    Returns
    -------
    ndarray
        the array of subpopulation frequency eta for sample s
    """
    assert F_hat.ndim == var_F_hat.ndim == 1
    eta = np.empty(len(F_hat), dtype=np.double) # this will be filled out by the call to projectppm

    # see pairtree source for naming conventions
    gamma_init = var_F_hat
    F_hat = F_hat / gamma_init

    # convert the adjacency matrix to a format that projectppm is expecting
    adjppm, degrees = _adj_to_adjppm(adj)

    # make all variables passed into projectppm be C-continguous arrays  
    # see: https://numpy.org/doc/stable/reference/generated/numpy.require.html
    eta = np.require(eta, requirements='C')
    F_hat = np.require(F_hat, requirements='C')
    gamma_init = np.require(gamma_init, requirements='C')
    degrees = np.require(degrees, requirements='C')
    adjppm = np.require(adjppm, requirements='C')

    flag = 0
    compute_eta = 1
    S = 1
    l = len(F_hat)
    # run projectppm projection algorithm, see pairtree source for more documentation on call
    _project_ppm.tree_cost_projection(
        flag, # flag
        compute_eta, # tells to compute eta 
        eta,
        l, # length of input data
        S, # number of samples
        F_hat,
        gamma_init,
        root,
        None,
        None,
        degrees,
        adjppm
    )

    return eta

def _init_projectppm_lib():
    """Initializes all the variables necessary to invoke the projection algorithm from 
    the projectppm library."""

    # define argument types that the projection C function is expecting
    real_arr_1d = npct.ndpointer(dtype=np.float64, ndim=1, flags='C')
    short_arr_1d = npct.ndpointer(dtype=ctypes.c_short, ndim=1, flags='C')
    short_arr_2d = npct.ndpointer(dtype=ctypes.c_short, ndim=2, flags='C')
    class Edge(ctypes.Structure):
        _fields_ = [('first', ctypes.c_short), ('second', ctypes.c_short)]
    c_edge_p = ctypes.POINTER(Edge)
    c_short_p = ctypes.POINTER(ctypes.c_short)
    print(os.path.join(os.path.dirname(__file__)))
    lib_path = os.path.join(os.path.dirname(__file__), 'projectppm', 'bin', 'libprojectppm.so')
    assert os.path.exists(lib_path), 'Could not find projectppm library. \
                                      Please ensure the libprojectppm repository has been cloned in the lib directory,\
                                      and that its source has been built. '

    # loads the projectppm library
    lib = ctypes.cdll.LoadLibrary(lib_path)

    # makes it so ctypes will ensure the argument types 
    # are what the projection function is expecting
    func = lib.tree_cost_projection
    func.argtypes = [
        ctypes.c_short,
        ctypes.c_short,
        real_arr_1d,
        ctypes.c_short,
        ctypes.c_short,
        real_arr_1d,
        real_arr_1d,
        ctypes.c_short,
        c_edge_p,
        c_short_p,
        short_arr_1d,
        short_arr_2d,
    ]
    func.restype = ctypes.c_double
    _project_ppm.tree_cost_projection = func

# Call the init function in here such that each time this file is imported everything we need to 
# use the projection algorithm is already prepared
_init_projectppm_lib() 

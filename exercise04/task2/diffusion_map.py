import numpy as np
import scipy.linalg as la
import scipy.spatial as spatial
import scipy.sparse as sp
import time

def diffusion_map(data_set, L, printTimes=False, printOutput=False):
    """implements the diffusion map algorithm using the helper functions within this file to compute the L largest eigenvalues and eigenfunctions

    :param data_set: the data set for which the eigenfunctions are of interest
    :type data_set: numpy.array
    :param L: the number of the largest eigenvalues of interest
    :type L: int
    :return: a tuple of the L largest eigenvalues lambda and eigenfunctions phi
    :rtype: (numpy.array, numpy.array)
    """
    start = time.time()
    #1. distance matrix D
    D = distance_map(data_set, data_set)
    if printTimes:
        end = time.time()
        print(f"1. Distance Matrix D:\t\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(D)
    #2. Epsilon e
    e = epsilon(D)
    if printTimes:
        end = time.time()
        print(f"2. Diameter of dataset e:\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(e)
    #3. Kernel matrix W
    W = kernel(e, D)
    if printTimes:
        end = time.time()
        print(f"3. Kernel matrix W:\t\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(W)
    #4. Inverse of diagonal normalization matrix P
    P = diag_norm_inv(W)
    if printTimes:
        end = time.time()
        print(f"4. Diagonal normalization matrix P:\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(P)
    #5. Normalized kernel matrix K
    K = norm_kernel(P, W)
    if printTimes:
        end = time.time()
        print(f"5. Kernel matrix K:\t\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(K)
    #6. diagonal normalization matrix Q
    Q = diag_norm(K)
    if printTimes:
        end = time.time()
        print(f"6. Diagonal normalization matrix Q:\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(Q)
    #7. symmetric matrix T
    T, Q_pow = symm_mat(Q, K)
    if printTimes:
        end = time.time()
        print(f"7. Symmetric matrix T:\t\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(T)
    #8. find the L+1 largest eigenvalues and eigenvectors of T
    a_l, v_l = eig_symm_mat(T, L)
    if printTimes:
        end = time.time()
        print(f"8. Eigenvalues of T:\t\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(a_l)
        print(v_l)
    #9. eigenvalues of T^(1/e)
    lambda_l = eigenvalues(a_l, e)
    if printTimes:
        end = time.time()
        print(f"9. Eigenvalues of T^(1/e):\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(lambda_l)
    #10. eigenvectors of T=Q^(-1)K
    phi_l = eigenvectors(v_l, Q_pow)
    if printTimes:
        end = time.time()
        print(f"10. Eigenvectors of Q^(-1)K:\t\t{end-start:.3f}s")
        start = end
    if printOutput:
        print(phi_l)
    #return 9. and 10. as tuple but change order from descending to ascending
    return lambda_l[::-1], phi_l[:, ::-1]

#The following are the used helper functions for the diffusion map algorithm, for the sake of clarity, they are not documented using docstring, but with the step number in front

#1. distance matrix D
def distance_map(X, Y):
    return spatial.distance_matrix(X, Y)


#2. Epsilon
def epsilon(D):
    return 0.05 * np.amax(D)


#3. Kernel matrix W
def kernel(e, D):
    return np.exp(- D ** 2 / e)


#4. Inverse of diagonal normalization matrix P
def diag_norm_inv(W):
    return np.diag(np.reciprocal(W.sum(axis=1)))


#5. Normalized kernel matrix K
def norm_kernel(P, W):
    return np.dot(P, np.dot(W, P))


#6. diagonal normalization matrix Q
def diag_norm(K):
    return np.diag(K.sum(axis=1))


#7. symmetric matrix T which returns also Q^(-0.5) for later use
def symm_mat(Q, K):
    Q_pow = np.diag(np.reciprocal(np.sqrt(Q.diagonal())))
    return np.dot(Q_pow, np.dot(K, Q_pow)), Q_pow


#8. find the L+1 largest eigenvalues of T with the help of 
def eig_symm_mat(T, L):
    return la.eigh(T, subset_by_index=[T.shape[0] - (L + 1), T.shape[0] - 1])


#9. eigenvalues of T^(1/e)
def eigenvalues(a_l, e):
    return a_l ** (1/(2*e))


#10. eigenvectors of T=Q^(-1)K
def eigenvectors(v_l, Q_pow):
    return np.dot(Q_pow, v_l)
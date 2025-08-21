import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, a * b)
    # Center the data
    X_mean = np.mean(X_flat, axis=0)
    X_centered = X_flat - X_mean
    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Select top k eigenvectors
    basis = eigvecs[:, :k]
    # Normalize basis vectors
    basis = basis / np.linalg.norm(basis, axis=0)
    return basis
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    N, a, b = X.shape
    X_flat = X.reshape(N, a * b)
    # Project each image onto the basis vectors
    projections = np.dot(X_flat, basis)
    return projections
    # END TODO
    
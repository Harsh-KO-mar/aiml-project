import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def initialise(self, X_train):
        """
        Initialize the self.centroids class variable, using the "k-means++" method, 
        Pick a random data point as the first centroid,
        Pick the next centroids with probability directly proportional to their distance from the closest centroid
        Function returns self.centroids as an np.array
        USE np.random for any random number generation that you may require 
        (Generate no more than K random numbers). 
        Do NOT use the random module at ALL!
        """
        # TODO
        N = X_train.shape[0]
        centroids = []
        # Pick first centroid randomly
        idx = np.random.choice(N)
        centroids.append(X_train[idx])
        for _ in range(1, self.n_clusters):
            # Compute distance to nearest centroid for each point
            dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X_train])
            probs = dist_sq / dist_sq.sum()
            next_idx = np.random.choice(N, p=probs)
            centroids.append(X_train[next_idx])
        self.centroids = np.array(centroids)
        return self.centroids
        # END TODO
    def fit(self, X_train):
        """
        Updates the self.centroids class variable using the two-step iterative algorithm on the X_train dataset.
        X_train has dimensions (N,d) where N is the number of samples and each point belongs to d dimensions
        Ensure that the total number of iterations does not exceed self.max_iter
        Function returns self.centroids as an np array
        """
        # TODO
        self.initialise(X_train)
        for _ in range(self.max_iter):
            # Assign clusters
            labels = np.array([
                np.argmin([np.linalg.norm(x - c) for c in self.centroids])
                for x in X_train
            ])
            # Update centroids
            new_centroids = np.array([
                X_train[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids
        # END TODO
    
    def evaluate(self, X):
        """
        Given N data samples in X, find the cluster that each point belongs to 
        using the self.centroids class variable as the centroids.
        Return two np arrays, the first being self.centroids 
        and the second is an array having length equal to the number of data points 
        and each entry being between 0 and K-1 (both inclusive) where K is number of clusters.
        """
        # TODO
        labels = np.array([
            np.argmin([np.linalg.norm(x - c) for c in self.centroids])
            for x in X
        ])
        return self.centroids, labels
        # END TODO

def evaluate_loss(X, centroids, classification):
    loss = 0
    for idx, point in enumerate(X):
        loss += np.linalg.norm(point - centroids[classification[idx]])
    return loss

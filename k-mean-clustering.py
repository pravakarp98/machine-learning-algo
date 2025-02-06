# implementation of thr k-mean algorith, and use it for image compression.

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *

def find_closet_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input Values
        centroids (ndarray): (K, n) centroids
        
    Rturns:
        idx (array_like): (m,) closest centroids
    """
    
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    
    return idx

X = load_data()
print(f"First five elements of X are: \n{X[:5]}")
print(f"The shape of X is:", X.shape)

initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx = find_closet_centroids(X, initial_centroids)

print(f"First three elements in idx are:", idx[:3])

find_closest_centroids_test(find_closet_centroids)

# Computing cetroid means
def compute_centroids(X, idx, K):
    """Returns the new centroids by computing the means of the data points assigned to each centroid.

    Args:
        X (ndarray): (m, n) Data points
        idx (ndarray): (m, ) Array containing index of closest centroid for each
        example in X. concretely, idx[i] contains the index of
        the centroid closest to example i
        K (int): number of centroids
        
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis = 0)
        
    return centroids

K = 3
centroids = compute_centroids(X, idx, K)
print(f"The centroids are: {centroids}")

compute_centroids_test(compute_centroids)
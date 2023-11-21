"""
Module for implementation of K-means clustering algorithm.

This module provides functions for clustering data using K-means algorithm. 
It is closely tied with numpy library for efficient numerical computations.

Functions:
generate_random_centroids(data: np.array, num_clusters: int) -> np.array: 
    Generates initial centroids as random data points.

compute_distance(point: np.array, centroids: np.array) -> np.array: 
    Computes Euclidean distance from each centroid to a point.

find_closest_centroid(distances: np.array) -> int:
    Returns the index of the closest centroid.

compute_centroid_mean(data_points: np.array) -> np.array: 
    Computes the mean point of input data points.

kmeans(data: np.array, num_clusters: int, max_iterations: int , max_depth: int, error_margin: int ) -> dict:
    Implementation of K-means clustering algorithm.
"""

import numpy as np

def generate_random_centroids(data, num_clusters):
    """ Generates initial random centroids for k-means algorithm """

    random_indices = np.random.randint(0, len(data), size=num_clusters)
    random_centroids = data[random_indices]

    return random_centroids

def compute_distance(point, centroids):
    """ Compute the distance of a point from the centroids """
    distances = np.sum((point - centroids)**2, axis=1)
    return distances

def find_closest_centroid(distances):
    """ Find the index of closest centroid """
    return np.argmin(distances)

def compute_centroid_mean(data_points):
    """ Compute the mean for the data points """
    return np.mean(data_points, axis=0)

def kmeans(data, num_clusters, max_iterations = 3, max_depth=1000, error_margin = 10):
    """
    Implementation of the K-means clustering algorithm.

    Parameters:
    data : np.array
        The input data to be partitioned into clusters.
    num_clusters : int
        The number of clusters to partition the data.
    max_iterations : int, optional
        The maximum number of times the algorithm will run k-means before terminating (default is 3).
    max_depth: int, optional
        The maximum number of iterations in each run before centroids are recalculated (default is 1000).
    error_margin: int, optional
        The smallest change in centroids' locations that is considered significant (default is 10).
 
    Returns:
    iteration_loss: dict
        A dictionary where the key is the iteration number and the value is a list containing 
            the final loss for the iteration,
            the final centroid locations,
            and the final assignment of data points to clusters.
    """

    iteration_loss = {}

    for iteration in range(max_iterations):

        centroids = generate_random_centroids(data, num_clusters)
        assignments = []
        depth_losses = []

        for i in range(max_depth):

            depth_loss = 0
            cluster_assignments = []

            for data_point in data:

                distances = compute_distance(data_point, centroids)
                closest_centroid_index = find_closest_centroid(distances)
                cluster_assignments.append(closest_centroid_index)

                depth_loss += distances[closest_centroid_index]

            if i > 0 and abs(depth_loss - depth_losses[-1]) < error_margin :
                break

            depth_losses.append(depth_loss)

            assignments = cluster_assignments

            for idx, _ in enumerate(centroids):
                centroid_points = data[np.array(assignments) == idx]
                centroids[idx] = compute_centroid_mean(centroid_points)

        iteration_loss[iteration] = [depth_losses[-1], centroids, assignments]

    return iteration_loss

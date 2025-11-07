import os
import argparse
import sys
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from config import *


def get_number_of_samples(n_clusters: int) -> int:
    '''
    Calculate the number of samples to collect for a given n_clusters.

    Parameters:
    n_clusters (int): The n_clusters for which the function will calculate the number of samples

    Returns:
    int: The number of samples to collect for the given n_clusters
    '''
    # Calculate the number of samples using the function f(x) = m * x + q
    # where f(x) is the number of samples and x is n_clusters
    # we know that f(MIN_N_CLUSTERS) = N_SAMPLES_FOR_MIN_N_CLUSTERS and f(MAX_N_CLUSTERS) = N_SAMPLES_FOR_MAX_N_CLUSTERS
    # so we have to solve the following linear system:
    # { N_SAMPLES_FOR_MIN_N_CLUSTERS = m * MIN_N_CLUSTERS + q
    # { N_SAMPLES_FOR_MAX_N_CLUSTERS = m * MAX_N_CLUSTERS + q

    m = (N_SAMPLES_FOR_MAX_N_CLUSTERS - N_SAMPLES_FOR_MIN_N_CLUSTERS) / (MAX_N_CLUSTERS - MIN_N_CLUSTERS)
    q = N_SAMPLES_FOR_MIN_N_CLUSTERS - (m * MIN_N_CLUSTERS)

    return round((m * n_clusters) + q)


def get_samples_and_centroids(k: int, kmeans: KMeans, X: np.ndarray, image_names: list, n_samples: int, collect_random=False) -> tuple:
    '''
    Get centroids and samples for every cluster for a specific k (n_clusters).
    If collect_random = True the selection of the samples is done by selecting half of the n_samples randomly,
    and the other half by incrementing the distance from the centroids, ignoring those in the selection.
    Otherwise if collect_random = False the selection of the samples is done only by incrementing distance.
    Note that some clusters might have less that n_samples elements, and so the function will return the
    amout of elements that the cluster have.

    Parameters:
    k (int): The number of clusters
    kmeans (KMeans): The KMeans object that divided the images in clusters
    X (ndarray): The arrays (n_samples, n_features) that kmeans fitted in clusters
    image_names (list): The list of image's names in the same order of X
    n_samples (int): The number of samples to collect
    collect_random (bool): If the selection of samples have to include random points (default False)

    Returns:
    list: The collected samples for every cluster for this k
    '''
    if not all(isinstance(x, str) for x in image_names):
        raise ValueError("image_names has to be a list of names")
    
    if k != kmeans.cluster_centers_.shape[0]:
        raise ValueError(f"k it's not equal to n_clusters of kmeans = {kmeans.cluster_centers_.shape[0]}")
    
    if X.ndim != 2:
        raise ValueError("X has to be a bidimensional array")
    
    if X.shape[1] != kmeans.cluster_centers_.shape[1]:
        raise ValueError("X and kmeans have a different n_features")
    
    samples = []
    centroids = []

    for cluster_id in range(k):
        cluster_points = X[kmeans.labels_ == cluster_id]

        if len(cluster_points) == 0:
            # control for an empty cluster case
            samples.append([])
            centroids.append(None)
            continue

        # since the cluster might have less points than n_samples
        actual_n_samples = min(n_samples, len(cluster_points) - 1)  # - 1 because we have to ignore the centroid

        centroid = kmeans.cluster_centers_[cluster_id]
        
        # distances of all points in cluster from centroid
        distances = np.linalg.norm(centroid - cluster_points, axis=1)
        sorted_indexes = np.argsort(distances)  # points closest to centroid first
        
        # decide how many samples to take by each method
        # if collect_random = True, half of the samples will be collected randomly and the other half by incrementig distance
        # otherwise samples are only collected by incrementing distance
        if collect_random:
            n_distance = actual_n_samples // 2
            n_random = actual_n_samples - n_distance
        else:
            n_distance = actual_n_samples
            n_random = 0
        
        # indexes (for X or image_names) for every point of the cluster
        original_indexes = np.where(kmeans.labels_ == cluster_id)[0]

        # append the closest point to the centroid in centroids
        centroids.append(image_names[original_indexes[sorted_indexes[0]]])
        
        # select by centroid proximity
        step = max(len(sorted_indexes) // max(n_distance, 1), 1)
        distance_selected = []
        for i in range(n_distance):
            distance_selected.append(sorted_indexes[i * step + 1])  # + 1 to ignore che closest one (the centroid)
        
        # select randomly from the remaining points
        remaining_indexes = list(set(range(len(cluster_points))) - set(distance_selected))
        if remaining_indexes:  # only if there are leftover points
            random_selected = np.random.choice(
                remaining_indexes,
                size=min(n_random, len(remaining_indexes)),
                replace=False
            )
        else:
            random_selected = []
        
        # merge both selections, map to original indexes
        selected_indexes = np.array(distance_selected + list(random_selected))
        selected_original_indexes = original_indexes[selected_indexes]
        
        # convert indexes to image names
        samples_for_cluster = [image_names[i] for i in selected_original_indexes]
        samples.append(samples_for_cluster)

    return samples, centroids


def clustering(clustering_path: str, *feature_paths: str) -> None:
    '''
    Perform KMeans clustering on image features for multiple values of k
    and save the results as JSON files.

    Parameters:
    clustering_path (str): The path where the results of the clustering will be saved
    *feature_paths (str): One or more paths where the features, saved as .pt, will be collected for KMeans

    Returns:
    None
    '''

    if not os.path.exists(clustering_path):
        raise ValueError(f"clustering_path provided doesn't exists: {clustering_path}")
    for fpath in feature_paths:
        if not os.path.exists(fpath):
            raise ValueError(f"clustering_path provided doesn't exists: {fpath}") 
        
    features = []  # this list will contain all features
    image_names = []  # this list will contain the names of the images in the same order of features

    # get all the features
    for fpath in feature_paths:
        for feature_name in os.listdir(fpath):
            if not feature_name.endswith(".pt"):
                continue  # ignore other types of files
            
            feature_path = os.path.join(fpath, feature_name)
            feature = torch.load(feature_path).numpy()

            # the vector might be (n_features,) so we convert the vector to (1, n_features) to be sure
            feature = feature.reshape(1, -1)

            features.append(feature)
            image_names.append(os.path.splitext(feature_name)[0])

    # X has to be (n_samples, n_features)
    X = np.vstack(features)
    

    all_labels = {image_name: [] for image_name in image_names}  # this dictionary will contain for each image's name (key) the label for each cluster


    # apply kmeans with many different k
    for k in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X)

        # save the labels for this k (n_clusters)
        for image_name, label in zip(image_names, kmeans.labels_):
            all_labels[image_name].append(int(label))

        # save the number of elements for every cluster
        cluster_counts = [int(np.sum(kmeans.labels_ == cluster_id)) for cluster_id in range(k)]

        # collect some samples for every clusters
        samples, centroids = get_samples_and_centroids(k, kmeans, X, image_names, get_number_of_samples(k))

        # print the results on the terminal
        print("For k =", k, "we have the following results:")
        print("- Inertia:", kmeans.inertia_)
        print("- Silhoutte Score:", silhouette_score(X, kmeans.labels_))
        print("- Cluster counts:", cluster_counts)

        # save on file
        data_to_save = {
            "k": k,
            "inertia_score": float(kmeans.score(X)),
            "silhouette_score": float(silhouette_score(X, kmeans.labels_)),
            "cluster_counts": cluster_counts,
            "centroids": centroids,
            "samples": samples
        }
        filename = get_clustering_filename(k)
        
        with open(os.path.join(clustering_path, filename), "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Results saved in {os.path.join(clustering_path, filename)}\n")

    # save all labels as a single json file
    # we need those to know on which image to operate if we see any issues with the clustering
    with open(os.path.join(clustering_path, ALL_CLUSTERING_LABELS_FILENAME), "w") as f:
        json.dump(all_labels, f, indent=4)
    
    print(f"{ALL_CLUSTERING_LABELS_FILENAME} saved in:", os.path.join(clustering_path, ALL_CLUSTERING_LABELS_FILENAME), "\n")

    print(f"Done clustering for {feature_paths}\n")

# =============================================================================


def execute_clustering(dataset: str) -> None:
    '''
    Starts the clustering for the desired dataset.
    
    Parameters:
    dataset: {DEFAULT, ADDED, AUGMENTED}

    Returns:
    None
    '''
    if dataset not in ["DEFAULT", "ADDED", "AUGMENTED"]:
        raise ValueError(f"Error: not valid dataset: {dataset} not in  [DEFAULT, ADDED, AUGMENTED]")
    
    print(f"Clustering for the dataset {dataset}...")

    if dataset == "DEFAULT":
        clustering(DEFAULT_CLUSTERING_DIRECTORY, FEATURES_DIRECTORY)
    elif dataset == "ADDED":
        clustering(ADDED_CLUSTERING_DIRECTORY, FEATURES_DIRECTORY, ADDED_FEATURES_DIRECTORY)
    else:
        clustering(AUGMENTED_CLUSTERING_DIRECTORY, AUGMENTED_FEATURES_DIRECTORY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts the clustering fot the desired dataset."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["DEFAULT", "ADDED", "AUGMENTED"],
        help="The dataset of features (DEFAULT, ADDED, AUGMENTED)",
    )
    args = parser.parse_args()
    execute_clustering(args.dataset)

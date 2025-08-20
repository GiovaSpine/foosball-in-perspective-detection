import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import random
from config import *



def collect_samples(k: int, kmeans: KMeans, X: np.ndarray, image_names: list) -> list:
    '''
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
    n_samples = get_number_of_samples(k)

    for cluster_id in range(k):
        cluster_points = X[kmeans.labels_ == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # distances of all points in cluster from centroid
        distances = np.linalg.norm(centroid - cluster_points, axis=1)
        sorted_indexes = np.argsort(distances)  # points closest to centroid first
        
        # decide how many samples to take by each method
        n_centroid = n_samples // 2
        n_random = n_samples - n_centroid
        
        original_indexes = np.where(kmeans.labels_ == cluster_id)[0]
        
        # select by centroid proximity
        step = max(len(sorted_indexes) // max(n_centroid, 1), 1)
        centroid_selected = []
        for i in range(min(n_centroid, len(cluster_points))):
            centroid_selected.append(sorted_indexes[i * step])
        
        # select randomly from the remaining points
        remaining_indexes = list(set(range(len(cluster_points))) - set(centroid_selected))
        if remaining_indexes:  # only if there are leftover points
            random_selected = np.random.choice(
                remaining_indexes,
                size=min(n_random, len(remaining_indexes)),
                replace=False
            )
        else:
            random_selected = []
        
        # merge both selections, map to original indexes
        selected_indexes = np.array(centroid_selected + list(random_selected))
        selected_original_indexes = original_indexes[selected_indexes]
        
        # convert indexes to image names
        samples_for_cluster = [image_names[i] for i in selected_original_indexes]
        samples.append(samples_for_cluster)

    return samples


def save_labels():
    '''
    save the labels for each image
    '''
    pass


def clustering(features_path, clustering_path):
    '''
    Perform KMeans clustering on image features for multiple values of k
    and save the results as JSON files.

    Parameters:
    features_path (str): The path of the features to cluster
    clustering_path (str): The path where the results of the clustering will be saved

    Returns:
    None
    '''
    if not os.path.exists(features_path) or not os.path.exists(clustering_path):
        error_message = ""
        if not os.path.exists(features_path):
            error_message += f"The following path doesn't exists: {features_path}\n"
        if not os.path.exists(clustering_path):
            error_message += f"The following path doesn't exists: {clustering_path}\n"    
        raise ValueError(error_message)
        
    features = []  # this list will contain all features
    image_names = []  # this list will contain the names of the images in the same order of features

    # get all the features
    for feature_name in os.listdir(features_path):
        if not feature_name.endswith(".pt"):
            continue  # ignore other types of files
        
        feature_path = os.path.join(features_path, feature_name)
        feature = torch.load(feature_path).numpy()

        # the vector might be (n_features,) so we convert the vector to (1, n_features) to be sure
        feature = feature.reshape(1, -1)

        features.append(feature)
        image_names.append(os.path.splitext(feature_name)[0])

    # X has to be (n_samples, n_features)
    X = np.vstack(features)


    # apply kmeans with many different k
    for k in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X)

        # print the results on the terminal
        print("For k =", k, "we have the following results:")
        print("- Inertia:", kmeans.inertia_)
        print("- Silhoutte Score:", silhouette_score(X, kmeans.labels_))
        print("- Centroids:", kmeans.cluster_centers_.shape)
        print("- Labels:", kmeans.labels_)

        # save the number of elements for every cluster
        cluster_counts = [int(np.sum(kmeans.labels_ == cluster_id)) for cluster_id in range(k)]

        # collect some samples for every clusters
        samples = collect_samples(k, kmeans, X, image_names)

        # save on file
        data_to_save = {
            "k": k,
            "inertia_score": float(kmeans.score(X)),
            "silhouette_score": float(silhouette_score(X, kmeans.labels_)),
            "cluster_counts": cluster_counts,
            "samples": samples
        }
        filename = f"{CLUSTERING_FILENAME}{k}.json"
        
        with open(os.path.join(clustering_path, filename), "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Results saved in {os.path.join(clustering_path, filename)}\n")

    print(f"Done clustering for {features_path}\n")


clustering(FEATURES_DIRECTORY, CLUSTERING_DIRECTORY)
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from config import *


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
        clusters_counts = [int(np.sum(kmeans.labels_ == cluster_id)) for cluster_id in range(k)]

        # collect some samples for every clusters that are evenly distributed in the cluster
        samples = []
        n_samples = get_number_of_samples(k)

        for cluster_id in range(k):
            cluster_points = X[kmeans.labels_ == cluster_id]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(centroid - cluster_points, axis=1)
            sorted_indexes = np.argsort(distances)
            selected_indexes = []
            step = max(len(sorted_indexes) // n_samples, 1)
            for i in range(min(n_samples, len(cluster_points))):
                selected_indexes.append(sorted_indexes[i*step])
            
            original_indexes = np.where(kmeans.labels_ == cluster_id)[0]
            selected_original_indexes = original_indexes[selected_indexes]
            samples_for_cluster = [image_names[i] for i in selected_original_indexes]
            samples.append(samples_for_cluster)

        # save on file
        data_to_save = {
            "k": k,
            "inertia_score": float(kmeans.score(X)),
            "silhouette_score": float(silhouette_score(X, kmeans.labels_)),
            "clusters_counts": clusters_counts,
            "samples": samples
        }
        filename = f"{CLUSTERING_FILENAME}{k}.json"
        
        with open(os.path.join(clustering_path, filename), "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Results saved in {os.path.join(clustering_path, filename)}\n")

    print(f"Done clustering for {features_path}\n")


clustering(IMAGES_FEATURES_DIRECTORY, IMAGES_CLUSTERING_DIRECTORY)
clustering(ZOOMED_IMAGES_FEATURES_DIRECTORY, ZOOMED_IMAGES_CLUSTERING_DIRECTORY)
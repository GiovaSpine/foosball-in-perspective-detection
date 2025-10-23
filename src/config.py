from pathlib import Path

# =========================================================

# DIRECTORIES

SRC_DIRECTORY = Path(__file__).resolve().parent  # absolute path of the src directory, knowing that this file is in the src folder
PROJECT_DIRECTORY = SRC_DIRECTORY.parent  # absolute path of the project directory, knowing that src is in the project direcotry

IMAGES_DATA_DIRECTORY = f"{PROJECT_DIRECTORY}/data/images"  # absolute directory of the data/images

FEATURES_DIRECTORY = f"{PROJECT_DIRECTORY}/data/features"  # absolute directory of the features obtained with a pretrained model on the data

LABELS_DIRECTORY = f"{PROJECT_DIRECTORY}/data/labels"  # absolute directory of the labels/annotations of the data

CLUSTERING_DIRECTORY = f"{PROJECT_DIRECTORY}/results/clustering"  # absolute directory where we will save the results of KMeans

CLUSTER_COUNTS_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/cluster-counts-images"  # absolute directory where we will save the images of cluster's counts

CLUSTER_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/cluster-images"  # absolute directory where we will save the images of some samples of a cluster

IMAGES_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/images_dataframe.parquet"  # absolute directory where we will save the dataframe of the data/images
LABELS_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/labels_dataframe.parquet"  # absolute directory where we will save the dataframe of the labels/annotations

# =========================================================

# ALLOWED EXTENSIONS FOR IMAGES DATA AND LABELS

IMAGES_DATA_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

LABELS_EXTENSION = ".txt"

# =========================================================

# FILENAMES

ALL_CLUSTERING_LABELS_FILENAME = "all_clustering_labels.json"

def get_clustering_filename(k: int) -> str:
    '''
    Get the filename for the clustering results for a specific k.

    Parameters:
    k (int): The number of clusters

    Returns:
    str: The filename
    '''
    if k < MIN_N_CLUSTERS or k > MAX_N_CLUSTERS:
        raise ValueError("k has to be in the interval [{MIN_N_CLUSTERS}, {MAX_N_CLUSTERS}]")
    
    return f"clustering_for_k_equal_{k}.json"

def get_cluster_counts_image_filename(k: int) -> str:
    '''
    Get the filename for the clustering counts graph for a specific k.

    Parameters:
    k (int): The number of clusters

    Returns:
    str: The filename
    '''
    if k < MIN_N_CLUSTERS or k > MAX_N_CLUSTERS:
        raise ValueError("k has to be in the interval [{MIN_N_CLUSTERS}, {MAX_N_CLUSTERS}]")
    
    return f"cluster_counts_image_k_{k}.png"

def get_cluster_image_filename(k: int, cluster_id: int) -> str:
    '''
    Get the filename for the image of cluster, identified with k and cluster_id.

    Parameters:
    k (int): The number of clusters
    cluster_id (int): The id of the cluster

    Returns:
    str: The filename
    '''
    if k < MIN_N_CLUSTERS or k > MAX_N_CLUSTERS:
        raise ValueError("k has to be in the interval [{MIN_N_CLUSTERS}, {MAX_N_CLUSTERS}]")
    
    if cluster_id < 0 or cluster_id >= k:
        raise ValueError(f"cluster_id has to be in the interval [0, {k - 1}]")
    
    return f"cluster_image_k_{k}_cluster_id_{cluster_id}.png"


# =========================================================

# CLUSTERING PARAMETERS

MIN_N_CLUSTERS = 2  # the lowest number of clusters to apply KMeans
MAX_N_CLUSTERS = 30  # the highest number of clusters to apply KMeans
N_SAMPLES_FOR_MIN_N_CLUSTERS = 30  # the number of samples images for every cluster, to collect when n_clusters = MIN_N_CLUSTERS
N_SAMPLES_FOR_MAX_N_CLUSTERS = 15  # the number of samples images for every cluster, to collect when n_clusters = MAX_N_CLUSTERS

# =========================================================
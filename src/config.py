

# =========================================================

# DIRECTORIES

IMAGES_DATA_DIRECTORY = "data/images"

FEATURES_DIRECTORY = "data/features"

LABELS_DIRECTORY = "data/labels"

CLUSTERING_DIRECTORY = "results/clustering"

CLUSTER_COUNTS_IMAGES_DIRECTORY = "results/images/cluster-counts-images"

CLUSTER_IMAGES_DIRECTORY = "results/images/cluster-images"

ROOT_DIRECTORY_FROM_NOTEBOOKS = ".."

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
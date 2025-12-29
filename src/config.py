import re
from pathlib import Path

# =========================================================

# DIRECTORIES

SRC_DIRECTORY = Path(__file__).resolve().parent  # absolute path of the src directory, knowing that this file is in that folder
PROJECT_DIRECTORY = SRC_DIRECTORY.parent  # absolute path of the project directory, knowing that src is in the project directory

# absolute path of the data/images
IMAGES_DATA_DIRECTORY = f"{PROJECT_DIRECTORY}/data/images"

# absolute path of the features obtained with a pretrained model on the data
FEATURES_DIRECTORY = f"{PROJECT_DIRECTORY}/data/features"

# absolute path of the labels/annotations of the data
LABELS_DIRECTORY = f"{PROJECT_DIRECTORY}/data/labels"

# absolute path of the added data/images
ADDED_IMAGES_DATA_DIRECTORY = f"{PROJECT_DIRECTORY}/added-data/images"

# absolute path of the added features obtained with a pretrained model on the data
ADDED_FEATURES_DIRECTORY = f"{PROJECT_DIRECTORY}/added-data/features"

# absolute path of the added labels/annotations of the data
ADDED_LABELS_DIRECTORY = f"{PROJECT_DIRECTORY}/added-data/labels"

# absolute path of the augmented data/images
AUGMENTED_IMAGES_DATA_DIRECTORY = f"{PROJECT_DIRECTORY}/augmented-data/images"

# absolute path of the augmented features obtained with a pretrained model on the data
AUGMENTED_FEATURES_DIRECTORY = f"{PROJECT_DIRECTORY}/augmented-data/features"

# absolute path of the augmented labels/annotations of the data
AUGMENTED_LABELS_DIRECTORY = f"{PROJECT_DIRECTORY}/augmented-data/labels"

# absolute path where we will save the results of KMeans on the first/default data
DEFAULT_CLUSTERING_DIRECTORY = f"{PROJECT_DIRECTORY}/results/clustering/default-clustering"

# absolute path where we will save the results of KMeans on the data, where there are added some images
ADDED_CLUSTERING_DIRECTORY = f"{PROJECT_DIRECTORY}/results/clustering/added-clustering"

# absolute path where we will save the results of KMeans on the augmented data
AUGMENTED_CLUSTERING_DIRECTORY = f"{PROJECT_DIRECTORY}/results/clustering/augmented-clustering"

# absolute path where we will save the images of cluster's counts for default clustering
DEFAULT_CLUSTER_COUNTS_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/default-cluster-counts-images"

# absolute path where we will save the images of some samples of a cluster for the default clustering
DEFAULT_CLUSTER_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/default-cluster-images"

# absolute path where we will save the images of cluster's counts for the added clustering
ADDED_CLUSTER_COUNTS_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/added-cluster-counts-images"

# absolute path where we will save the images of some samples of a cluster for the added clustering
ADDED_CLUSTER_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/added-cluster-images"

# absolute path where we will save the images of cluster's counts for the added clustering
AUGMENTED_CLUSTER_COUNTS_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/augmented-cluster-counts-images"

# absolute path where we will save the images of some samples of a cluster for the added clustering
AUGMENTED_CLUSTER_IMAGES_DIRECTORY = f"{PROJECT_DIRECTORY}/results/images/augmented-cluster-images"

# absolute path where we will save the dataframe of the data/images for the default dataset
DEFAULT_IMAGES_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/default_images_dataframe.parquet"

# absolute path where we will save the dataframe of the labels/annotations for the default dataset
DEFAULT_LABELS_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/default_labels_dataframe.parquet"

# absolute path where we will save the dataframe of the data/images for the added dataset
ADDED_IMAGES_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/added_images_dataframe.parquet"

# absolute path where we will save the dataframe of the labels/annotations for the added dataset
ADDED_LABELS_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/added_labels_dataframe.parquet"

# absolute path where we will save the dataframe of the data/images for the augmented dataset
AUGMENTED_IMAGES_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/augmented_images_dataframe.parquet"

# absolute path where we will save the dataframe of the labels/annotations for the augmented dataset
AUGMENTED_LABELS_DATAFRAME_DIRECTORY = f"{PROJECT_DIRECTORY}/results/dataframes/augmented_labels_dataframe.parquet"

# absolute path of the config file for yolo
YOLO_CONFIG_DIRECTORY = f"{PROJECT_DIRECTORY}/yolo/yolo-config.yaml"

# absolute path where the runs will be saved
YOLO_RUNS_DIRECTORY = f"{PROJECT_DIRECTORY}/yolo/runs"

# txt file that contains the paths of the images for the training set
TRAIN_TXT_DIRECTORY = f"{PROJECT_DIRECTORY}/yolo/data-sets-division/train.txt"

# txt file that contains the paths of the images for the validation set
VALIDATION_TXT_DIRECTORY = f"{PROJECT_DIRECTORY}/yolo/data-sets-division/val.txt"

# txt file that contains the paths of the images for the test set
TEST_TXT_DIRECTORY = f"{PROJECT_DIRECTORY}/yolo/data-sets-division/test.txt"

# =========================================================

# ALLOWED EXTENSIONS FOR IMAGES DATA AND LABELS

IMAGES_DATA_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

LABELS_EXTENSION = ".txt"

# =========================================================

# FILENAMES

# name for the json file that for each image, will contain a list that specify for each k, which cluster the image belongs to
ALL_CLUSTERING_LABELS_FILENAME = "all_clustering_labels.json"

# name for the json file that will contain the results of a KMeans for a specific k
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

# name of the image that shows for a specific k, all the cluster sizes
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

# name of the image that shows some samples of a specific cluster of a specific k analysis
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

# name of the augmented image obtained from a original image (source)
def get_augmented_image_name(source_image_name: str) -> str:
    '''
    Get the filename for an augmented image, created from a source image, that doesn't have an extension

    Parameters:
    source_image_name (str): The source/orginal image without extension

    Returns:
    str: The filename in the form prefix_number_source image name.extension
    '''
    # it has to have a structure: prefix_number_source image name.extension

    folder = Path(AUGMENTED_IMAGES_DATA_DIRECTORY)
    pattern = re.compile(rf"da_(\d+)_{re.escape(source_image_name)}\.png")

    max_k = -1
    # let's search to see if there is already an image with this name, with a different number
    for file in folder.glob(f"da_*_{source_image_name}.png"):
        match = pattern.match(file.name)
        if match:
            k = int(match.group(1))
            max_k = max(max_k, k)

    next_k = max_k + 1   # first will be 0 if none exists
    return f"da_{next_k}_{source_image_name}.png"

# =========================================================

# CLUSTERING PARAMETERS

MIN_N_CLUSTERS = 2  # the lowest number of clusters to apply KMeans
MAX_N_CLUSTERS = 20  # the highest number of clusters to apply KMeans
N_SAMPLES_FOR_MIN_N_CLUSTERS = 30  # the number of samples images for every cluster, to collect when n_clusters = MIN_N_CLUSTERS
N_SAMPLES_FOR_MAX_N_CLUSTERS = 15  # the number of samples images for every cluster, to collect when n_clusters = MAX_N_CLUSTERS

# =========================================================

# given the direction of a foosball table
# (where the direction is the direction from the center of the upper rectangle (given by the first 4 keypoint)
# towards the point in the middle between the first 2 keypoints)
# theta is simply the angle of this direction
# for this project theta should be between 40 and 140

MIN_THETA = 40.0
MAX_THETA = 140.0


# =========================================================


IMAGES_DIRECTORY = "data/images"

IMAGES_EXTENSIONS = [".jpg", ".jpeg", ".png"]

FEATURES_DIRECTORY = "data/features"

CLUSTERING_DIRECTORY = "results/clustering"

CLUSTERING_FILENAME = "clustering_for_k_equal_"

MIN_N_CLUSTERS = 2
MAX_N_CLUSTERS = 30
N_SAMPLES_FOR_MIN_N_CLUSTERS = 50  # the number of samples images for every cluster, to collect when n_clusters = MIN_N_CLUSTERS
N_SAMPLES_FOR_MAX_N_CLUSTERS = 30  # the number of samples images for every cluster, to collect when n_clusters = MAX_N_CLUSTERS

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

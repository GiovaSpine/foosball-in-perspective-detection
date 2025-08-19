

IMAGES_DIRECTORY = "data/images"
ZOOMED_IMAGES_DIRECTORY = "data/zoomed-images"

IMAGES_EXTENSIONS = [".jpg", ".jpeg", ".png"]

IMAGES_FEATURES_DIRECTORY = "data/features/images-features"
ZOOMED_IMAGES_FEATURES_DIRECTORY = "data/features/zoomed-images-features"

IMAGES_CLUSTERING_DIRECTORY = "results/clustering/images-clustering"
ZOOMED_IMAGES_CLUSTERING_DIRECTORY = "results/clustering/zoomed-images-clustering"

CLUSTERING_FILENAME = "clustering_for_k_equal_"

MIN_N_CLUSTERS = 2
MAX_N_CLUSTERS = 50
N_SAMPLES_FOR_MIN_N_CLUSTERS = 40  # the number of samples images for every cluster, to collect when n_clusters = MIN_N_CLUSTERS
N_SAMPLES_FOR_MAX_N_CLUSTERS = 8  # the number of samples images for every cluster, to collect when n_clusters = MAX_N_CLUSTERS

def get_number_of_samples(n_clusters: int) -> int:
    '''
    '''
    # { N_SAMPLES_FOR_MIN_N_CLUSTERS = m * MIN_N_CLUSTERS + q
    # { N_SAMPLES_FOR_MAX_N_CLUSTERS = m * MAX_N_CLUSTERS + q

    m = (N_SAMPLES_FOR_MAX_N_CLUSTERS - N_SAMPLES_FOR_MIN_N_CLUSTERS) / (MAX_N_CLUSTERS - MIN_N_CLUSTERS)
    q = N_SAMPLES_FOR_MIN_N_CLUSTERS - (m * MIN_N_CLUSTERS)

    return round((m * n_clusters) + q)

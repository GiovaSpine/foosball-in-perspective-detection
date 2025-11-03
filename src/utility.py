
# here we have some functions that are common in more files

import os
import re
import json
from config import *


def load_clustering_data(clustering_path: str) -> dict:
    '''
    Load clustering data as a dictionary in the form {k, data}.
    
    Parameters:
        clustering_path (str): The path where the results of the clustering will be saved.
    
    Returns:
        dict: The dictionary in the form {k, data}
    '''
    # get all the clustering json files (we have to ingore other types of json files)
    clustering_filename = re.sub(r'\d+', '*', get_clustering_filename(MIN_N_CLUSTERS))
    json_files = list(Path(clustering_path).glob(clustering_filename))

    def get_k_from_json(json_file):
        # get k value from a json
        with open(json_file, "r") as f:
            data = json.load(f)
        return data.get("k", 0)

    # sort for k
    json_files = sorted(json_files, key=get_k_from_json)

    clustering_data = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        clustering_data[data['k']] = data

    return clustering_data


def load_all_clustering_label(clustering_path: str) -> dict:
    '''
    Load all_clustering_label as a dictionary in the form {image name, list}.
    
    Parameters:
        clustering_path (str): The path where the results of the clustering where saved.
    
    Returns:
        dict: The dictionary in the form {image name, list}
    '''
    with open(os.path.join(clustering_path, ALL_CLUSTERING_LABELS_FILENAME), "r") as f:
        data = json.load(f)

    return data

# ---------------------------------------------------------


def find_image_path(image_name: str, *paths_to_look: str) -> str | None:
    '''
    Find the complete path given the image name without extension.
    
    Parameters:
        image_name (str): The name of the image without extension
        paths_to_look: One or more paths to search the image
    
    Returns:
        str o None: Complete path if found, None otherwise
    '''
    for path in paths_to_look:
        for ext in IMAGES_DATA_EXTENSIONS:
            for candidate_ext in [ext.lower(), ext.upper()]:
                candidate_path = os.path.join(path, image_name + candidate_ext)
                if os.path.exists(candidate_path):
                    return candidate_path
    return None


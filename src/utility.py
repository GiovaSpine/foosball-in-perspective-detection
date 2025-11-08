
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


def find_label_path(label_name: str, *paths_to_look: str) -> str | None:
    '''
    Find the complete path given the image name without extension.
    
    Parameters:
        image_name (str): The name of the label without extension
        paths_to_look: One or more paths to search the image
    
    Returns:
        str o None: Complete path if found, None otherwise
    '''
    for path in paths_to_look:
        candidate_path = os.path.join(path, label_name + LABELS_EXTENSION)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def label_loading(label_path: str) -> tuple:
    '''
    '''
    with open(label_path) as label:
        content = label.readline()
    numbers = [float(x) for x in content[1:].split()]  # we ignore the first number that is always 0, because we only have one class, the foosball table
    
    # the bounding box is rapresented as x and y of the center of the rectangle, and width and height of the rectangle
    bounding_box = (numbers[0], numbers[1], numbers[2], numbers[3])
    
    # the keypoints are rapresented as x, y and visibility (2: visible, 1: not visible, 0: not present)
    keypoints = []
    for i in range(4, 28, 3):
        keypoints.append((numbers[i], numbers[i+1], int(numbers[i+2])))

    return bounding_box, keypoints

def denormalize(width: int, height: int, bounding_box: list = None, keypoints: list = None) -> tuple:
    '''

    '''
    if bounding_box != None:
        bounding_box = (bounding_box[0] * width, bounding_box[1] * height, bounding_box[2] * width, bounding_box[3] * height)
    
    if keypoints != None:
        for i in range(0, 8):
            keypoints[i] = (keypoints[i][0] * width, keypoints[i][1] * height, keypoints[i][2])

    return bounding_box, keypoints

def normalize(width: int, height: int, bounding_box: list = None, keypoints: list = None) -> tuple:
    '''

    '''
    if bounding_box != None:
        bounding_box = (bounding_box[0] / width, bounding_box[1] / height, bounding_box[2] / width, bounding_box[3] / height)
    
    if keypoints != None:
        for i in range(0, 8):
            keypoints[i] = (keypoints[i][0] / width, keypoints[i][1] / height, keypoints[i][2])

    return bounding_box, keypoints
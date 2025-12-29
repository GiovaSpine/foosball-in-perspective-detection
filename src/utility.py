
# here we have some functions that are common in more files

import os
import re
import json
import random
import numpy as np
import cv2
from config import *

# LOADING CLUSTERINGS

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

# LOADING IMAGES AND LABELS

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
    Loads and returns the label from a specified path.
    
    Parameters:
        label_path (str): The path of the label
    
    Returns:
        tuple: The resulting label as bounding_box, keypoints
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
    if bounding_box is not None:
        bounding_box = (bounding_box[0] / width, bounding_box[1] / height, bounding_box[2] / width, bounding_box[3] / height)
    
    if keypoints is not None:
        for i in range(0, 8):
            keypoints[i] = (keypoints[i][0] / width, keypoints[i][1] / height, keypoints[i][2])

    return bounding_box, keypoints

# ---------------------------------------------------------

# MATH

def calculate_intersection(line1: tuple, line2: tuple) -> tuple:
    '''
    Calculate the intersection of 2 lines, each rapresented as 2 points in 2d.

    Parameters:
    line1 (tuple): The first line as (x1, y1), (x2, y2)
    line2 (tuple): The second line as (x1, y1), (x2, y2)

    Returns:
    tuple: The point of intersection if it exists
    '''
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def determinant(a, b):
        return a[0] * b[1] - a[1] * b[0]

    divisor = determinant(x_diff, y_diff)
    if divisor == 0:
        raise ValueError(f"The lines do not intersect. {line1}, {line2}")

    d = (determinant(*line1), determinant(*line2))
    x = determinant(d, x_diff) / divisor
    y = determinant(d, y_diff) / divisor
    return x, y

# ---------------------------------------------------------

# DATA AUGMENTATION


def save_augmented_data(
        image_name: str,
        augmented_image: np.ndarray,
        augmented_bbox: list,
        augmented_kps: list,
        horizontal_flip: bool = False
    ) -> bool:
    '''
    Saves an augmented image and its labels in the augmented data directory.

    Parameters:
    image_name (str): The name of the image without extension
    augmented_image (np.ndarray): The augmented image produced by albumentations
    augmented_bbox (list): The new bbox
    augmented_keypoints (list): The new kepoints
    horizontal_flip (bool): Wether in the tranformation there was the HorizontalFlip

    Returns:
    bool: True if it's able to save the image and labels, False otherwise
    '''
    # filenames and paths
    augmented_image_name = get_augmented_image_name(image_name)
    augmented_labels_name = os.path.splitext(augmented_image_name)[0] + LABELS_EXTENSION

    augmented_labels_path = os.path.join(AUGMENTED_LABELS_DIRECTORY, augmented_labels_name)
    augmented_image_path = os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, augmented_image_name)
    
    # we have to make the label
    # WARNING: we have a rule: the highest keypoint (lowest y) has to be either the first keypoint or the second
    # rotating the image can cause this rule to not be respected, so we have to check the keypoints

    # be careful about the points with visibility 0, because the value is 0 (the lowest)
    augmented_kps = np.array(augmented_kps)
    visible = augmented_kps[:, 2] != 0  # boolean mask of visible keypoints
    min_index = np.argmin(augmented_kps[visible, 1])  # min index for the visible
    min_index = np.arange(len(augmented_kps))[visible][min_index]  # convert to original array

    if min_index > 1:
        # the rule is NOT followed
        if min_index == 2 or min_index == 3:
            # we have to change 2 with 0, 3 with 1, and follow the clockwise order for the rest
            aux_kps = augmented_kps.copy()

            swap_map = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}

            for dst, src in swap_map.items():
                aux_kps[dst] = augmented_kps[src]

            augmented_kps = aux_kps
        else:
            # something went wrong, but it shouldn't be possibile to have one of the last 4 keypoints to be the highest
            # it can happen for excessive rotations
            print(f"WARNING: not valid augmented keypoints for {image_name}")
            return False
    
    if horizontal_flip:
        # it was applied HorizontalFlip
        # we have to reorder the keypoints to respect our rule
        aux_kps = augmented_kps.copy()

        swap_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}

        for dst, src in swap_map.items():
            aux_kps[dst] = augmented_kps[src]

        augmented_kps = aux_kps
    
    h, w = augmented_image.shape[:2]  # we need to normalize the keypoints
    _, augmented_kps = normalize(w, h, keypoints=augmented_kps)

    # save the labels
    with open(augmented_labels_path, "w") as f:
        f.write("0 ")
        for bbox_number in augmented_bbox:
            f.write(str(bbox_number) + " ")
        for kp in augmented_kps:
            # normalize the keypoits
            x = "0" if kp[0] == 0.0 else str(kp[0])
            y = "0" if kp[1] == 0.0 else str(kp[1])
            v = str(int(kp[2]))
            f.write(f"{x} {y} {v} ")

    # save the image
    cv2.imwrite(augmented_image_path, augmented_image)
    print(f"Saved augmented image in {augmented_image_path}")
    return True


def max_centered_scale(width: int, height: int, keypoints: list, margin_px: float = 0.0, max_scale: float = 1.2) -> float:
    '''
    Calculates the max scale possible to apply in the center of the image, so that all the resulting keypoints are inside
    the image, with a optional margin.
    To limit the scale there is the max_scale parameter.

    Paramters:
    width (int): The width of the image
    height (int): The height of the image
    keypoints (list): The list of keypoints that have to be inside the image
    margin_px (float): The margin from the border that the resulting keypoints can have at max
    max_scale (float): The max scale to not surpass in any way

    Returns:
    float: The calculated max scale
    '''
    # coordinates extraction
    pts = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        pts.append((x, y))

    half_w = width / 2.0
    half_h = height / 2.0

    # new position = scale * (old position - center) + center

    upper_limits = []

    for (x, y) in pts:
        dx = abs(x - half_w)
        dy = abs(y - half_h)

        # actual half (consider the margin_px)
        act_half_w = half_w - margin_px
        act_half_h = half_h - margin_px
        if act_half_w <= 0 or act_half_h <= 0:
            # can't scale: margin too big
            return 1.0

        if dx == 0:
            upper_x = float('inf')
        else:
            upper_x = act_half_w / dx

        if dy == 0:
            upper_y = float('inf')
        else:
            upper_y = act_half_h / dy

        upper = min(upper_x, upper_y)
        upper_limits.append(upper)

    if len(upper_limits) == 0:
        return float('inf')  # theoretically

    max_scale = min(upper_limits)

    # do not allow scale less that 1.0
    if max_scale < 1.0:
        return 1.0

    # to avoid having a big scale we have max_scale
    return min(max_scale, float(max_scale))


def min_centered_scale(width: int, height: int, keypoints: list, margin_px: float = 0.0, min_scale: float = 1.2) -> float:
    '''
    Calculates the min scale possible to apply in the center of the image, so that the reflection don't have the foosball table
    To limit the scale there is the min_scale parameter.

    Paramters:
    width (int): The width of the image
    height (int): The height of the image
    keypoints (list): The list of keypoints that have to be inside the image
    margin_px (float): The margin from the border that the resulting keypoints can have at max
    min_scale (float): The min scale to not surpass in any way

    Returns:
    float: The calculated min scale
    '''
    # scaling down means introducing some space
    # on width: (w - (w * scale)) / 2
    # on height: (h - (h * scale)) / 2
    # let's considera an image with a single point randomly
    # the reflected point is not in the image if the distance between the original point and the border
    # scaled down is lower than the space introduced

    # distance * scale < space ?

    min_x = (np.min(keypoints, axis=0)[0])
    max_x = (np.max(keypoints, axis=0)[0])
    min_y = (np.min(keypoints, axis=0)[1])
    max_y = (np.max(keypoints, axis=0)[1])

    upper_dist = min_y - margin_px
    lower_dist = height - max_y - margin_px
    left_dist = min_x - margin_px
    right_dist = width - max_x - margin_px

    scale_1 = width / ((2 * left_dist) + width)
    scale_2 = width / ((2 * right_dist) + width)
    scale_3 = height / ((2 * lower_dist) + height)
    scale_4 = height / ((2 * upper_dist) + height)

    return max(scale_1, scale_2, scale_3, scale_4)


def find_max_traslation(width: int, height: int, keypoints: list, affine_scale: float, margin_px=5.0) -> tuple:
    '''
    Calculates the max traslation possible to apply on the scaled image, so that all the resulting keypoints
    are inside the image, with a optional margin.

    Paramters:
    width (int): The width of the image
    height (int): The height of the image
    keypoints (list): The list of keypoints that have to be inside the image
    affine_scale (float): The scale we have to apply before the traslation
    margin_px (float): The margin from the border that the resulting keypoints can have at max

    Returns:
    tuple: The calculated max traslation in the form (max_trasl_left, max_trasl_right), (max_trasl_up, max_trasl_down)
    '''
    min_x = np.min(keypoints, axis=0)[0] # the most left x
    max_x = np.max(keypoints, axis=0)[0] # the most right x
    min_y = np.min(keypoints, axis=0)[1] # the highest y
    max_y = np.max(keypoints, axis=0)[1] # the lowest y

    half_w = width / 2.0 
    half_h = height / 2.0

    space_w = abs((width - (width * affine_scale)) / 2)
    space_h = abs((height - (height * affine_scale)) / 2)

    # new position = scale * (old position - center) + center
    new_min_x = affine_scale * (min_x - half_w) + half_w
    new_max_x = affine_scale * (max_x - half_w) + half_w
    new_min_y = affine_scale * (min_y - half_h) + half_h
    new_max_y = affine_scale * (max_y - half_h) + half_h

    if affine_scale >= 1.0:
        
        # the max traslation towards left is given already by new_min_x
        # the max traslation towards up is given already by new_min_y
        # for the ones towards right and down we simply do:
        #min_trasl_x = min(new_min_x, 2 * (width - new_max_x))
        #min_trasl_y = min(new_min_y, 2 * (height - new_max_y))
        #max_trasl_x = min(width - new_max_x, 2 * new_min_x)
        #max_trasl_y = min(height - new_max_y, 2 * new_min_y)

        min_trasl_x = min(new_min_x, width - new_max_x +  (2 * space_w))
        min_trasl_y = min(new_min_y, height - new_max_y + (2 * space_h))
        max_trasl_x = min(width - new_max_x, new_min_x + space_w)
        max_trasl_y = min(height - new_max_y, new_min_y + space_h)

        return (-min_trasl_x + margin_px, max_trasl_x - margin_px), (-min_trasl_y + margin_px, max_trasl_y - margin_px)
    else:
        max_trasl_x = min(new_min_x - (2.0 * space_w), width - new_max_x)
        max_trasl_y = min(new_min_y - (2.0 * space_h), height - new_max_y)
        min_trasl_x = min((space_w + (width * affine_scale) - new_max_x) - space_w, new_min_x)
        min_trasl_y = min((space_h + (height * affine_scale) - new_max_y) - space_h, new_min_y)

        return (-min_trasl_x + margin_px, max_trasl_x - margin_px), (-min_trasl_y + margin_px, max_trasl_y - margin_px)


def clean_labels(width: int, height: int, bboxes: list, keypoints: list) -> tuple:
    '''
    Cleans the labels produced by albumentations.
    When we set remove_invisible = False and use border_mode = cv2.BORDER_REFLECT
    for every reflected side or angle there is a set keypoints, even if they are outside the image.
    In the worst case scenario, where every reflection is present (for example: image scaled down in the center)
    we will have len(augmentations["keypoints"]) = number of expected keypoints * 9
    (9, because the sum of the angles and sides + center is 9)
    We have to check in those sets of keypoints to see which one is the central one, if there is,
    and we have to check that there won't be any other foosball table in the image, because we only want one.

    Parameters:
    width (int): The width of the augmented image
    height (int): The height of the augmented image
    bboxes (list): The bboxes produced by albumentations
    keypoints (list): The keypoints produced by albumentations

    Returns:
    tuple: The cleaned bboxes, keypoints and the result of the cleaning, because there might be more
           foosball tables in the image, or no one (result = false)
    '''

    keypoint_sets = []
    valid_sets = []

    for i in range(0, len(keypoints), 8):
        keypoint_sets.append(keypoints[i:i+8])
        valid_sets.append(False)  # as of right know we don't know if it's valid
    
    # check every keypoint set
    for i in range(len(keypoint_sets)):
        kp_set = keypoint_sets[i]
        count_valid_kp = 0

        for j in range(0, 4):  # check the first 4 keypoints
            if 0 <= kp_set[j][0] < width and 0 <= kp_set[j][1] < height:
                # the keypoint is valid
                count_valid_kp += 1

        # check results...
        if count_valid_kp == 4:
            # we have a valid set, with the first 4 keypoints valid
            valid_sets[i] = True
        elif count_valid_kp == 0:
            # no one of the first 4 keypoints are inside the image
            # BUT, THAT DOESN'T MEAN THE REFLECTED FOOSBALL TABLE IS NOT VISIBLE
            # there might be the 2 of the first 4 keypoints in the angle outside the image, but the line
            # that connects them fall inside the image, so the reflected foosball table is visible
            for j in range(0, 4):  # check the first 4 keypoints
                # check the line j, j + 1
                next_index = (j + 1) % 4

                x1, y1, _ = kp_set[j]
                x2, y2, _ = kp_set[next_index]

                # linear interpolation
                N_STEPS = 20  # number of steps
                t_values = np.linspace(0, 1, N_STEPS + 2)  # + 2 to include (x1, y1) (x2, y2)
                points = [(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in t_values]

                for point in points:
                    if 0 <= point[0] < width and 0 <= point[1] < height:
                        # a point of the line is inside the image
                        # the reflected partial foosball table is visible
                        return [], [], False
            
            # if we arrive here not partial foosball table is visible
            valid_sets[i] = False
        else:
            # we have a set that has some of its first 4 keypoints valid, but not all of them
            # certainly, we have a partial reflected foosball table in the image
            return [], [], False
    
    if sum(valid_sets) > 1:
        # it can happen in images where the foosball table is small to appear completly in the reflection
        print("Error: more valid keypoint sets (complete foosball table clone in the reflection)")
        return [], [], False
    if sum(valid_sets) == 0:
        # there isn't a single set that have all the first 4 keypoints inside the image
        # it shouldn't happen
        print("Error: no valid keypoint sets")
        return [], [], False
    
    # else, we know which one of the keypoint sets is the central one
    valid_index = valid_sets.index(1)
    fixed_kps = keypoint_sets[valid_index]
        
    # let's fix the 4 last keypoints
    # since albumentations doesn't change the visibility
    for i in range(4, 8):
        x, y, _ = fixed_kps[i]
        if x < 0 or x >= width or y < 0 or y >= height:
            fixed_kps[i] = (0.0, 0.0, 0)
    
    if len(fixed_kps) != 8:
        # there is some bad error, but it shouldn't happen
        print("Bad Error: len(fixed_kps) != 8")
        return [], [], False
    
    # ----------------
    # let's clean bboxes
    if len(bboxes) > 1:
        # there are more bboxes, we take only the one that contain all the fixed_kps
        fixed_bbox = []
        for bb in bboxes:
            n_kps_inside = 0
            for x, y, v in fixed_kps:
                if v == 0:
                    n_kps_inside += 1  # we consider it inside the bboxes
                    continue
                # if the distance between the visible kp and the center of the bb is greater than its dimension, the bb is not valid
                if (abs(x - (width * bb[0])) < (bb[2] * width) / 2.0) and (abs(y - (height * bb[1])) < (bb[3] * height) / 2.0):
                    n_kps_inside += 1

            if n_kps_inside == 8:
                # the bb contains all the keypoints, it's a valid bb
                fixed_bbox.append(bb)

        if len(fixed_bbox) != 1:
            # there is some error
            print("Error: unable to clean bboxes: len(fixed_bbox) != 1")
            return [], [], False
        else:
            return fixed_bbox, fixed_kps, True
    else:
        # the bounding box is already valid
        return bboxes, fixed_kps, True


def crop_decision(width: int, height: int, keypoints: list, offset: int = 5) -> tuple:
    '''
    It calculates a random crop as x_min, x_max, y_min, y_max for the Crop
    tranformation, that won't cause the first 4 keypoints to be outside
    the image.

    Parameters:
    width (int): The width of the image
    height (int): The height of the image
    keypoints (list): The first 4 keypoints
    offset (int): The offset the crop should have from a keypoint
    
    Returns:
    tuple: The calculated crop as x_min, x_max, y_min, y_max
    '''

    min_x_kp = round(np.min(keypoints, axis=0)[0]) # the most left x
    max_x_kp = round(np.max(keypoints, axis=0)[0]) # the most right x
    min_y_kp = round(np.min(keypoints, axis=0)[1]) # the highest y
    max_y_kp = round(np.max(keypoints, axis=0)[1]) # the lowest y

    if min_x_kp - offset < 0:
        x_min = 0
    else:
        x_min = random.randint(0, min_x_kp - offset)

    if max_x_kp + offset > width - 1:
        x_max = width - 1
    else:
        x_max = random.randint(max_x_kp + offset, width - 1)
    
    if min_y_kp - offset < 0:
        y_min = 0
    else:
        y_min = random.randint(0, min_y_kp - offset)

    if max_y_kp + offset > height - 1:
        y_max = height - 1
    else:
        y_max = random.randint(max_y_kp + offset, height - 1)            
    
    return x_min, x_max, y_min, y_max

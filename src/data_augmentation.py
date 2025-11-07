
# python file for data augmentation

# by looking at the data and labels profiling, we see that we need more:
# - images with an angle between [0, 80], and [125, 175]
# - images with a center that is not close to (0.53, 0.68)
# - images that are squares and vertical rectangles
# - images with low resolution
# - darker images
# - satureted images
# - images where some keypoints of the upper rectangle have a visibility of 1
# - images where some keypoints of the lower rectangle have a visibility of 0 and 2
# - images where the bounding box covers less that 20%, above 85%, around 37% and 70% of the image

# by looking at the clusters, we see that we need more:
# - weirdly rotated images
# - high contrast images
# - vertical rotated images

# by looking directly at the images, we see that we need more:
# - images where there are some obstacles that cover the play area

import os
import random
from typing import NamedTuple, List, Tuple
import albumentations as A 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utility import *
from config import *



# LOADING AND SAVING

def labels_loading(image_name: str, width: int=0, height: int=0, denormalize: str="none") -> tuple:
    '''
    Loads the labels from an image.
    If denormalize="none", the bounding box and the keypoints will stay normalized, according to the COCO Keypoints 1.0 format.
    If denormalize != "none", the function needs width and height to denormalize either the bounding box, the keypoints or both.
    denormalize can be {none, bbox, keypoints, both}

    Parameters:
    image_names (str): The image name without extension
    width (int): The width of the image, needed if we want to denormalize
    height (int): The height of the image, needed if we want to denormalize
    denormalize (str): Specifies if the function has to denormalize the labels. It can be "none" if no denormalization is needed,
                       "bbox" if only the bbox has to be denormalized, "keypoints" if only the keypoints have to be denormalized,
                        or "both" if both need to be denormalized

    Returns:
    tuple: The labels in the form bbox, keypoints
    '''
    if denormalize not in ["none", "bbox", "keypoints", "both"]:
        raise ValueError(f"Error: not valid 'denormalize': {denormalize} not in ['none', 'bbox', 'keypoints', 'both']")
    
    if denormalize != "none" and (width <= 0 or height <= 0):
        raise ValueError(f"Error: not valid 'height', 'width', can't denormalize")
    
    bbox_mul_x = 1.0
    bbox_mul_y = 1.0
    kps_mul_x = 1.0
    kps_mul_y = 1.0
    
    if denormalize == "bbox":
        bbox_mul_x = width
        bbox_mul_y = height
    if denormalize == "keypoints":
        kps_mul_x = width
        kps_mul_y = height
    if denormalize == "both":
        bbox_mul_x = width
        bbox_mul_y = height
        kps_mul_x = width
        kps_mul_y = height     

    # we need to look both at LABELS_DIRECTORY and ADDED_LABELS_DIRECTORY
    try:
        with open(os.path.join(LABELS_DIRECTORY, image_name + LABELS_EXTENSION), "r") as f:
            labels = f.read().strip().split()
    except Exception:
        try:
            with open(os.path.join(ADDED_LABELS_DIRECTORY, image_name + LABELS_EXTENSION), "r") as f:
                labels = f.read().strip().split()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Label {image_name + LABELS_EXTENSION} not found in {LABELS_DIRECTORY} or {ADDED_LABELS_DIRECTORY}"
            )
        
    # (0 for the class, 4 numbers for the bbox and the rest for the keypoints)
    bbox = [float(labels[1]) * bbox_mul_x, float(labels[2]) * bbox_mul_y, float(labels[3]) * bbox_mul_x, float(labels[4]) * bbox_mul_y]
    kps = labels[5:]
    keypoints = [(float(kps[i]) * kps_mul_x, float(kps[i+1]) * kps_mul_y, int(kps[i+2])) for i in range(0, len(kps), 3)]

    return bbox, keypoints



def save_augmented_data(image_name: str, augmented_image: np.ndarray, augmented_bbox: list, augmented_kps: list) -> None:

    # filenames and paths
    augmented_image_name = get_augmented_image_name(image_name)
    augmented_labels_name = os.path.splitext(augmented_image_name)[0] + LABELS_EXTENSION

    augmented_labels_path = os.path.join(AUGMENTED_LABELS_DIRECTORY, augmented_labels_name)
    augmented_image_path = os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, get_augmented_image_name(image_name))

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
            print(f"WARNING: not valid augmented keypoints for {image_name}")
    
    h, w = augmented_image.shape[:2]  # we need to normalize the keypoints

    # save the labels
    with open(augmented_labels_path, "w") as f:
        f.write("0 ")
        for bbox_number in augmented_bbox:
            f.write(str(bbox_number) + " ")
        for kp in augmented_kps:
            # normalize the keypoits
            x = "0" if kp[0] == 0.0 else str(kp[0] / float(w))
            y = "0" if kp[1] == 0.0 else str(kp[1] / float(h))
            v = str(int(kp[2]))
            f.write(f"{x} {y} {v} ")

    # save the image
    cv2.imwrite(augmented_image_path, augmented_image)
    print(f"Saved augmented image in {augmented_image_path}")

# =============================================================================

# UTILITY FUNCTIONS

def distribute_evenly(A: int, B: int) -> list:
    '''
    Creates a list of uniform A non negative integers whose sum is B.

    Parameters:
    A (int): The length of the list to generate
    B (int): The valuse that has to be the sum of the list
    
    Returns:
        list: The list of uniform values
    '''
    if B < A:
        raise ValueError("B has to be >= A")

    base = B // A
    remainder = B % A

    return [base + 1] * remainder + [base] * (A - remainder)


def max_centered_scale(width: int, height: int, keypoints: list, margin_px: float = 0.0, max_scale: float = 1.2) -> float:
    '''
    Calculates the max scale possible to apply in the center of the image, so that all the resulting keypoints are inside the image, with a optional margin.
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

        
def find_max_traslation(width: int, height: int, keypoints: list, affine_scale: float, margin_px=5.0) -> tuple:
    '''
    Calculates the max traslation possible to apply on the scaled image, so that all the resulting keypoints are inside the image, with a optional margin.

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

    # new position = scale * (old position - center) + center
    new_min_x = affine_scale * (min_x - half_w) + half_w
    new_max_x = affine_scale * (max_x - half_w) + half_w
    new_min_y = affine_scale * (min_y - half_h) + half_h
    new_max_y = affine_scale * (max_y - half_h) + half_h

    # the max traslation towards left is given already by new_min_x
    # the max traslation towards up is given already by new_min_y
    # for the ones towards right and down we simply do:
    max_trasl_x = width - new_max_x
    max_trasl_y = height - new_max_y

    return (-new_min_x + margin_px, max_trasl_x - margin_px), (-new_min_y + margin_px, max_trasl_y - margin_px)


def clean_keypoints(width: int, height: int, keypoints: list) -> tuple:
    '''
    '''

    # keypoints is an array "logically" divided in pieces of 8
    # it might be difficult to find the central piece, and so we look at each piece
    # if a piece have at least one of the first 4 keypoints outside the image, it's not the central
    # the central can also have one of the first 4 keypoints outside the image, in that case no piece have
    # all the 4 first keypoints inside the image and result is false

    valid_pieces = 0
    valid_piece_index = -1

    for i in range(0, len(keypoints), 8):
        count_valid_kp = 0

        for j in range(0, 4):
            if 0 <= keypoints[i + j][0] < width and 0 <= keypoints[i + j][1] < height:
                # the keypoint is valid
                count_valid_kp += 1
            else:
                # the keypoint is NOT valid, we can already break, because the piece won't be valid
                break

        if count_valid_kp == 4:
            # we have a valid piece, with the first 4 keypoints valid
            valid_pieces += 1
            valid_piece_index = i

    if valid_pieces == 0:
        # there are some problems for the central piece, that we can solve by chaingin the angle
        return [], False
    elif valid_pieces > 1:
        # It can happen in images where the foosball table is small to appear completly in the reflection
        print("Error: more valid_pieces > 1 (complete foosball table clone in the reflection)")
        return [], False
        
    # we might already fix the 4 last keypoints
    # if they are outside the image they should be 0, 0, 0

    fixed = keypoints[valid_piece_index : valid_piece_index + 4]
    for x, y, v in keypoints[valid_piece_index + 4 : valid_piece_index + 8]:
        if x < 0 or x >= width or y < 0 or y >= height:
            v = 0   # mark as not visible
            x = 0.0
            y = 0.0
        fixed.append((x, y, v))

    return fixed, True


def get_n_images_to_remove_mask(n_images_to_remove: int, all_cluster_counts: list) -> list:
    '''
    '''
     # the imbalance_mask contains a weight for each cluster, that says how important to remove its images
    # (0.0 do not remove images, 1.0 remove a lot of images)
    imbalance_mask = []
    for cluster_row in all_cluster_counts:
        k = len(cluster_row)
        N = sum(cluster_row)
        mean_val = N / k
        
        # simil normalization to TwoSlopeNorm
        aux = []
        for value in cluster_row:
            if value > mean_val:
                norm_val = (value - mean_val) / mean_val   # above the average
            else:
                norm_val = 0.0
            aux.append(norm_val)
        imbalance_mask.append(aux)

    weight_sum = 0.0
    for row in imbalance_mask:
        weight_sum += sum(row)

    n_images_to_remove_mask = []

    for row in imbalance_mask:
        aux = []
        for element in row:
            if element > 0:
                aux.append(round(element * (n_images_to_remove / weight_sum)))
            else:
                aux.append(0)
        n_images_to_remove_mask.append(aux)
    
    return n_images_to_remove_mask


# =============================================================================

# TRANSFORMATIONS

def get_clustering_transformation(
        affine_scale: float,
        x_trasl: int,
        y_trasl: int,
        angle_limit: tuple[float, float],
        new_width: int,
        new_height: int
    ) -> A.Compose:
    '''
    Calculates the transformation for the clustering data augmentation, for an image.

    Parameters:
    affine_scale (float): The scale to apply
    x_trasl (int): The translation along the x axis to apply
    y_trasl (int): The translation along the y axis to apply
    angle_limit (tuple[float, float]): The angle interval where the angle to apply will be chosen 
    new_width (int): The new width after resizing
    new_height (int): The new height after resizing

    Returns:
    A.Compose: The calculated transformation
    '''
    # to balance the cluster we don't want to change drastically the image (the image has to end up in the same cluster)
    transform = A.Compose(
                [
                    A.Affine(
                        scale=(1.0, affine_scale),
                        translate_px={"x": x_trasl, "y": y_trasl},
                        rotate=angle_limit,
                        shear=0,
                        fit_output=False,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.Resize(height = new_height, width= new_width),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-5.0, 5.0), g_shift_limit=(-5.0, 5.0), b_shift_limit=(-5.0, 5.0)),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.7),
                    A.OneOf([
                        A.Blur(blur_limit=5, p=1.0),
                        A.GaussianBlur(sigma_limit=(0.1, 0.7), p=1.0)
                    ], p=0.5)
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format="yolo", label_fields=[])
            )
    
    return transform

# =============================================================================

# DATA AUGMENTATION

def cluster_data_augmentation(image_name: str, n_images_to_generate: int) -> None:
    '''
    Applies data augmentation to generate n_images_to_generate augmented images, from a specific image to balance a cluster.
    The results are saved in the augmented-data folder.

    Parameters:
    image_name (str): The name of the image where data augmentation will be applied
    n_images_to_generate (int): The amount of new augmented images to generate

    Returns:
    None
    '''
    # path loading
    image_path = find_image_path(image_name, IMAGES_DATA_DIRECTORY, ADDED_IMAGES_DATA_DIRECTORY)
    if image_path == None:
        raise FileNotFoundError(
            f"Image {image_name} not found in {IMAGES_DATA_DIRECTORY} or {ADDED_IMAGES_DATA_DIRECTORY}"
        )
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # labels loading
    # Albumentations can work with normalized boundinb boxes, but not with normalized keypoints, so we have to denormalize them
    bbox, keypoints = labels_loading(image_name, w, h, denormalize="keypoints")
    

    # when applying a geometric transform (scale, traslation, rotation), we can cause some of the keypoints to be cutted out from the image
    # but we don't want the first 4 keypoints to be cutted out
    # for this reason we will find the precise max scale and max traslation that won't cause this behaviour
    # and iteratively we will change the angle of the rotation, until all the first 4 keypoints are present in the image
    max_affine_scale = max_centered_scale(w, h, keypoints[0:4], margin_px=10.0, max_scale=1.1)
    affine_scale = random.uniform(1.0, max_affine_scale)

    (min_x_trasl, max_x_trasl), (min_y_trasl, max_y_trasl) = find_max_traslation(w, h, keypoints[0:4], affine_scale, margin_px=10.0)
    x_trasl = random.uniform(min_x_trasl/3.0, max_x_trasl/3.0)
    y_trasl = random.uniform(min_y_trasl/3.0, max_y_trasl/3.0)

    angle_limit = (-6.0, 6.0)
    ANGLE_STEP = 0.5

    # let's find a new width and height: let's say if we have a resolution of 2.000.000 pixel
    # we can have a max increment or decrement of 100 (because we can't increment or decrement of 100 in low resolution images)
    increment = int((100 * h * w) / 2000000)
    new_width = w + random.randrange(-increment, increment)
    new_height = int((new_width * h) / w)  # keep the original aspect ratio
    
    # with distribution we know how many times we need to apply data augmentation for this image
    iteration = 0
    while iteration < n_images_to_generate:            
        
        # we have to recalculate it because angle_limit can change iteratively
        transform = get_clustering_transformation(affine_scale, x_trasl, y_trasl, angle_limit, new_width, new_height)

        augmentations = transform(
            image=image,
            bboxes=[bbox],  # we have only one bounding box per image
            keypoints=keypoints
        )

        # WARNING: if in the transformation uses remove_invisible=False, and cv2.BORDER_REFLECT
        # we can have a max of 9 * number of expected keypoints, because we have the reflection in every angle
        # and the keypoints that should have a visibility of 0, do not have it
        # we have to do some cleaning
        cleaned_keypoints, result = clean_keypoints(new_width, new_height, augmentations["keypoints"])

        # check if the trasformation caused to have more bboxes, more than 8 keypoints or the first 4 keypoints to be cutted out from the image
        if len(augmentations["bboxes"]) > 1 or not result:
            angle_limit = (angle_limit[0] + ANGLE_STEP, angle_limit[1] - ANGLE_STEP) # reduce the angle limit interval
            if(angle_limit[0] > 0.0 or angle_limit[1] < 0.0):
                # it means there isn't a possible valid interval, and so we should go on
                # (it might happen because the reflection cause to have more keypoints)
                print(f"WARNING: unable to apply data augmentation for {image_name}")
                iteration += 1
                continue
            else:
                # redo the iteration
                continue

        save_augmented_data(image_name, augmentations["image"], augmentations["bboxes"][0], cleaned_keypoints)

        iteration += 1



# =============================================================================

# MAIN LOOP FOR CLUSTERING DATA AUGMENTATION

def clustering_data_augmentation():
    '''
    Generates new images from the default + added dataset to balance the clusters.
    For some k, it looks for the cluster with the max amount of elements, and based on that number, decided how many
    images to generate per cluster.
    It doesn't iteretate for every k, since balancing one might unbalance another.
    It uses a small tranformation, that doesn't change drastically the image, for the data augmentation because we want
    the new images to fall in the same cluster where they were generated.
    Saves the images and labels in the augmented-data folder
    '''
    clustering_data = load_clustering_data(ADDED_CLUSTERING_DIRECTORY)
    all_clustering_labels = load_all_clustering_label(ADDED_CLUSTERING_DIRECTORY)

    # we don't consider every k, because the difference from a k to the next might be to small, so we move in K_STEP
    # (balancing a K might unbalance another)
    K_STEP = 3

    # max difference, between max_count and the count of a cluster, that we need to surpass to have data augmentation
    MAX_DIFFERENCE = 20

    # max number of images generable with data augmentation from a single image
    # it's important because if we have a cluster with a few image, and a high max_count
    # we shouldn't generate to many images from the few one (we might introduce a bias)
    MAX_AUG_PER_IMAGE = 2

    # ideality is a number from 0.0 to 1.0, that limits the amount of images to generate
    # we want to limit the images generated because increasing the number of a cluster for balance, might change the balance for other clusters
    # [min_ideality, max_ideality] is the interval where we will take a random ideality, and for every k we reduce this interval
    # MIN_IDEALITY_END is the last value (for the last k) of min_ideality
    # IDEALITY_STEP is the amount used for reducing
    min_ideality = 0.6
    MIN_IDEALITY_END = 0.2
    max_ideality = 1.0
    IDEALITY_STEP = (min_ideality - MIN_IDEALITY_END) / ((MAX_N_CLUSTERS - MIN_N_CLUSTERS) / K_STEP)  # /K_STEP because we move in K_STEP

    for k in range(MAX_N_CLUSTERS, MIN_N_CLUSTERS - 1, -K_STEP):
        index = k - MIN_N_CLUSTERS  # to access all_clustering_labels we need this index
        
        cluster_counts = clustering_data[k]["cluster_counts"]
        max_count = max(cluster_counts)

        # we will use this array to limt the amount of images to generate
        ideality_array = np.random.uniform(min_ideality, max_ideality, k)

        for cluster_id in range(k):
            # let's see how big is the difference in number of images
            n_images_diff = max_count - cluster_counts[cluster_id]
            if  n_images_diff > MAX_DIFFERENCE:
                # we apply some data augmentation

                n_images_to_generate = min(cluster_counts[cluster_id] * MAX_AUG_PER_IMAGE, max_count - MAX_DIFFERENCE - cluster_counts[cluster_id])
                n_images_to_generate = round(n_images_to_generate * ideality_array[cluster_id])  # reduce the amount of images to generate
                
                # let's grab the image names that belongs in this cluster
                image_names = []
                for image, labels in all_clustering_labels.items():
                    if labels[index] == cluster_id: image_names.append(image)

                if len(image_names) > n_images_to_generate:
                    # it means that we have to grab randomly n_images_to_generate images and generate one image from each
                    np.random.shuffle(image_names)
                    image_names = image_names[0:n_images_to_generate]
                    distribution = [1] * n_images_to_generate  # distribution says how many augmented images has to generate each image
                else:
                    # it means that some images have to generate more than one image
                    distribution = distribute_evenly(len(image_names), n_images_to_generate)  # distribution says how many augmented images has to generate each image

                print(f"For k={k}, cluster_id={cluster_id}, we generate {n_images_to_generate} images")
                for i in range(len(image_names)):
                    cluster_data_augmentation(image_names[i], distribution[i])
        
        # update the ideality array
        min_ideality -= IDEALITY_STEP
        max_ideality -= IDEALITY_STEP


def remove_some_clustering_augmented_data():
    '''
    After clustering the augmented data generated for the clusters balance, Removes some augmented images to balance even more.
    '''

    clustering_data = load_clustering_data(AUGMENTED_CLUSTERING_DIRECTORY)
    all_clustering_labels = load_all_clustering_label(AUGMENTED_CLUSTERING_DIRECTORY)

    # the amount of augemented images to remove in totale
    N_IMAGES_TO_REMOVE = 500

    all_cluster_counts = []
    for i in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS):
        all_cluster_counts.append(clustering_data[i]["cluster_counts"])
    
    # n_images_to_remove_mask says for each (k, cluster_id) the amount of images to remove
    # the sum of all those images is around N_IMAGES_TO_REMOVE
    n_images_to_remove_mask = get_n_images_to_remove_mask(N_IMAGES_TO_REMOVE, all_cluster_counts)

    # let's find n_to_remove_in_cluster augmented images
    # we don't remove original images
    aug_image_prefix = get_augmented_image_name("").split("_", 1)[0]
    
    n_removed = 0  # the counter that counts the actually removed
    for k in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS):
        index = k - MIN_N_CLUSTERS

        for cluster_id in range(k):
            # the amount of images to remove for this cluster:
            n_to_remove_in_cluster = n_images_to_remove_mask[index][cluster_id]

            if n_to_remove_in_cluster == 0:
                # we don't have images to remove
                continue

            count = 0
            for image_name, labels in all_clustering_labels.items():
                # if the image belong in the cluster and is an augmented image
                if labels[index] == cluster_id and image_name.split("_", 1)[0] == aug_image_prefix:

                    image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
                    label_path = find_label_path(image_name, AUGMENTED_LABELS_DIRECTORY)

                    if image_path == None or label_path == None:
                        print(f"Warning: can't delete {image_name}: unable to find path (maybe it was already deleted)")
                        continue
                    else:
                        # we remove the image
                        if os.path.isfile(image_path) and os.path.isfile(label_path):  # just to be safe
                            os.remove(image_path)
                            os.remove(label_path)
                            count += 1
                            n_removed += 1
                            print(f"Removed the image: {image_name}")
                        else:
                            print(f"Warning: can't delete {image_name}: the image or the labels are not files")

                    # update the counter of removed images  
                    if count >= n_to_remove_in_cluster:
                        break

            if n_to_remove_in_cluster != count:
                print(f"Warning: can't delete {n_to_remove_in_cluster} in (k={k}, id={cluster_id}). There aren't enough augmented images")

    print(f"Removed {n_removed} images in total")
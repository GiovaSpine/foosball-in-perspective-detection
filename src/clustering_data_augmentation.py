import os
import random
import albumentations as A 
import numpy as np
import cv2
from utility import *
from config import *


# =============================================================================

# LOCAL UTILITY FUNCTIONS

def distribute_evenly(A: int, B: int) -> list:
    '''
    Creates a list of uniform A non negative
    integers whose sum is B.

    Parameters:
    A (int): The length of the list to generate
    B (int): The valuse that has to be the sum
             of the list
    
    Returns:
        list: The list of uniform values
    '''
    if B < A:
        raise ValueError("B has to be >= A")

    base = B // A
    remainder = B % A

    return [base + 1] * remainder + [base] * (A - remainder)


def get_n_images_to_remove_mask(n_images_to_remove: int, all_cluster_counts: list) -> list:
    '''
    Generate a mask indicating how many images to remove from each cluster.
    The function calculates an amount to remove for each cluster based on 
    the distribution of images. Clusters with more images than the average 
    will get a higher weight and thus will remove more images. The resulting 
    mask contains integers for each cluster, and the sum of all integers is 
    approximately equal to n_images_to_remove.

    Parameters
    n_images_to_remove (int): Total number of images to remove across all clusters.
    all_cluster_counts (list): Each sublist represents a cluster group, and each integer 
                                represents the count of images in that cluster.

    Returns
    list: A mask of the same shape as `all_cluster_counts`, where each element
        is the number of images to remove from the corresponding cluster.
    '''
    # check parameters
    if not isinstance(n_images_to_remove, int) or n_images_to_remove < 0:
        raise ValueError(f"n_images_to_remove must be a non-negative integer")
    
    if not all_cluster_counts or not isinstance(all_cluster_counts, list):
        raise ValueError("all_cluster_counts must be a non-empty list of lists")

    for row in all_cluster_counts:
        if not row or not isinstance(row, list):
            raise ValueError("Each element of all_cluster_counts must be a non-empty list")
        if any((not isinstance(x, int) or x < 0) for x in row):
            raise ValueError(f"All counts in all_cluster_counts must be non-negative integers")
        
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
                        scale=(affine_scale, affine_scale),
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
                    A.Spatter(intensity=(0.0, 0.075), mode="mud", p=0.5),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.7),
                    A.OneOf([
                        A.Blur(blur_limit=3, p=1.0),
                        A.GaussianBlur(sigma_limit=(0.1, 0.5), p=1.0),
                        A.MotionBlur(blur_limit=(3.0, 7.0), angle_range=(0.0, 360.0), p=1.0)
                    ], p=0.4)
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format="yolo", label_fields=[])
            )
    
    return transform

# =============================================================================

# DATA AUGMENTATION

def cluster_data_augmentation(image_name: str, n_images_to_generate: int) -> None:
    '''
    Applies data augmentation to generate n_images_to_generate augmented images, from a specific
    image to balance a cluster.
    The results are saved in the augmented-data folder.

    Parameters:
    image_name (str): The name of the image where data augmentation will be applied
    n_images_to_generate (int): The amount of new augmented images to generate

    Returns:
    None
    '''
    if not isinstance(n_images_to_generate, int):
        raise TypeError("n_images_to_generate has to be an integer")
    if n_images_to_generate < 0:
        raise ValueError("n_images_to_generate should be >= 0")
    # path loading
    image_path = find_image_path(image_name, IMAGES_DATA_DIRECTORY, ADDED_IMAGES_DATA_DIRECTORY)
    if image_path == None:
        raise FileNotFoundError(f"Image {image_name} not found in {IMAGES_DATA_DIRECTORY} or {ADDED_IMAGES_DATA_DIRECTORY}")
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # labels loading
    # Albumentations can work with normalized boundinb boxes, but not with normalized keypoints, so we have to denormalize them
    label_path = find_label_path(image_name, LABELS_DIRECTORY, ADDED_LABELS_DIRECTORY)
    if label_path == None:
        raise FileNotFoundError(f"Image {label_path} not found in {LABELS_DIRECTORY} or {ADDED_LABELS_DIRECTORY}")
    bbox, keypoints = label_loading(label_path)
    _, keypoints = denormalize(w, h, keypoints=keypoints)

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
        _, cleaned_keypoints, result = clean_labels(new_width, new_height, augmentations["bboxes"], augmentations["keypoints"])

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


def remove_some_clustering_augmented_data() -> None:
    '''
    Removes some augmented images from cluster that exceed the average they should have
    to balance the distributions even more.
    '''

    clustering_data = load_clustering_data(AUGMENTED_CLUSTERING_DIRECTORY)
    all_clustering_labels = load_all_clustering_label(AUGMENTED_CLUSTERING_DIRECTORY)

    # the amount of augemented images to remove in total
    N_IMAGES_TO_REMOVE = 1500

    all_cluster_counts = []
    for i in range(MIN_N_CLUSTERS, MAX_N_CLUSTERS):
        all_cluster_counts.append(clustering_data[i]["cluster_counts"])
    
    # n_images_to_remove_mask says for each (k, cluster_id) the amount of images to remove
    # the sum of all those images is around N_IMAGES_TO_REMOVE
    n_images_to_remove_mask = get_n_images_to_remove_mask(N_IMAGES_TO_REMOVE, all_cluster_counts)
    for i in n_images_to_remove_mask:
        print(i)
    exit(1)

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
                print(f"Warning: can't delete {n_to_remove_in_cluster} images in (k={k}, id={cluster_id}). There aren't enough augmented images")

    print(f"Removed {n_removed} images in total")

# =============================================================================

# MAIN FUNCTION THAT DECIDES THE AMOUNT OF IMAGES TO GENERATE

def clustering_data_augmentation():
    '''
    Generates new images from the default + added dataset to balance the clusters.
    For some k, it looks for the cluster with the max amount of elements, and based on that number, decided how many
    images to generate per cluster.
    It doesn't iterate for every k, since balancing one might unbalance another.
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
    
    remove_some_clustering_augmented_data()



clustering_data_augmentation()


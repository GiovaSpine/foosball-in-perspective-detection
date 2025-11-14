import os
import random
import math
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from config import *
from utility import *

# =============================================================================

# TRANSFORMATIONS

def get_ring_transformation(
        affine_scale: float,
        x_trasl: int,
        y_trasl: int,
        angle: float,
        new_width: int,
        new_height: int,
        increase_saturation: bool
    ) -> A.Compose:
    '''
    Calculates the transformation for the ring data augmentation, with the parameters for an image.

    Parameters:
    affine_scale (float): The scale to apply
    x_trasl (int): The translation along the x axis to apply
    y_trasl (int): The translation along the y axis to apply
    angle (float): The angle to apply
    new_width (int): The new width after resizing
    new_height (int): The new height after resizing
    increase_saturation (bool): Whether the transformation has to increase the saturation or not

    Returns:
    A.Compose: The calculated transformation
    '''
    if increase_saturation:
        saturation = (10.0, 50.0) 
    else:
        saturation = (0.0, 0.0)
    
    RGB_SHIFT = 10
    
    transform = A.Compose(
                [
                    A.Affine(
                        scale=(affine_scale, affine_scale),
                        translate_px={"x": x_trasl, "y": y_trasl},
                        rotate=(angle, angle),
                        shear=0,
                        fit_output=False,
                        keep_ratio = True,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.Resize(height = new_height, width= new_width, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.1), mode="mud", p=0.5),
                    A.HueSaturationValue(hue_shift_limit=(0.0, 0.0), sat_shift_limit=saturation, val_shift_limit=(0.0, 0.0), p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
                    A.OneOf([
                        A.Blur(blur_limit=5, p=1.0),
                        A.GaussianBlur(sigma_limit=(0.1, 0.7), p=1.0),
                        A.MotionBlur(blur_limit=(3.0, 15.0), angle_range=(0.0, 135.0), p=1.0)
                    ], p=0.5)
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format="yolo", label_fields=[])
            )
    
    return transform


def get_rotate_transformation(
        angle: float,
        increase_saturation: bool
    ) -> A.Compose:
    '''
    Calculates the transformation for the rotate data augmentation, with the parameters for an image.

    Parameters:
    angle (float): The angle to apply
    increase_saturation (bool): Whether the transformation has to increase the saturation or not

    Returns:
    A.Compose: The calculated transformation
    '''
    if increase_saturation:
        saturation = (10.0, 50.0) 
    else:
        saturation = (0.0, 0.0)

    RGB_SHIFT = 10
    
    transform = A.Compose(
                [
                    A.Affine(
                        scale=(1.0, 1.0),
                        translate_px={"x": 0, "y": 0},
                        rotate=(angle, angle),
                        shear=0,
                        fit_output=True,  # to cause the actual image to rotate
                        keep_ratio = True,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.1), mode="mud", p=0.5),
                    A.HueSaturationValue(hue_shift_limit=(0.0, 0.0), sat_shift_limit=saturation, val_shift_limit=(0.0, 0.0), p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
                    A.OneOf([
                        A.Blur(blur_limit=5, p=1.0),
                        A.GaussianBlur(sigma_limit=(0.1, 0.7), p=1.0),
                        A.MotionBlur(blur_limit=(3.0, 15.0), angle_range=(0.0, 135.0), p=1.0)
                    ], p=0.5)
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format="yolo", label_fields=[])
            )
    
    return transform


def get_square_transformation(
        min_side: int,
        new_side: int,
        increase_saturation: bool
    ) -> A.Compose:
    '''
    Calculates the transformation for the square data augmentation, with the parameters for an image.

    Parameters:
    min_side (int): The min between width and height of the image
    new_side (int): The new side dimension of the square, for resizing
    increase_saturation (bool): Whether the transformation has to increase the saturation or not

    Returns:
    A.Compose: The calculated transformation
    '''
    if increase_saturation:
        saturation = (10.0, 50.0) 
    else:
        saturation = (0.0, 0.0)

    RGB_SHIFT = 10
    
    transform = A.Compose(
                [
                    A.AtLeastOneBBoxRandomCrop(min_side, min_side),
                    A.Resize(new_side, new_side),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                     A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.1), mode="mud", p=0.5),
                    A.HueSaturationValue(hue_shift_limit=(0.0, 0.0), sat_shift_limit=saturation, val_shift_limit=(0.0, 0.0), p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
                    A.OneOf([
                        A.Blur(blur_limit=5, p=1.0),
                        A.GaussianBlur(sigma_limit=(0.1, 0.7), p=1.0),
                        A.MotionBlur(blur_limit=(3.0, 15.0), angle_range=(0.0, 135.0), p=1.0)
                    ], p=0.5)
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format="yolo", label_fields=[])
            )
    
    return transform

# =============================================================================

# DATA AUGMENTATION FUNCTIONS

def ring_data_augmentation(n_per_square: int, ring_x_values: list, ring_y_values: list) -> None:
    '''
    The ring data augmentation consists in traslating images such that the center of the foosball table
    falls in a square of a specif ring, that is described by ring_x_values and ring_y_values.

    Parameters:
    n_per_square (int): Number of images that each square of the ring should have
    ring_x_values (list): The x values of the centers of each square of the ring
    ring_y_values (list): The y values of the centers of each square of the ring

    Returns:
    None
    '''

    SQUARE_STEP = 0.07  # it's a constant for the centers heatmap with n=10
    
    # if we make:
    # for x in ring_x_values:
    #     for y in ring_y_values:
    # we are checking the entire area, but we need to check only the border
    squares = []

    for x in ring_x_values:
        squares.append((x, ring_y_values[0]))
    for y in ring_y_values[1:]:
        squares.append((ring_x_values[-1], y))
    for x in reversed(ring_x_values[:-1]):
        squares.append((x, ring_y_values[-1]))
    for y in ring_y_values[-2:0:-1]:
        squares.append((ring_x_values[0], y))

    image_names = []  # list where we will save the image names
    labels = []  # list where we will save the labels
    traslations = []  # the calculated traslations corresponding to the images
    scales = []  # the calculated scales corresponding to the images

    print("Selecting images for ring balacing...")
    for x, y in squares:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        # let's try to traslate those images to the target position
        n_per_square_count = 0
        for i in range(len(df_shuffled)):
              
            image_name = df_shuffled.iloc[i]['filename']

            # image loading
            image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
            if image_path == None:
                raise FileNotFoundError(f"Image {image_name} not found in {AUGMENTED_IMAGES_DATA_DIRECTORY}")
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            # labels loading
            label_path = find_label_path(image_name, AUGMENTED_LABELS_DIRECTORY)
            if label_path == None:
                raise FileNotFoundError(f"Image {label_path} not found in {AUGMENTED_LABELS_DIRECTORY}")
            bbox, keypoints = label_loading(label_path)
            _, keypoints = denormalize(w, h, keypoints=keypoints)

            # let's decide a scale
            if df_shuffled.iloc[i]['dimension'] < 30.0:
                # scale a lot less than 1.0
                min_scale = min_centered_scale(w, h, keypoints[0:4], margin_px=3.0, min_scale=0.2)
                affine_scale = random.uniform(min_scale, 1.0)
            elif df_shuffled.iloc[i]['dimension'] > 60.0:
                # scale a bit greater than 1.0
                max_scale = max_centered_scale(w, h, keypoints[0:4], margin_px=3.0, max_scale=1.3)
                affine_scale = random.uniform(1.0, max_scale)
            else:
                affine_scale = 1.0

            # the traslation to do in normalized coordinates to reach the center of the square (x, y) + RANDOM VALUE
            norm_x_trasl = x + random.uniform(-(SQUARE_STEP / 2.0), SQUARE_STEP / 2.0) - df_shuffled.iloc[i]['center'][0]
            norm_y_trasl = y + random.uniform(-(SQUARE_STEP / 2.0), SQUARE_STEP / 2.0) - df_shuffled.iloc[i]['center'][1]

            (min_x_trasl, max_x_trasl), (min_y_trasl, max_y_trasl) = find_max_traslation(w, h, keypoints[0:4], affine_scale, margin_px=4.0)

            # actual traslation to reach the square
            actual_x_trasl = norm_x_trasl * w
            actual_y_trasl = -norm_y_trasl * h  # - to convert to the image coordinate system

            if min_x_trasl <= actual_x_trasl <= max_x_trasl and min_y_trasl <= actual_y_trasl <= max_y_trasl:
                # it's possible to traslate the image, save the image name, traslation and scale
                image_names.append(image_name)
                labels.append((bbox, keypoints))
                traslations.append((actual_x_trasl, actual_y_trasl))
                scales.append(affine_scale)
                n_per_square_count += 1  # update the counter
                if n_per_square_count >= n_per_square:
                    # we reached the image goal for this square (x, y)
                    break
            
            if(i >= len(df_shuffled) - 1):
                print(f"Square ({round(x, 2)}, {round(y, 2)}): selected {n_per_square_count} images out of {n_per_square}")
        
    
    print("\nNumber of images selected:", len(image_names))
    print("Number of unique images:", len(set(image_names)), "\n")
    print("Applying data augmentation...")

    df_cutted = df[df["filename"].isin(image_names)]

    # now we can apply data augmentation
    for i in range(len(image_names)):
        image_name = image_names[i]
        image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
        if image_path == None:
            raise FileNotFoundError(f"Image {image_name} not found in {AUGMENTED_IMAGES_DATA_DIRECTORY}")
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        bbox, keypoints = labels[i]
        x_trasl, y_trasl = traslations[i]
        affine_scale = scales[i]

        df_row = df_cutted.loc[df_cutted["filename"] == image_name]

        new_width = w
        new_height = h

        # let's decide a resolution
        if df_row.iloc[0]["resolution"] == "High":
            probability = 0.7
            if random.random() < probability:
                # with a probability of 'probability', we decrease the resolution to Low
                ratio = w / h
                if max(w, h) == w:
                    new_width = random.randrange(480, 640)
                    new_height = round(new_width / ratio)
                else:
                    new_height = random.randrange(480, 640)
                    new_width = round(new_height * ratio)

        # let's decide an angle
        rotation = 0
        if df_row.iloc[0]["theta"] >= 90.0:
            probability = 0.7
            if random.random() < probability:
                # with a probability of 'probability', we change the angle toward 152.0
                angle = max(np.random.normal(loc=152.0, scale=6.5), 179.9)
                rotation =  angle - df_row.iloc[0]["theta"]
        else:
            probability = 0.85
            if random.random() < probability:
                # with a probability of 'probability', we change the angle toward 37.5
                angle = min(np.random.normal(loc=37.5, scale=6.5), 0.0)
                rotation = angle - df_row.iloc[0]["theta"]

        # let's decide a saturation
        increase_saturation = False
        if df_row.iloc[0]["saturation_mean"] > 75.0:
            probability = 0.8
            if random.random() < probability:
                # with a probability of 'probability', we increase the saturation
                increase_saturation = True

        # angles to try
        ANGLE_STEP = 20
        if rotation != 0.0:
            rotations = np.linspace(rotation, 0, ANGLE_STEP).tolist()
        else:
            rotations = [0.0]
        
        for r in rotations:
            # we have to recalculate it because angle_limit can change iteratively
            transform = get_ring_transformation(affine_scale, x_trasl, y_trasl, r, new_width, new_height, increase_saturation)
           
            augmentations = transform(
                image=image,
                bboxes=[bbox],  # we have only one bounding box per image
                keypoints=keypoints
            )

            # WARNING: if in the transformation uses remove_invisible=False, and cv2.BORDER_REFLECT
            # we can have a max of 9 * number of expected keypoints, because we have the reflection in every angle
            # we have to do some cleaning
            cleaned_bbox, cleaned_keypoints, result = clean_labels(new_width, new_height, augmentations["bboxes"], augmentations["keypoints"])

            # check the result of the cleaning
            if not result:
                # there are some problems with the kypoints or bouding boxes
                if r == 0.0:
                    # we even tried with 0.0, it means there is not a valid angle to apply
                    print(f"WARNING: unable to apply data augmentation for {image_name}")
                continue
            
            save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints)
            break


def rotate_data_augmentation(n_to_generate: int, angle_min: float, angle_max: float, offset_angle: float = 30.0) -> None:
    '''
    The rotate data augmentation consists in generating n_to_generate new image, by rotating a horizontal image,
    with a theta between [angle_min, angle_max], by an angle around 90 or -90 degrees +- random.uniform(0.0, offset_angle),
    such that we increase the number of vertical rectangle images and increase the images with a theta different from
    angles in [angle_min, angle_max].

    Parameters:
    n_to_generate (int): Number of images to generate
    angle_min (float): The min angle an image should have, as theta, to be selected
    angle_max (float): The max angle an image should have, as theta, to be selected
    offset_angle (float): Offset from which a random value, between 0.0 and offset_angle, is added to the rotation

    Returns:
    None
    '''

    print("Applying rotate data augmentation...")

    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    count = 0
    for i in range(len(df_shuffled)):
        if angle_min <= df_shuffled.iloc[i]["theta"] <= angle_max and df_shuffled.iloc[i]["shape"] == "Horizontal Rectangle":
            image_name = df_shuffled.iloc[i]["filename"]

            # image loading
            image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
            if image_path == None:
                raise FileNotFoundError(f"Image {image_name} not found in {AUGMENTED_IMAGES_DATA_DIRECTORY}")
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            # labels loading
            label_path = find_label_path(image_name, AUGMENTED_LABELS_DIRECTORY)
            if label_path == None:
                raise FileNotFoundError(f"Image {label_path} not found in {AUGMENTED_LABELS_DIRECTORY}")
            bbox, keypoints = label_loading(label_path)
            _, keypoints = denormalize(w, h, keypoints=keypoints)

            # let's decide a saturation
            increase_saturation = False
            if df_shuffled.iloc[i]["saturation_mean"] > 75.0:
                probability = 0.8
                if random.random() < probability:
                    # with a probability of 'probability', we increase the saturation
                    increase_saturation = True
            
            ANGLE_STEP = 10
            rotations = np.linspace(random.uniform(0.0, offset_angle), 0, ANGLE_STEP).tolist()
            for r in rotations:

                # we try to add an offset, to increase the flipped angle
                if abs(angle_min - 90.0) < abs(angle_max - 90.0):
                    # rotate to the right
                    angle = -90.0 + r
                else:
                    # rotate to the left
                    angle = 90.0 - r

                transform = get_rotate_transformation(angle, increase_saturation)

                augmentations = transform(
                    image=image,
                    bboxes=[bbox],  # we have only one bounding box per image
                    keypoints=keypoints
                )
                new_height, new_width = augmentations["image"].shape[:2]
                
                # WARNING: if in the transformation uses remove_invisible=False, and cv2.BORDER_REFLECT
                # we can have a max of 9 * number of expected keypoints, because we have the reflection in every angle
                # and the keypoints that should have a visibility of 0, do not have it
                # we have to do some cleaning
                cleaned_bbox, cleaned_keypoints, result = clean_labels(new_width, new_height, augmentations["bboxes"], augmentations["keypoints"])

                # check the result of the cleaning
                if not result:
                    # there are some problems with the kypoints or bouding boxes
                    if r == 0.0:
                        # we even tried with 0.0, it means there is not a valid angle to apply
                        print(f"WARNING: unable to apply data augmentation for {image_name}")
                    continue

                if save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints):
                    count += 1
                break

            
            if count >= n_to_generate:
                break

    if count != n_to_generate:
        print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")
    else:
        print(f"{n_to_generate} images generated")


def theta_data_augmentation(n_to_generate: int, angle_interval: tuple[float, float], desired_angle: float, angle_offset: float = 10.0) -> None:
    '''
    The theta data augmentations consists in generating n_to_generate new images by taking images in a certain angle_interval,
    and applying a rotation such that the resulting theta is around desired_angle plus a random value, that is between -angle_offset
    and angle_offset

    Parameters:
    n_to_generate (int): The number of images to generate
    angle_interval (tuple[float, float]): The angle interval from which images are selected for augmentation
    desired_angle (float): The desired angle the image should have as theta
    angle_offset (float): The value from which a random value will be taken to add it to the desired_angle
    
    Returns:
    None
    '''

    print("Applying theta data augmentation...")

    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    count = 0
    for i in range(len(df_shuffled)):

        if angle_interval[0] <= df_shuffled.iloc[i]["theta"] <= angle_interval[1]:
            image_name = df_shuffled.iloc[i]["filename"]

            # image loading
            image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
            if image_path == None:
                raise FileNotFoundError(f"Image {image_name} not found in {AUGMENTED_IMAGES_DATA_DIRECTORY}")
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            # labels loading
            label_path = find_label_path(image_name, AUGMENTED_LABELS_DIRECTORY)
            if label_path == None:
                raise FileNotFoundError(f"Image {label_path} not found in {AUGMENTED_LABELS_DIRECTORY}")
            bbox, keypoints = label_loading(label_path)
            _, keypoints = denormalize(w, h, keypoints=keypoints)

            new_width = w
            new_height = h

            # let's decide a resolution
            if df_shuffled.iloc[i]["resolution"] == "High":
                probability = 0.55
                if random.random() < probability:
                    # with a probability of 'probability', we decrease the resolution to Low
                    ratio = w / h
                    if max(w, h) == w:
                        new_width = random.randrange(480, 640)
                        new_height = round(new_width / ratio)
                    else:
                        new_height = random.randrange(480, 640)
                        new_width = round(new_height * ratio)

            # let's decide a saturation
            increase_saturation = False
            if df_shuffled.iloc[i]["saturation_mean"] > 75.0:
                probability = 0.8
                if random.random() < probability:
                    # with a probability of 'probability', we increase the saturation
                    increase_saturation = True
                
            theta = df_shuffled.iloc[i]["theta"]

            # calculate the rotation to do
            angle = np.random.normal(loc=desired_angle, scale=6.5) + random.uniform(-angle_offset, angle_offset)
            rotation = angle - theta

            ANGLE_STEP = 10
            rotations = np.linspace(rotation, 0.0, ANGLE_STEP).tolist()
            for r in rotations:
                transform = get_ring_transformation(1.0, 0.0, 0.0, r, new_width, new_height, increase_saturation)

                augmentations = transform(
                    image=image,
                    bboxes=[bbox],  # we have only one bounding box per image
                    keypoints=keypoints
                )
                new_height, new_width = augmentations["image"].shape[:2]
                
                # WARNING: if in the transformation uses remove_invisible=False, and cv2.BORDER_REFLECT
                # we can have a max of 9 * number of expected keypoints, because we have the reflection in every angle
                # we have to do some cleaning
                cleaned_bbox, cleaned_keypoints, result = clean_labels(new_width, new_height, augmentations["bboxes"], augmentations["keypoints"])

                # check the result of the cleaning
                if not result:
                    # there are some problems with the kypoints or bouding boxes
                    if r == 0.0:
                        # we even tried with 0.0, it means there is not a valid angle to apply
                        print(f"WARNING: unable to apply data augmentation for {image_name}")
                    continue
                
                if save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints):
                    count += 1
                break

            if count >= n_to_generate:
                break

    if count != n_to_generate:
        print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")
    else:
        print(f"{n_to_generate} images generated")


def square_data_augmentation(n_to_generate: int, original_shape: str):
    '''
    The square data augmentation consists in cropping images that have
    a shape equal to original_shape, to make an image that is a square.

    Parameters:
    n_to_generate (int): The number of images to generate
    original_shape (str): The shape that a image has to have to be
                          selected for data augmentation
    
    Returns:
    None
    '''

    print("Applying square data augmentation...")

    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    count = 0
    for i in range(len(df_shuffled)):
        
        if df_shuffled.iloc[i]["shape"] == original_shape:

            image_name = df_shuffled.iloc[i]["filename"]

            # image loading
            image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
            if image_path == None:
                raise FileNotFoundError(f"Image {image_name} not found in {AUGMENTED_IMAGES_DATA_DIRECTORY}")
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            # labels loading
            label_path = find_label_path(image_name, AUGMENTED_LABELS_DIRECTORY)
            if label_path == None:
                raise FileNotFoundError(f"Image {label_path} not found in {AUGMENTED_LABELS_DIRECTORY}")
            bbox, keypoints = label_loading(label_path)
            _, keypoints = denormalize(w, h, keypoints=keypoints)

            # let's decide a resolution
            new_side = min(w, h)
            if df_shuffled.iloc[i]["resolution"] == "High":
                probability = 0.55
                if random.random() < probability:
                    # with a probability of 'probability', we decrease the resolution to Low
                    new_side = random.randrange(480, 640)
            
            # let's decide a saturation
            increase_saturation = False
            if df_shuffled.iloc[i]["saturation_mean"] > 75.0:
                probability = 0.8
                if random.random() < probability:
                    # with a probability of 'probability', we increase the saturation
                    increase_saturation = True

            min_side = min(w, h)

            # WARNING: min_side has to be greater that the max side of the bounding box
            # otherwise the image can't be fully in a square
            if min_side < max(bbox[2] * w, bbox[3] * h):
                # unable to crop the image without distorting the bbox
                continue
            
            transform = get_square_transformation(min_side, new_side, increase_saturation)

            augmentations = transform(
                image=image,
                bboxes=[bbox],  # we have only one bounding box per image
                keypoints=keypoints
            )
            new_height, new_width = augmentations["image"].shape[:2]
            
            # WARNING: if in the transformation uses remove_invisible=False, and cv2.BORDER_REFLECT
            # we can have a max of 9 * number of expected keypoints, because we have the reflection in every angle
            # we have to do some cleaning
            cleaned_bbox, cleaned_keypoints, result = clean_labels(new_width, new_height, augmentations["bboxes"], augmentations["keypoints"])

            # check the result of the cleaning
            if not result:
                # there are some problems with the kypoints or bouding boxes
                print(f"WARNING: unable to apply data augmentation for {image_name}")
                continue
            
            if save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints):
                count += 1
            
            if count >= n_to_generate:
                break
        
    if count != n_to_generate:
        print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")
    else:
        print(f"{n_to_generate} images generated")


# =============================================================================

def dataframes_data_augmentation():
    '''
    '''

    images_df = pd.read_parquet(AUGMENTED_IMAGES_DATAFRAME_DIRECTORY)
    labels_df = pd.read_parquet(AUGMENTED_LABELS_DATAFRAME_DIRECTORY)
    # remove extension from filename
    images_df['filename'] = images_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: os.path.splitext(x)[0])
    # combine the 2 dataframe
    global df
    df = images_df.merge(labels_df, on="filename", how="inner")

    # let's already add theta
    theta = [math.degrees(math.atan2(y,x)) for x, y in df["direction"]]
    df = df.assign(theta = theta)

    # second ring of the centers heatmap
    # WARNING, THE VALUES OF THE HEATMAP GRAPH ARE IN ANOTHER COORDINATE SYSTEM
    second_ring_x_values = np.linspace(0.25, 0.74, 8)
    second_ring_y_values = np.linspace(0.86, 0.36, 8)
    print("Second ring data augmentation...")
    ring_data_augmentation(n_per_square=20, ring_x_values=second_ring_x_values, ring_y_values=second_ring_y_values)

    # third ring of the centers heatmap
    # WARNING, THE VALUES OF THE HEATMAP GRAPH ARE IN ANOTHER COORDINATE SYSTEM
    third_ring_x_values = np.linspace(0.31, 0.67, 6)
    third_ring_y_values = np.linspace(0.79, 0.44, 6)
    print("Third ring data augmentation...")
    ring_data_augmentation(n_per_square=50, ring_x_values=third_ring_x_values, ring_y_values=third_ring_y_values)


    rotate_data_augmentation(n_to_generate=320, angle_min=91.0, angle_max=130.0, offset_angle=50.0)
    rotate_data_augmentation(n_to_generate=340, angle_min=48.0, angle_max=90.0, offset_angle=50.0)


    theta_data_augmentation(280, (0.0, 75.0), 37.0)
    theta_data_augmentation(280, (100.0, 180.0), 150.0)

    square_data_augmentation(500, "Horizontal Rectangle")
    square_data_augmentation(300, "Vertical Rectangle")


dataframes_data_augmentation()




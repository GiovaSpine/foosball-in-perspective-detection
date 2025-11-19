import os
import glob
import random
import math
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from config import *
from utility import *



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
        saturation = (10.0, 20.0) 
    else:
        saturation = (-10.0, 10.0)
    
    RGB_SHIFT = 10
    HUE_SHIFT = (-10.0, 10.0)
    VAL_SHIFT = (-10.0, 10.0)
    
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
                    A.Spatter(intensity=(0.0, 0.075), mode="mud", p=0.45),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=saturation, val_shift_limit=VAL_SHIFT, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=(-60.0, -40.0), val_shift_limit=VAL_SHIFT, p=1.0),
                    ], p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
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


def get_rotate_transformation(
        angle: float,
        increase_saturation: bool,
        new_width: int,
        new_height: int,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int
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
        saturation = (10.0, 20.0) 
    else:
        saturation = (-10.0, 10.0)
    
    RGB_SHIFT = 10
    HUE_SHIFT = (-10.0, 10.0)
    VAL_SHIFT = (-10.0, 10.0)
    
    transform = A.Compose(
                [
                    A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=1.0),
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
                    A.Resize(height=new_height, width=new_width, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.075), mode="mud", p=0.45),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=saturation, val_shift_limit=VAL_SHIFT, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=(-60.0, -40.0), val_shift_limit=VAL_SHIFT, p=1.0),
                    ], p=1.0),
                    A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=saturation, val_shift_limit=VAL_SHIFT, p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
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


def get_flip_transformation(
        angle: float,
        increase_saturation: bool,
        new_width: int,
        new_height: int,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int
    ):
    '''
    Calculates the transformation for the flip data augmentation, with the parameters for an image.

    Parameters:
    new_width (int): The new width after resizing
    new_height (int): The new height after resizing
    increase_saturation (bool): Whether the transformation has to increase the saturation or not

    Returns:
    A.Compose: The calculated transformation
    '''
    if increase_saturation:
        saturation = (10.0, 20.0) 
    else:
        saturation = (-10.0, 10.0)
    
    RGB_SHIFT = 10
    HUE_SHIFT = (-10.0, 10.0)
    VAL_SHIFT = (-10.0, 10.0)
    
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
                    A.HorizontalFlip(p=1.0),
                    A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=1.0),
                    A.Resize(height = new_height, width= new_width, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.075), mode="mud", p=0.45),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=saturation, val_shift_limit=VAL_SHIFT, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=(-60.0, -40.0), val_shift_limit=VAL_SHIFT, p=1.0),
                    ], p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
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
        saturation = (10.0, 20.0) 
    else:
        saturation = (-10.0, 10.0)
    
    RGB_SHIFT = 10
    HUE_SHIFT = (-10.0, 10.0)
    VAL_SHIFT = (-10.0, 10.0)
    
    transform = A.Compose(
                [
                    A.AtLeastOneBBoxRandomCrop(min_side, min_side),
                    A.Resize(new_side, new_side),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    A.RGBShift(r_shift_limit=(-RGB_SHIFT, RGB_SHIFT), g_shift_limit=(-RGB_SHIFT, RGB_SHIFT), b_shift_limit=(-RGB_SHIFT, RGB_SHIFT), p=1.0),
                    A.Spatter(intensity=(0.0, 0.075), mode="mud", p=0.45),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=saturation, val_shift_limit=VAL_SHIFT, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=HUE_SHIFT, sat_shift_limit=(-60.0, -40.0), val_shift_limit=VAL_SHIFT, p=1.0),
                    ], p=1.0),
                    A.ISONoise(intensity=(0.0, 0.1), p=0.6),
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

            min_scale = min_centered_scale(w, h, keypoints[0:4], margin_px=3.0, min_scale=0.2)
            max_scale = max_centered_scale(w, h, keypoints[0:4], margin_px=3.0, max_scale=1.3)

            SCALE_STEP = 10
            scales = np.linspace(min_scale, max_scale, SCALE_STEP)
            np.random.shuffle(scales)

            theta = df_shuffled.iloc[i]['theta']

            ROTATION_STEP = 50
            if theta <= 90.0:
                rotations = np.linspace(-theta, 90.0-theta, ROTATION_STEP)
            else:
                rotations = np.linspace(180.0-theta, -theta+90.0, ROTATION_STEP)
                
            SHUFFLE_PROB = 0.5
            # with a probability of 1.0 - SHUFFLE_PROB we don't shuffle the rotations array
            # so angles of 0.0 and 180.0 are privileged (note the orders on np.linspaces)
            if random.random() < SHUFFLE_PROB:
                np.random.shuffle(rotations)

            # let's try different rotation and scale until the image fit in that position
            for affine_scale in scales:

                # the traslation to do in normalized coordinates to reach the center of the square (x, y) + RANDOM VALUE
                norm_x_trasl = x + random.uniform(-(SQUARE_STEP / 2.0), SQUARE_STEP / 2.0) - df_shuffled.iloc[i]['center'][0]
                norm_y_trasl = y + random.uniform(-(SQUARE_STEP / 2.0), SQUARE_STEP / 2.0) - df_shuffled.iloc[i]['center'][1]
                (min_x_trasl, max_x_trasl), (min_y_trasl, max_y_trasl) = find_max_traslation(w, h, keypoints[0:4], affine_scale, margin_px=4.0)
                # actual traslation to reach the square
                actual_x_trasl = norm_x_trasl * w
                actual_y_trasl = -norm_y_trasl * h  # - to convert to the image coordinate system

                if not(min_x_trasl <= actual_x_trasl <= max_x_trasl and min_y_trasl <= actual_y_trasl <= max_y_trasl):
                    # unable to find a good scale
                    continue
                
                is_saved = False  # condition to stop
                for rotation in rotations:

                    # let's decide a resolution
                    new_width = w
                    new_height = h
                    if df_shuffled.iloc[i]["resolution"] == "High":
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

                    # let's decide a saturation
                    increase_saturation = False
                    if df_shuffled.iloc[i]["saturation_mean"] > 75.0:
                        probability = 0.8
                        if random.random() < probability:
                            # with a probability of 'probability', we increase the saturation
                            increase_saturation = True

                    transform = get_ring_transformation(affine_scale, actual_x_trasl, actual_y_trasl, rotation, new_width, new_height, increase_saturation)
           
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
                        # not valid rotation
                        continue
                    
                    is_saved = save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints)
                    break
                
                if is_saved:
                    n_per_square_count += 1
                    break
            if n_per_square_count >= n_per_square:
                # able to reach the goal
                print(f"Square ({round(x, 2)}, {round(y, 2)}): reached goal of {n_per_square} images")
                break
        if n_per_square_count < n_per_square:
            # unable to reach the goal
            print(f"Square ({round(x, 2)}, {round(y, 2)}): unable to reach goal: selected {n_per_square_count} images out of {n_per_square}")    


def rotate_data_augmentation(max_per_image: int, n_to_generate: int, angle_min: float, angle_max: float, offset_angle: float = 30.0) -> None:
    '''
    The rotate data augmentation consists in generating n_to_generate new image, by rotating a horizontal image,
    with a theta between [angle_min, angle_max], by an angle around 90 or -90 degrees +- random.uniform(0.0, offset_angle),
    such that we increase the number of vertical rectangle images and increase the images with a theta different from
    angles in [angle_min, angle_max].

    Parameters:
    max_per_image (int): Max amount of images a single image can generate
    n_to_generate (int): Number of images to generate
    angle_min (float): The min angle an image should have, as theta, to be selected
    angle_max (float): The max angle an image should have, as theta, to be selected
    offset_angle (float): Offset from which a random value, between 0.0 and offset_angle, is added to the rotation

    Returns:
    None
    '''

    print("Applying rotate data augmentation...")

    count = 0
    for _ in range(max_per_image):  # to not risk of chosing the same images too many times
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

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

                # let's decide a crop
                # instead of implementening some limits that prevents cropping an image into a horizontal
                # rectangle, we don't do that, because the amount of horizontal rectangles produced is limited
                # Warning: by rotating the image of anf angle around 90 or -90, width becomes height and viceversa
                crop_probability = 0.7
                if random.random() < crop_probability:
                    y_min = 0
                    y_max = h - 1
                    x_min, x_max, _, _ = crop_decision(w, h, keypoints[0:4])
                else:
                    x_min, y_min, x_max, y_max = 0, 0, w - 1, h - 1

                # warning, with the crop we have a new dimension
                w = y_max - y_min
                h = x_max - x_min

                # let's decide a resolution
                # warning, with the crop we have a new dimension
                new_width = w
                new_height = h
                if df_shuffled.iloc[i]["resolution"] == "High":
                    probability = 0.4
                    if random.random() < probability:
                        # with a probability of 'probability', we decrease the resolution to Low
                        ratio = w / h
                        if max(w, h) == w:
                            new_width = random.randrange(480, 640)
                            new_height = round(new_width / ratio)
                        else:
                            new_height = random.randrange(480, 640)
                            new_width = round(new_height * ratio)

                
                ANGLE_STEP = 15
                rotations = np.linspace(random.uniform(0.0, offset_angle), 0, ANGLE_STEP).tolist()
                np.random.shuffle(rotations)
                for r in rotations:
                    # we try to add an offset, to increase the flipped angle
                    if abs(angle_min - 90.0) < abs(angle_max - 90.0):
                        # rotate to the right
                        angle = -90.0 + r
                    else:
                        # rotate to the left
                        angle = 90.0 - r

                    transform = get_rotate_transformation(angle, increase_saturation, new_width, new_height, x_min, y_min, x_max, y_max)

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
                        if count >= n_to_generate:
                            print(f"{count} images generated")
                            return
                        break

    print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")


def theta_data_augmentation(max_per_image: int, n_to_generate: int, angle_interval: tuple[float, float], desired_angle: float, angle_offset: float = 10.0) -> None:
    '''
    The theta data augmentations consists in generating n_to_generate new images by taking images in a certain angle_interval,
    and applying a rotation such that the resulting theta is around desired_angle plus a random value, that is between -angle_offset
    and angle_offset

    Parameters:
    max_per_image (int): Max amount of images a single image can generate
    n_to_generate (int): The number of images to generate
    angle_interval (tuple[float, float]): The angle interval from which images are selected for augmentation
    desired_angle (float): The desired angle the image should have as theta
    angle_offset (float): The value from which a random value will be taken to add it to the desired_angle
    
    Returns:
    None
    '''

    print("Applying theta data augmentation...")

    count = 0
    for _ in range(max_per_image):  # to not risk of chosing the same images too many times
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        for i in range(len(df_shuffled)):
            # let's take images with a theta in the angle_interval
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

                # let's decide a crop
                # instead of implementening some limits that prevents cropping an image into a horizontal
                # rectangle, we don't do that, because the amount of horizontal rectangles produced is limited
                # Warning: by rotating the image of anf angle around 90 or -90, width becomes height and viceversa
                crop_probability = 0.7
                if random.random() < crop_probability:
                    x_min, x_max, y_min, y_max = crop_decision(w, h, keypoints[0:4], offset=50)
                else:
                    x_min, y_min, x_max, y_max = 0, 0, w - 1, h - 1

                # warning, with the crop we have a new dimension
                w = x_max - x_min
                h = y_max - y_min
                
                # let's decide a resolution
                new_width = w
                new_height = h
                if df_shuffled.iloc[i]["resolution"] == "High":
                    probability = 0.4
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

                ANGLE_STEP = 15
                rotations = np.linspace(rotation, 0.0, ANGLE_STEP).tolist()
                for r in rotations:
                    transform = get_rotate_transformation(r, increase_saturation, new_width, new_height, x_min, y_min, x_max, y_max)
                    
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
                        if count >= n_to_generate:
                            print(f"{count} images generated")
                            return
                        break

    print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")


def flip_data_augmentation(max_per_image: int, n_to_generate: int, angle_interval: tuple[float, float], angle_offset: float = 10.0) -> None:
    '''
    The flip data augmentation consists in doing a horizontal flip for all the images with a theta in the interval
    angle_interval.

    Parameters:
    max_per_image (int): Max amount of images a single image can generate
    n_to_generate (int): The amount of images to generate
    angle_interval (tuple[float, float]): The angle interval for images selection
    angle_offset (float): A value greater than 0.0 from which a random value will be chosen for a rotation
    
    Returns:
    None
    '''

    print("Applying flip data augmentation")

    count = 0
    for _ in range(max_per_image):  # to not risk of chosing the same images too many times
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        for i in range(len(df_shuffled)):
            # let's take images with a theta in the angle_interval
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

                # let's decide a crop
                # instead of implementening some limits that prevents cropping an image into a horizontal
                # rectangle, we don't do that, because the amount of horizontal rectangles produced is limited
                # Warning: by rotating the image of anf angle around 90 or -90, width becomes height and viceversa
                crop_probability = 0.7
                if random.random() < crop_probability:
                    x_min, x_max, y_min, y_max = crop_decision(w, h, keypoints[0:4], offset=50)
                else:
                    x_min, y_min, x_max, y_max = 0, 0, w - 1, h - 1

                # warning, with the crop we have a new dimension
                w = x_max - x_min
                h = y_max - y_min

                # let's decide a resolution
                new_width = w
                new_height = h
                if df_shuffled.iloc[i]["resolution"] == "High":
                    probability = 0.4
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
                
                ANGLE_STEP = 15
                if theta - angle_offset < 0:
                    rotations = np.linspace(0.0, 2 * angle_offset, ANGLE_STEP)
                elif theta + angle_offset > 180.0:
                    rotations = np.linspace(-2 * angle_offset, 0.0, ANGLE_STEP)
                else:
                    rotations = np.linspace(-angle_offset, angle_offset, ANGLE_STEP)
                np.random.shuffle(rotations)

                rotations = rotations.tolist()
                
                for r in rotations:
                    transform = get_flip_transformation(r, increase_saturation, new_width, new_height, x_min, y_min, x_max, y_max)
                            
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
                        if r == rotations[len(rotations) - 1]:
                            # we even tried the last angle
                            print(f"WARNING: unable to apply data augmentation for {image_name}")
                        continue
                    
                    if save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints):
                        count += 1
                        if count >= n_to_generate:
                            print(f"{count} images generated")
                            return
                        break

    print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")


def square_data_augmentation(max_per_image: int, n_to_generate: int, original_shape: str):
    '''
    The square data augmentation consists in cropping images that have
    a shape equal to original_shape, to make an image that is a square.

    Parameters:
    max_per_image (int): Max amount of images a single image can generate
    n_to_generate (int): The number of images to generate
    original_shape (str): The shape that a image has to have to be
                          selected for data augmentation
    
    Returns:
    None
    '''

    print("Applying square data augmentation...")

    count = 0
    for _ in range(max_per_image):

        df_shuffled = df.sample(frac=1).reset_index(drop=True)

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
                        print(f"{count} images generated")
                        return

    print(f"Unable to generate {n_to_generate}: generated {count}/{n_to_generate}")


def delete_too_augmented_images(index_threshold: int) -> None:
    '''
    All augmented images have a name in the form:
    prefix_number_name of the original image.extension.
    This function deletes all augmented images that have a number
    above index_threshold in their name.

    Parameters:
    index_threshold (int): The number from which all augmented images
                           with a number superior to this, will be
                           deleted

    Returns:
    None
    '''
    aug_image_form = get_augmented_image_name("").split("_")

    images = glob.glob(os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, f"{aug_image_form[0]}_*_*{aug_image_form[2]}"))
    labels = glob.glob(os.path.join(AUGMENTED_LABELS_DIRECTORY, f"{aug_image_form[0]}_*_*{LABELS_EXTENSION}"))
    if len(images) != len(labels):
        raise FileNotFoundError("Error: there is incongruity between the augmented images and labels")

    count = 0
    for file in images:
        number = int(os.path.basename(file).split("_")[1])
        if number > index_threshold:
            os.remove(file) 
            count += 1
    
    for file in labels:
        number = int(os.path.basename(file).split("_")[1])
        if number > index_threshold:
            os.remove(file)
       
    print(f"{count} images deleted")


# =============================================================================

def dataframes_data_augmentation():
    '''
    '''

    # it's better if we don't apply data augmention on augmented images
    # so we work on the added + default dataset
    images_df = pd.read_parquet(ADDED_IMAGES_DATAFRAME_DIRECTORY)
    labels_df = pd.read_parquet(ADDED_LABELS_DATAFRAME_DIRECTORY)

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
    #ring_data_augmentation(n_per_square=20, ring_x_values=second_ring_x_values, ring_y_values=second_ring_y_values)

    # third ring of the centers heatmap
    # WARNING, THE VALUES OF THE HEATMAP GRAPH ARE IN ANOTHER COORDINATE SYSTEM
    third_ring_x_values = np.linspace(0.31, 0.67, 6)
    third_ring_y_values = np.linspace(0.79, 0.44, 6)
    print("Third ring data augmentation...")
    ring_data_augmentation(n_per_square=50, ring_x_values=third_ring_x_values, ring_y_values=third_ring_y_values)

    rotate_data_augmentation(max_per_image=3, n_to_generate=550, angle_min=91.0, angle_max=130.0, offset_angle=50.0)
    rotate_data_augmentation(max_per_image=3, n_to_generate=550, angle_min=48.0, angle_max=90.0, offset_angle=50.0)

    theta_data_augmentation(max_per_image=3, n_to_generate=320, angle_interval=(0.0, 70.0), desired_angle=37.0)
    theta_data_augmentation(max_per_image=3, n_to_generate=350, angle_interval=(120.0, 180.0), desired_angle=150.0)

    flip_data_augmentation(max_per_image=4, n_to_generate=400, angle_interval=(0.0, 76.0))
    flip_data_augmentation(max_per_image=4, n_to_generate=500, angle_interval=(98.0, 180.0))

    square_data_augmentation(max_per_image=2, n_to_generate=900, original_shape="Horizontal Rectangle")
    square_data_augmentation(max_per_image=4, n_to_generate=600, original_shape="Vertical Rectangle")

    delete_too_augmented_images(index_threshold=9)


dataframes_data_augmentation()




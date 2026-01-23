import os
import glob
from functools import wraps
import inspect
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

# constants used by the transformations functions
RGB_SHIFT = 10
HUE_SHIFT = (-10.0, 10.0)
VAL_SHIFT = (-10.0, 10.0)


def add_saturation_param(func):
    '''
    Decorator that handles the increase_saturation parameters
    and generates the saturation level accordingly
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        increase_saturation = bound_args.arguments.get("increase_saturation", False)
        
        if increase_saturation:
            saturation = (10.0, 20.0)
        else:
            saturation = (-10.0, 10.0)
        
        # Add saturation to kwargs (or bound_args)
        kwargs["saturation"] = saturation
        
        return func(*args, **kwargs)
    
    return wrapper


@add_saturation_param
def get_ring_transformation(
        affine_scale: float,
        x_trasl: int,
        y_trasl: int,
        angle: float,
        new_width: int,
        new_height: int,
        increase_saturation: bool,
        saturation: tuple=None,
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
    saturation (tuple): The level of saturation the tranformation choose from

    Returns:
    A.Compose: The calculated transformation
    '''
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


@add_saturation_param
def get_rotate_transformation(
        angle: float,
        new_width: int,
        new_height: int,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        increase_saturation: bool,
        saturation: tuple=None,
    ) -> A.Compose:
    '''
    Calculates the transformation for the rotate data augmentation, with the parameters for an image.

    Parameters:
    angle (float): The angle to apply
    new_width (int): The new width after resizing
    new_height (int): The new height after resizing
    x_min (int): The x-coordinate of the top-left corner of the crop rectangle.
    y_min (int): The y-coordinate of the top-left corner of the crop rectangle.
    x_max (int): The x-coordinate of the bottom-right corner of the crop rectangle.
    y_max (int): The y-coordinate of the bottom-right corner of the crop rectangle.
    increase_saturation (bool): Whether the transformation has to increase the saturation or not
    saturation (tuple): The level of saturation the tranformation choose from

    Returns:
    A.Compose: The calculated transformation
    '''
    transform = A.Compose(
                [
                    A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=1.0),
                    A.Affine(
                        scale=(1.0, 1.0),
                        translate_px={"x": 0, "y": 0},
                        rotate=(angle, angle),
                        shear=0,
                        fit_output=True,
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


@add_saturation_param
def get_flip_transformation(
        angle: float,
        new_width: int,
        new_height: int,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        increase_saturation: bool,
        saturation: tuple=None,
    ):
    '''
    Calculates the transformation for the flip data augmentation, with the parameters for an image.

    Parameters:
    angle (float): The angle to apply
    new_width (int): The new width after resizing
    new_height (int): The new height after resizing
    x_min (int): The x-coordinate of the top-left corner of the crop rectangle.
    y_min (int): The y-coordinate of the top-left corner of the crop rectangle.
    x_max (int): The x-coordinate of the bottom-right corner of the crop rectangle.
    y_max (int): The y-coordinate of the bottom-right corner of the crop rectangle.
    increase_saturation (bool): Whether the transformation has to increase the saturation or not
    saturation (tuple): The level of saturation the tranformation choose from

    Returns:
    A.Compose: The calculated transformation
    '''
    transform = A.Compose(
                [   
                    A.Affine(
                        scale=(1.0, 1.0),
                        translate_px={"x": 0, "y": 0},
                        rotate=(angle, angle),
                        shear=0,
                        fit_output=True,
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


@add_saturation_param
def get_square_transformation(
        min_side: int,
        new_side: int,
        increase_saturation: bool,
        saturation: tuple=None,
    ) -> A.Compose:
    '''
    Calculates the transformation for the square data augmentation, with the parameters for an image.

    Parameters:
    min_side (int): The min between width and height of the image
    new_side (int): The new side dimension of the square, for resizing
    increase_saturation (bool): Whether the transformation has to increase the saturation or not
    saturation (tuple): The level of saturation the tranformation choose from

    Returns:
    A.Compose: The calculated transformation
    '''
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
    The centers heatmap graph is a 10x10 grid, formed by squares.
    A ring is formed by squares and has the following shape for example:
    # # # # #
    #       #
    #       #
    #       #
    # # # # #

    The ring data augmentation consists in traslating images such that the center of the foosball table
    falls in a square of a specfic ring, that is described by ring_x_values and ring_y_values.

    Parameters:
    n_per_square (int): Number of images that each square of the ring should have
    ring_x_values (list): The x values of the centers of each square of the ring
    ring_y_values (list): The y values of the centers of each square of the ring

    Returns:
    None
    '''
    # check parameters
    if not isinstance(n_per_square, int) or n_per_square <= 0:
        raise ValueError(f"n_per_square must be a positive integer")
    if not isinstance(ring_x_values, list | np.ndarray):
        raise ValueError("ring_x_values must be a list")
    if not all(isinstance(x, (float, int, np.float32)) for x in ring_x_values):
        raise ValueError("All elements in ring_x_values must be floats or ints")
    if not isinstance(ring_y_values, list | np.ndarray):
        raise ValueError("ring_y_values must be a list")
    if not all(isinstance(y, (float, int, np.float32)) for y in ring_y_values):
        raise ValueError("All elements in ring_y_values must be floats or ints")
    if len(ring_x_values) != len(ring_y_values):
        raise ValueError(f"ring_x_values and ring_y_values must have the same length")

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

    print("Applying ring data augmentation...")
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
                rotations = np.linspace(-(abs(theta - MIN_THETA)), 90.0 - theta, ROTATION_STEP)
            else:
                rotations = np.linspace(-theta + 90.0, abs(MAX_THETA - theta), ROTATION_STEP)
                
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


def theta_data_augmentation(
        max_per_image: int,
        n_to_generate: int,
        angle_interval: tuple[float, float],
        desired_angle: float, 
        angle_offset: float = 10.0
    ) -> None:
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
    # check parameters
    if not isinstance(max_per_image, int) or max_per_image <= 0:
        raise ValueError(f"max_per_image must be a positive integer")
    if not isinstance(n_to_generate, int) or n_to_generate <= 0:
        raise ValueError(f"n_to_generate must be a positive integer")
    if not isinstance(angle_interval, tuple):
        raise ValueError(f"angle_interval must be a tuple")
    if len(angle_interval) != 2:
        raise ValueError(f"angle_interval must have exactly 2 elements")
    if not all(isinstance(a, (float, int)) for a in angle_interval):
        raise ValueError("Both elements of angle_interval must be float or int")
    if not isinstance(desired_angle, float):
        raise TypeError(f"angle_max must be a float")
    if not isinstance(angle_offset, float):
        raise TypeError(f"offset_angle must be a float")

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
                    transform = get_rotate_transformation(r, new_width, new_height, x_min, y_min, x_max, y_max, increase_saturation)
                    
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


def flip_data_augmentation(
        max_per_image: int,
        n_to_generate: int,
        angle_interval: tuple[float, float],
        angle_offset: float = 10.0
    ) -> None:
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
    # check parameters
    if not isinstance(max_per_image, int) or max_per_image <= 0:
        raise ValueError(f"max_per_image must be a positive integer")
    if not isinstance(n_to_generate, int) or n_to_generate <= 0:
        raise ValueError(f"n_to_generate must be a positive integer")
    if not isinstance(angle_interval, tuple):
        raise ValueError(f"angle_interval must be a tuple")
    if len(angle_interval) != 2:
        raise ValueError(f"angle_interval must have exactly 2 elements")
    if not all(isinstance(a, (float, int)) for a in angle_interval):
        raise ValueError("Both elements of angle_interval must be float or int")
    if not isinstance(angle_offset, float):
        raise TypeError(f"offset_angle must be a float")
    
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
                if theta - angle_offset < MIN_THETA:
                    # careful: the image will be flipped
                    rotations = np.linspace(-2 * angle_offset, 0.0, ANGLE_STEP)
                elif theta + angle_offset > MAX_THETA:
                    # careful: the image will be flipped
                    rotations = np.linspace(0.0, 2 * angle_offset, ANGLE_STEP)
                else:
                    rotations = np.linspace(-angle_offset, angle_offset, ANGLE_STEP)
                np.random.shuffle(rotations)

                rotations = rotations.tolist()
                
                for r in rotations:
                    transform = get_flip_transformation(r, new_width, new_height, x_min, y_min, x_max, y_max, increase_saturation)
                            
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
                    
                    if save_augmented_data(image_name, augmentations["image"], cleaned_bbox[0], cleaned_keypoints, horizontal_flip=True):
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
    # check parameters
    if not isinstance(max_per_image, int) or max_per_image <= 0:
        raise ValueError(f"max_per_image must be a positive integer")
    if not isinstance(n_to_generate, int) or n_to_generate <= 0:
        raise ValueError(f"n_to_generate must be a positive integer")
    if original_shape not in ["Horizontal Rectangle", "Vertical Rectangle"]:
        raise ValueError(
            f"original_shape must be either 'Horizontal Rectangle' or 'Vertical Rectangle', "
        )

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
    "prefix_number_name of the original image.extension."
    This function deletes all augmented images that have a number
    above index_threshold in their name.

    Parameters:
    index_threshold (int): The number from which all augmented images
                           with a number superior to this, will be
                           deleted

    Returns:
    None
    '''
    # check parameter
    if not isinstance(index_threshold, int) or index_threshold <= 0:
        raise ValueError(f"index_threshold must be a positive integer")
    
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
       
    print(f"{count} agumented images deleted above index {index_threshold}")


# =============================================================================

def dataframes_data_augmentation():
    '''
    Generate new augmented images from the default + added dataset
    guided by image and labels dataframes.
    Saves the images and labels in the augmented-data folder
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
    ring_data_augmentation(n_per_square=25, ring_x_values=second_ring_x_values, ring_y_values=second_ring_y_values)

    # third ring of the centers heatmap
    # WARNING, THE VALUES OF THE HEATMAP GRAPH ARE IN ANOTHER COORDINATE SYSTEM
    third_ring_x_values = np.linspace(0.31, 0.67, 6)
    third_ring_y_values = np.linspace(0.79, 0.44, 6)
    print("Third ring data augmentation...")
    ring_data_augmentation(n_per_square=50, ring_x_values=third_ring_x_values, ring_y_values=third_ring_y_values)

    theta_data_augmentation(max_per_image=3, n_to_generate=400, angle_interval=(MIN_THETA, 90.0), desired_angle=62.0)
    theta_data_augmentation(max_per_image=3, n_to_generate=400, angle_interval=(90, MAX_THETA), desired_angle=110.0)

    flip_data_augmentation(max_per_image=3, n_to_generate=800, angle_interval=(MIN_THETA, 89.0))
    flip_data_augmentation(max_per_image=3, n_to_generate=800, angle_interval=(89.0, MAX_THETA))

    square_data_augmentation(max_per_image=2, n_to_generate=1400, original_shape="Horizontal Rectangle")
    square_data_augmentation(max_per_image=4, n_to_generate=600, original_shape="Vertical Rectangle")

    delete_too_augmented_images(index_threshold=9)


dataframes_data_augmentation()




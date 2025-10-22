
# we need to prepare the dataframes of the images and the labels
# these dataframes will be used in notebooks and during data augmentation

import os
from PIL import Image
import numpy as np
import pandas as pd
from config import *

def load_labes_dataframe():
    '''
    Load the main dataframe that will contain for every image the following informations:
    - visibility for each point
    - center: the center of the foosball table
    - direction: the normalized direction where the foosball table aim
    - dimension: the percentage of the image that is covered by the bounding box
    '''
    data = []
    for file in os.listdir(LABELS_DIRECTORY):
        if file.endswith(LABELS_EXTENSION):

            with open(os.path.join(LABELS_DIRECTORY, file)) as label:
                content = label.readline()

            numbers = [float(x) for x in content[1:].split()]  # we ignore the first number that is always 0, because we only have one class, the foosball table

            # WARNING: we have to convert the position from the image coordinate system to a coordinate system more readable
            
            # the bounding box is rapresented as x and y of the center of the rectangle, and width and height of the rectangle
            bounding_box = (numbers[0], 1.0 - numbers[1], numbers[2], numbers[3])
            
            # the keypoint are rapresente as x, y and visibility (2: visible, 1: not visible, 0: not present)
            keypoints = []
            for i in range(4, 28, 3):
                keypoints.append((numbers[i], 1.0 - numbers[i+1], int(numbers[i+2])))

            # let's calculate the dimension: how much the bounding box covers the image
            dimension = bounding_box[2] * bounding_box[3]

            # let's calculate the center of the foosball table
            # for simplicity, we consider the center of the upper rectangle as the center of the foosball table

            # intersection between the line (keypoints[0], keypoints[2]) and the line (keypoints[1], keypoints[3])
            def calculate_intersection(line1, line2):
                '''
                ...
                '''
                x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
                y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

                def determinant(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                divisor = determinant(x_diff, y_diff)
                if divisor == 0:
                    raise ValueError('The lines do not intersect.')

                d = (determinant(*line1), determinant(*line2))
                x = determinant(d, x_diff) / divisor
                y = determinant(d, y_diff) / divisor
                return [x, y]
            

            center = calculate_intersection([keypoints[0][0:2], keypoints[2][0:2]], [keypoints[1][0:2], keypoints[3][0:2]])

            # let's calculate the normalized direction of the foosball table
            # for simplicity, we consider the first 2 keypoints of the upper rectangle as the reference for the direction

            # we find the average point between keypoints[0] and keypoint[1]
            average_point = [(keypoints[0][0] + keypoints[1][0]) / 2.0, (keypoints[0][1] + keypoints[1][1]) / 2.0]
            
            # the direction from center to average_point is average_point - center
            direction = [average_point[0] - center[0], average_point[1] - center[1]]
            direction = direction / np.linalg.norm(direction)
            
            
            data.append({
                "filename": file,
                "center": center,
                "direction": direction,
                "dimension": dimension,
            })
            
    return pd.DataFrame(data)


def load_images_dataframe():
    '''
    Load the main dataframe that will contain for every image the following informations:
    - ratio = width/height
    - shape = {Square, Horizontal Rectangle, Vertical Rectangle}
    - resolution = {High, Low}
    - brightness_mean: The mean of the brightness of the image, when converted to HSV
    - saturation_mean: The mean of the saturation of the image, when converted to HSV
    '''
    data = []
    for file in os.listdir(IMAGES_DATA_DIRECTORY):
        if file.endswith(tuple(IMAGES_DATA_EXTENSIONS + [ext.upper() for ext in IMAGES_DATA_EXTENSIONS])):

            img = Image.open(os.path.join(IMAGES_DATA_DIRECTORY, file)).convert("RGB")
            width, height = img.size
            ratio = width / height

            if 0.95 <= ratio <= 1.05:
                shape = "Square"
            elif ratio > 1.05:
                shape = "Horizontal Rectangle"
            else:
                shape = "Vertical Rectangle"

            # 640 x 640 is the input dimension of YOLO
            # if max(width, height) > 640, that means the image has to be downscaled for YOLO (no interpolation)
            # if max(width, height) < 640, that means the image has to be upscaled for YOLO (interpolation)
            # interpolation might bring some noise to informations
            if max(width, height) >= 640:
                resolution = "High"
            else:
                resolution = "Low"

            # convert to HSV to find brightness and saturation means
            hsv_img = img.convert("HSV")
            h, s, v = hsv_img.split()
            brightness_mean = np.array(v).mean()
            saturation_mean = np.array(s).mean()

            data.append({
                "filename": file,
                "ratio": ratio,
                "shape": shape,
                "resolution": resolution,
                "brightness_mean": brightness_mean,
                "saturation_mean": saturation_mean
            })

    return pd.DataFrame(data)


# =============================================================================

# load the dataframes and save them as .parquet

df1 = load_images_dataframe()
df1.to_parquet(IMAGES_DATAFRAME_DIRECTORY)

df2 = load_labes_dataframe()
df2.to_parquet(LABELS_DATAFRAME_DIRECTORY)

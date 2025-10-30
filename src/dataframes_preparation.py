
# we need to prepare the dataframes of the images and the labels
# these dataframes will be used in notebooks and during data augmentation

import os
import argparse
import sys
from PIL import Image
import numpy as np
import pandas as pd
from config import *

def load_labels_dataframe():
    '''
    Load the main dataframe that will contain for every image the following informations:
    - center: the center of the foosball table
    - direction: the normalized direction where the foosball table aim
    - visibilities: the visibilities of the keypoints
    - dimension: the percentage of the image that is covered by the bounding box
    - highest_keypoint: the id of the keypoint that is the highest on the y axis (lowest high in the image)
    '''
    data = []
    for file in os.listdir(LABELS_DIRECTORY):
        if file.endswith(LABELS_EXTENSION):

            with open(os.path.join(LABELS_DIRECTORY, file)) as label:
                content = label.readline()
            numbers = [float(x) for x in content[1:].split()]  # we ignore the first number that is always 0, because we only have one class, the foosball table

            # WARNING: we have to convert the position from the image coordinate system, where the top left point is 0
            # to a coordinate system more familiar, where the bottom left point is 0
            # knowing that the labels are normalized
            # new_x = old_x
            # new_y = 1.0 - old_y

            def convert_old_y(old_y: float, visibility: int = 2) -> float:
                '''
                Converts the y coordinate from the coordinate system of the image, where 0 is at the top,
                to the coordinate system where 0 is at the bottom, being careful about the visibility 0.

                Parameters:
                old_y (float): The y coordinate to convert
                visibility (int): The visibility of the point where the y was taken

                Returns:
                float: The converted y coordinate
                '''
                if visibility == 0: return 0.0
                else: return 1.0 - old_y
            
            # the bounding box is rapresented as x and y of the center of the rectangle, and width and height of the rectangle
            bounding_box = (numbers[0], convert_old_y(numbers[1]), numbers[2], numbers[3])
            
            # the keypoints are rapresented as x, y and visibility (2: visible, 1: not visible, 0: not present)
            keypoints = []
            visibilities = []
            for i in range(4, 28, 3):
                visibility = int(numbers[i+2])
                keypoints.append((numbers[i], convert_old_y(numbers[i+1], visibility)))
                visibilities.append(visibility)

            highest_keypoint = np.argmax((np.array(keypoints))[:,1])
            # if highest_keypoint is different from 0 or 1 we have an error in the annotations
            if highest_keypoint > 1:
                raise ValueError(f"Warning: the highest keypoint for {file} is {highest_keypoint}. Unable to generate dataframe")

            # let's calculate the dimension: how much the bounding box covers the image as a percentage
            dimension = (bounding_box[2] * bounding_box[3]) * 100.0

            # let's calculate the center of the foosball table
            # for simplicity, we consider the center of the upper rectangle as the center of the foosball table

            # intersection between the line (keypoints[0], keypoints[2]) and the line (keypoints[1], keypoints[3])
            def calculate_intersection(line1: tuple, line2: tuple) -> tuple:
                '''
                Calculate the intersection of 2 lines, each rapresented as 2 points in 2d.

                Parameters:
                line1 (tuple): The first line
                line2 (tuple): The second line

                Returns:
                tuple: The point of intersection if it exists
                '''
                x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
                y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

                def determinant(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                divisor = determinant(x_diff, y_diff)
                if divisor == 0:
                    raise ValueError("The lines do not intersect.")

                d = (determinant(*line1), determinant(*line2))
                x = determinant(d, x_diff) / divisor
                y = determinant(d, y_diff) / divisor
                return x, y
            
            center = calculate_intersection((keypoints[0], keypoints[2]), (keypoints[1], keypoints[3]))

            # let's calculate the normalized direction of the foosball table
            # the direction goes from the center to the point in the middle between the first and the second keypoint
            # WARNING: this middle point is NOT the average point, because of perspective
            # the middle point is the intersection between the lines that connect the first to the second keypoints, and the center to the vanishing point
            # the vanishing point is obtained by intersecting the lines that connect the first keypoint to the fourth, and the second to the third
            vanishing_point = calculate_intersection((keypoints[0], keypoints[3]), (keypoints[1], keypoints[2]))
            middle_point = calculate_intersection((keypoints[0], keypoints[1]), (center, vanishing_point))
            direction = [middle_point[0] - center[0], middle_point[1] - center[1]]
            direction = direction / np.linalg.norm(direction)
            
            data.append({
                "filename": file,
                "center": center,
                "direction": direction,
                "visibilities": visibilities,
                "dimension": dimension,
                "highest_keypoint": highest_keypoint
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


def save_dataframe(df_to_generate: str) -> None:
    '''
    Generates and saves the requested dataframe.
    
    Parameters:
    df_to_generate: {IMAGES, LABELS}

    Returns:
    None
    '''
    df_to_generate = df_to_generate.upper()

    if df_to_generate == "IMAGES":
        print(f"Loading dataframe for {df_to_generate}...")
        df = load_images_dataframe()
        output_path = IMAGES_DATAFRAME_DIRECTORY
    elif df_to_generate == "LABELS":
        print(f"Loading dataframe for {df_to_generate}...")
        df = load_labels_dataframe()
        output_path = LABELS_DATAFRAME_DIRECTORY
    else:
        print(f"Error: {df_to_generate} is not a valid request to generate a dataframe")
        sys.exit(1)

    print(f"Saving dataframe for {df_to_generate} in {output_path}...")
    df.to_parquet(output_path)
    print(f"Saved in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates and saves the requested dataframe."
    )
    parser.add_argument(
        "df_to_generate",
        type=str,
        help="The dataframe to generate (IMAGES, LABELS)",
    )
    args = parser.parse_args()
    save_dataframe(args.df_to_generate)



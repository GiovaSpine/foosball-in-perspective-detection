
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

import pandas as pd
from config import *

# load the dataframes
images_df = pd.read_parquet(IMAGES_DATAFRAME_DIRECTORY)
labels_df = pd.read_parquet(LABELS_DATAFRAME_DIRECTORY)






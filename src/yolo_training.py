from ultralytics import YOLO
from config import *

# model loading
model = YOLO("yolov8n-pose.pt")

name = "train05"  # name of the training

# training
model.train(
    project=YOLO_RUNS_DIRECTORY,
    name=name,
    data=YOLO_CONFIG_DIRECTORY,
    classes=[0],
    pose=13.0,
    epochs=100,
    patience=20,
    batch=-1,
    lr0=0.01,
    weight_decay=0.0007,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    workers=4
)


from ultralytics import YOLO
from config import *

model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

model.train(data=YOLO_CONFIG_DIRECTORY, epochs=1, imgsz = 640)
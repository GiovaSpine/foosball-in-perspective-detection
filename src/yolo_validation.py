import os
import json
from ultralytics import YOLO
from config import *

# model loading
model = YOLO("yolo/runs/train05/weights/best.pt")

# results
results = model.val(project=YOLO_RUNS_DIRECTORY)

# write results
with open(os.path.join(YOLO_RUNS_DIRECTORY, "val", "pose_val_metrics.json"), "w") as f:
    json.dump(results.results_dict, f, indent=4)
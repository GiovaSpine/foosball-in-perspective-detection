from flask import Flask, request, redirect, url_for, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import os
from pathlib import Path
import uuid



app = Flask(__name__, template_folder="templates")


# Carico il modello YOLO keypoints già allenato
APP_DIRECTORY = Path(__file__).resolve().parent
model = YOLO(os.path.join(APP_DIRECTORY, "model", "best.pt"))

UPLOAD_FOLDER = os.path.join(APP_DIRECTORY, "uploads")

@app.route("/")
def index():
    '''index'''
    return render_template("index.html")

@app.route("/documentation")
def documentation():
    '''documentation'''
    return render_template("documentation.html")

@app.route("/results")
def results():
    '''results'''
    return render_template("results.html")

@app.route("/guide")
def guide():
    '''guide'''
    return render_template("guide.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    '''predict'''
    if request.method == "GET":
        return render_template("predict.html")

    # we receive the image from request
    if "photo" not in request.files:
        return jsonify({"error": "No photo"}), 400

    file = request.files["photo"]
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)
    result = results[0]

    os.remove(filepath)

    keypoints = []
    if result.keypoints is not None:
        for kps in result.keypoints.xy:
            keypoints.append(kps.tolist())
    
    bounding_boxes = []
    if result.boxes is not None:
        for bbox in result.boxes.xywh:
            bounding_boxes.append(bbox.tolist())
    
    confidence = []
    if result.keypoints is not None:
        for kp_conf in result.keypoints.conf:
            keypoints.append(kp_conf.tolist())


    return jsonify({
        "keypoints": keypoints,
        "bounding_box": bounding_boxes,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

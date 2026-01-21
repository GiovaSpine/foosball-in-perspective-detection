from flask import Flask, request, redirect, url_for, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import os
from pathlib import Path
import uuid
from algorithms.algorithms import *



app = Flask(__name__, template_folder="templates")

APP_DIRECTORY = Path(__file__).resolve().parent
model = YOLO(os.path.join(APP_DIRECTORY, "model", "best.pt"))

UPLOAD_FOLDER = os.path.join(APP_DIRECTORY, "uploads")

@app.route("/")
def index():
    '''
    Renders and returns the main homepage of the application.

    This route handles requests to the root URL ("/") and serves
    the `index.html` template.
    '''
    return render_template("index.html")


@app.route("/project-documentation")
def project_documentation():
    '''
    Renders and returns the project documentation page.

    This route handles requests to the "/project-documentation" URL
    and serves the `project_documentation.html` template.
    '''
    return render_template("project_documentation.html")


@app.route("/api-documentation")
def api_documentation():
    '''
    Renders and returns the API documentation page.

    This route handles requests to the "/api-documentation" URL
    and serves the `api_documentation.html` template.
    '''
    return render_template("api_documentation.html")


@app.route("/results")
def results():
    '''
    Renders and returns the results page.

    This route handles requests to the "/results" URL
    and serves the `results.html` template.
    '''
    return render_template("results.html")


@app.route("/predict/choose-from-samples")
def choose_from_samples():
    '''
    Renders and returnss the page for selecting a sample for prediction.

    This route handles requests to the "/predict/choose-from-samples" URL
    and serves the `choose_sample.html` template.
    '''
    return render_template("choose_sample.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    '''
    Handles image prediction requests.

    GET: 
        Renders the prediction page (`predict.html`) for the user to upload an image
        and test the model and APIs.
    
    POST:
        Receives an image file from the request, saves it temporarily, and passes it
        to the prediction model. Extracts keypoints, bounding boxes, and confidence
        scores from the model's result, then deletes the temporary file.
        Returns a JSON response containing:
            - "keypoints": list of keypoint coordinates [[x1, y2], ...]
            - "bounding_boxes": list of bounding box coordinates (x, y, w, h)
            - "confidence": list of confidence scores for each keypoint [c1, ...]

    Returns:
        HTML template (on GET) or JSON response (on POST).
    '''
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
            confidence.append(kp_conf.tolist())

    return jsonify({
        "keypoints": keypoints,
        "bounding_boxes": bounding_boxes,
        "confidence": confidence
    })


@app.route("/translate-position", methods=["POST"])
def translate_position():
    '''
    Translates a point from image coordinates to foosball table coordinates,
    based on four lower keypoints (play area).

    This route accepts a POST request with a JSON body containing:
        - "lower_keypoints": list of four keypoints defining a quadrilateral (play area)
        - "point": the coordinates of the point to be translated

    Error Handling:
        - 400: Missing JSON body or invalid input types
        - 422: Non-convex quadrilateral or point outside quadrilateral, or invalid values
        - 500: Algorithm failure or unexpected errors

    Returns:
        JSON containing the translated point:
            {"translated_point": [x, y]}
    '''
    # we receive the 4 lower keypoints (the last 4) and the point to translate
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body"}), 400

    lower_keypoints = data.get("lower_keypoints")
    point = data.get("point")

    if not lower_keypoints or not point:
        return jsonify({"error": "Missing data"}), 400
    
    try:
        translated_point = translate_point(point, lower_keypoints)

    except NonConvexQuadrilateralError as e:
        return jsonify(error=str(e)), 422

    except PointOutsideQuadrilateralError as e:
        return jsonify(error=str(e)), 422

    except AlgorithmFailedError:
        return jsonify(error="Unable to translate the point"), 500

    except TypeError as e:
        return jsonify(error=f"Input type error: {e}"), 400
    
    except ValueError as e:
        return jsonify(error=f"Input value error: {e}"), 422

    except Exception:
        return jsonify(error="Unexpected error"), 500

    return jsonify({
        "translated_point": translated_point,
    })


@app.route("/get-player-lines", methods=["POST"])
def get_player_lines():
    '''
    Calculates the eight player lines based on provided keypoints.

    This route accepts a POST request with a JSON body containing:
        - "keypoints": list of all keypoints required to compute player lines.

    Error Handling:
        - 400: Missing JSON body or invalid input types
        - 422: Non-convex quadrilateral or invalid input values
        - 500: Algorithm failure or unexpected errors

    Returns:
        JSON containing the calculated player lines:
            {"player_lines": [[[x1, y1], [x2, y2]], ...]}
    '''
    # we receive all the keypoints
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body"}), 400

    keypoints = data.get("keypoints")

    if not keypoints:
        return jsonify({"error": "Missing data"}), 400
    
    try:
        player_lines = calculate_player_lines(keypoints)

    except NonConvexQuadrilateralError as e:
        return jsonify(error=str(e)), 422
    
    except AlgorithmFailedError as e:
        return jsonify(error=str(e)), 500
    
    except TypeError as e:
        return jsonify(error=f"Input type error: {e}"), 400
    
    except ValueError as e:
        return jsonify(error=f"Input value error: {e}"), 422
    
    except Exception:
        return jsonify(error="Unexpected error"), 500

    return jsonify({
        "player_lines": player_lines,
    })


@app.route("/clean-keypoints", methods=["POST"])
def clean_keypoints():
    '''
    Cleans raw predicted keypoints based on the provided image dimensions.

    This route accepts a POST request with a JSON body containing:
        - "keypoints": list of keypoints to clean
        - "width": width of the image
        - "height": height of the image

    Error Handling:
        - 400: Missing JSON body, keypoints, or image dimensions; invalid input types
        - 422: Invalid input values
        - 500: Algorithm failure or unexpected errors

    Returns:
        JSON containing the cleaned keypoints:
            {"cleaned_keypoints": [[x1, y1], ...]}
    '''
    # we receive keypoints, width and height
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body"}), 400

    keypoints = data.get("keypoints")
    width = data.get("width")
    height = data.get("height")

    if not keypoints:
        return jsonify({"error": "Missing keypoints data"}), 400

    if width is None or height is None:
        return jsonify({"error": "Missing width or height"}), 400
    
    try:
        cleaned_keypoints = keypoints_cleaning(keypoints, width, height)

    except AlgorithmFailedError as e:
        return jsonify(error=str(e)), 500

    except TypeError as e:
        return jsonify(error=f"Input type error: {e}"), 400
    
    except ValueError as e:
        return jsonify(error=f"Input value error: {e}"), 422
    
    except Exception:
        return jsonify(error="Unexpected error"), 500
    
    return jsonify({
        "cleaned_keypoints": cleaned_keypoints,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)




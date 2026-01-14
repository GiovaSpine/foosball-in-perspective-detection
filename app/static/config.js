

export const colors = {
    photo_canvas_color: "rgb(228, 228, 228)",
    reference_canvas_color: "rgb(64, 64, 255)",
    keypoint_color: "orange",
    bounding_box_color: "red",
    play_area_color: "rgba(255, 0, 255, 0.4)",
    player_lines_color: "yellow",
    translated_point_color: "orange",
    x_axis_color: "red",
    y_axis_color: "rgb(0, 255, 0)",
    z_axis_color: "blue"
}


export const icons = {
    grid_path: "./static/images/icons/Grid.svg",
    coordinate_system_path: "./static/images/icons/Coordinate System.svg",
    eye_path: "./static/images/icons/eye.svg",
    eye_hidden_path: "./static/images/icons/eye-hidden.svg",
}


export const messages = {
    no_image_provided: "No image is provided.",
    no_valid_prediction: "Invalid prediction: YOLO was not able to produce keypoints.\n(remember that the model requires an image of a foosball table in a spectator view)",
    translate_position_instructions: "Select a point inside the highlighted area",
    keypoints_cleaned: "The keypoints were cleaned."
}
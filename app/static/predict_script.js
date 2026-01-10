import { image_position_and_scale, draw_photo, draw_reference, draw_translated_point } from "./image_renderer.js"
import { wait_for_click_or_escape } from "./events.js";
import { page_pos_to_canvas_pos, canvas_pos_to_image_pos } from "./utils.js";
import { colors, icons, messages } from "./config.js";

// get canvas elements
const photo_input = document.getElementById("photo_input");
export const photo_canvas = document.getElementById("photo_canvas");
export const pctx = photo_canvas.getContext("2d");
export const reference_canvas = document.getElementById("reference_canvas");
export const rctx = reference_canvas.getContext("2d");

// get message and image title element
const message = document.getElementById("message");
const image_title = document.getElementById("image_title");

// set canvas size
const rect1 = photo_canvas.getBoundingClientRect();
photo_canvas.width = rect1.width;
photo_canvas.height = rect1.height;
const rect2 = reference_canvas.getBoundingClientRect();
reference_canvas.width = rect2.width;
reference_canvas.height = rect2.height;

// canvas color
pctx.fillStyle = colors.photo_canvas_color;
pctx.fillRect(0, 0, photo_canvas.width, photo_canvas.height);
draw_reference();

// button functions
window.toggle_keypoints = toggle_keypoints;
window.toggle_bounding_box = toggle_bounding_box;
window.toggle_play_area = toggle_play_area;
window.toggle_edges = toggle_edges;
window.toggle_player_lines = toggle_player_lines;
window.translate_position = translate_position;

// state to store the current image, its rendering parameters on the canvas,
// the latest model prediction, and visualization flags.
export let state = {
  photo: null,
  image_x: 0,
  image_y: 0,
  image_scale: 1.0,
  prediction: null,
  toggle_keypoints: false,
  toggle_bounding_box: false,
  toggle_play_area: false,
  toggle_edges: false,
  toggle_player_lines: false,
  player_lines: null
}

// ========================================================

if (sessionStorage.getItem("photo")) {
  // there is already a photo in the browser session
  const photo = JSON.parse(sessionStorage.getItem("photo"));

  image_title.textContent = photo.name;

  // there is also the prediction
  state.prediction = JSON.parse(sessionStorage.getItem("prediction"));

  state.photo = new Image();
  state.photo.src = photo.data;
  state.photo.onload = () => {
    const [x, y, scale] = image_position_and_scale(state.photo.width, state.photo.height, photo_canvas.width, photo_canvas.height);
    state.image_x = x;
    state.image_y = y;
    state.image_scale = scale;
    draw_photo(state);
  }
}


photo_input.addEventListener("change", async (e) => {
  // the user chose a photo
  message.textContent = "";

  // reset the reference canvas if there were translated point
  draw_reference();

  const file = e.target.files[0];
  if (!file) return;

  image_title.textContent = file.name;

  // save the photo in the browser session
  const reader = new FileReader();
  reader.onload = () => {
    sessionStorage.setItem("photo", JSON.stringify({
      data: reader.result,
      name: file.name
    }));
  };
  reader.readAsDataURL(file);

  // send the image to the server
  const formData = new FormData();
  formData.append("photo", file);

  // receive the prediction
  const response1 = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  let prediction = await response1.json();
  console.log(prediction)

  if (prediction.keypoints.length != 0) {
    // let's use the clean keypoints API
    const response2 = await fetch("/clean-keypoints", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        keypoints: prediction.keypoints[0],
      })
    });
    if (!response2.ok) {
      const err = await response2.json();
      message.textContent = "Error: " + err.error;
    } else {
      const cleaned_keypoints = await response2.json();
      prediction.keypoints[0] = cleaned_keypoints.keypoints;
    }

  }
  state.prediction = prediction;

  // check the prediction
  if (state.prediction.keypoints.length == 0) {
    // no valid prediction
    message.textContent = messages.no_valid_prediction;
    // we can't visualize anything
    state.toggle_keypoints = false;
    state.toggle_bounding_box = false;
    state.toggle_play_area = false;
    state.toggle_edges = false;
    state.toggle_player_lines = false;
    update_icons();
  }

  // the user want to see update the player_lines, so we have to update them
  if (state.toggle_player_lines) {
    // there can be an error, like one of the faces on the side is not convex
    const result = await get_player_lines();
    if (!result) {
      state.toggle_player_lines = false;
      update_icons();
    }
  }

  // save the prediction in the session
  sessionStorage.setItem("prediction", JSON.stringify(state.prediction));

  // show the image and the results on the canvas
  const url = URL.createObjectURL(file);
  state.photo = new Image();
  state.photo.onload = () => {
    const [x, y, scale] = image_position_and_scale(state.photo.width, state.photo.height, photo_canvas.width, photo_canvas.height);
    state.image_x = x;
    state.image_y = y;
    state.image_scale = scale;
    draw_photo(state);
  };
  state.photo.src = url;
});


async function get_player_lines() {
  // we need to ask the server where are the player lines

  // send to the server the 4 lower keypoints and the interested point
  const response = await fetch("/get-player-lines", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      keypoints: state.prediction.keypoints[0],
    })
  });
  // there can be an error, like ...
  if (!response.ok) {
    const err = await response.json();
    message.textContent = "Error: " + err.error;
    return false;
  }

  const data = await response.json();
  const player_lines = data.player_lines;
  state.player_lines = player_lines;
  console.log(player_lines)
  return true
}

// --------------------------------------------------------
// BUTTON FUNCTIONS

function update_icons() {

  if (state.toggle_keypoints) {
    const button = document.getElementById("toggle_keypoints");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_path;
  } else {
    const button = document.getElementById("toggle_keypoints");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_hidden_path;
  }

  if (state.toggle_bounding_box) {
    const button = document.getElementById("toggle_bounding_box");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_path;
  } else {
    const button = document.getElementById("toggle_bounding_box");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_hidden_path;
  }

  if (state.toggle_play_area) {
    const button = document.getElementById("toggle_play_area");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_path;
  } else {
    const button = document.getElementById("toggle_play_area");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_hidden_path;
  }

  if (state.toggle_edges) {
    const button = document.getElementById("toggle_edges");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_path;
  } else {
    const button = document.getElementById("toggle_edges");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_hidden_path;
  }

  if (state.toggle_player_lines) {
    const button = document.getElementById("toggle_player_lines");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_path;
  } else {
    const button = document.getElementById("toggle_player_lines");
    const icon = button.querySelector(".icon");
    icon.src = icons.eye_hidden_path;
  }
}


function check_photo_and_prediction() {
  if (state.photo == null) {
    // there is no image
    message.textContent = messages.no_image_provided;
    return false;
  }
  if (state.prediction.keypoints.length == 0) {
    // no valid prediction
    message.textContent = messages.no_valid_prediction;
    return false;
  }
  return true;
}


function toggle_keypoints() {
  if (!check_photo_and_prediction()) return;

  message.textContent = "";

  if (state.toggle_keypoints) state.toggle_keypoints = false;
  else state.toggle_keypoints = true;

  update_icons();
  draw_photo();
}


function toggle_bounding_box() {
  if (!check_photo_and_prediction()) return;

  message.textContent = "";

  if (state.toggle_bounding_box) state.toggle_bounding_box = false;
  else state.toggle_bounding_box = true;

  update_icons();
  draw_photo();
}


function toggle_play_area() {
  if (!check_photo_and_prediction()) return;

  message.textContent = "";

  if (state.toggle_play_area) state.toggle_play_area = false;
  else state.toggle_play_area = true;

  update_icons();
  draw_photo();
}


function toggle_edges() {
  if (!check_photo_and_prediction()) return;

  message.textContent = "";

  if (state.toggle_edges) state.toggle_edges = false;
  else state.toggle_edges = true;

  update_icons();
  draw_photo();
}


async function toggle_player_lines() {
  if (!check_photo_and_prediction()) return;

  message.textContent = "";

  if (state.toggle_player_lines) state.toggle_player_lines = false;
  else {
    state.toggle_player_lines = true;
    // there can be an error, like one of the faces on the side is not convex
    const result = await get_player_lines();
    if (!result) state.toggle_player_lines = false;
  }

  update_icons();
  draw_photo();
}


function enable_buttons(enable) {
  // enable or disable all buttons
  document.getElementById("photo_input").disabled = !enable;
  document.getElementById("toggle_keypoints").disabled = !enable;
  document.getElementById("toggle_bounding_box").disabled = !enable;
  document.getElementById("toggle_play_area").disabled = !enable;
  document.getElementById("toggle_edges").disabled = !enable;
  document.getElementById("toggle_player_lines").disabled = !enable;
  document.getElementById("translate_position").disabled = !enable;
}


async function translate_position() {
  if (!check_photo_and_prediction()) return;

  message.textContent = messages.translate_position_instructions;

  // block all buttons
  enable_buttons(false);

  // draw the area to select the keypoint
  let delete_play_area = true;
  if (state.toggle_play_area) {
    // the play area was already on, so this function shouldn't delete it
    delete_play_area = false;
  } else {
    // the play area was off
    state.toggle_play_area = true;
    draw_photo();
  }

  function quit_translate_position() {
    if (delete_play_area) {
      state.toggle_play_area = false;
      draw_photo();
    }
    // enable all buttons
    enable_buttons(true);
  }

  // the last 4 keypoints are the ones for the play area
  const lower_keypoints = state.prediction.keypoints[0].slice(-4);

  // what's the point to translate ?
  const e = await wait_for_click_or_escape();
  if (e == null) {
    quit_translate_position();  // the user pressed esc
    message.textContent = "";
    return;
  }
  const [cursur_x_canvas, cursur_y_canvas] = page_pos_to_canvas_pos(e.pageX, e.pageY);
  const [cursur_x_image, cursur_y_image] = canvas_pos_to_image_pos(cursur_x_canvas, cursur_y_canvas);
  const point = [cursur_x_image, cursur_y_image];
  
  // send to the server the 4 lower keypoints and the interested point
  const response = await fetch("/translate-position", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      lower_keypoints: lower_keypoints,
      point: point
    })
  });
  // there can be an error, like the quadrilater is not convex or the point is outside the quadrilateral
  if (!response.ok) {
    const err = await response.json();
    message.textContent = "Error: " + err.error;
    quit_translate_position();
    return;
  }

  const data = await response.json();
  const translated_point = data.translated_point;
  draw_translated_point(translated_point)

  quit_translate_position();
  message.textContent =
    "Translated point from \n(" + Math.round(cursur_x_image) + ", " +
    Math.round(cursur_y_canvas) + ") to (" +
    translated_point[0].toFixed(2) + ", " +
    translated_point[1].toFixed(2) + ")";
}


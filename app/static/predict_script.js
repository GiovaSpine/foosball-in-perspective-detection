import { image_position_and_scale, draw, } from "./image_renderer.js"
import { wait_for_click_or_escape } from "./events.js";
import { page_pos_to_canvas_pos, canvas_pos_to_image_pos } from "./utils.js";

const photo_input = document.getElementById("photo_input");
export const photo_canvas = document.getElementById("photo_canvas");
export const pctx = photo_canvas.getContext("2d");
// const reference_canvas = document.getElementById("reference_canvas");
// export const rctx = photo_canvas.getContext("2d");

// set canvas size
const rect = photo_canvas.getBoundingClientRect();
photo_canvas.width = rect.width;
photo_canvas.height = rect.height;

// canvas color
pctx.fillStyle = "rgb(230, 230, 230)";
pctx.fillRect(0, 0, photo_canvas.width, photo_canvas.height);

// button functions
window.show_keypoints = show_keypoints;
window.show_bounding_box = show_bounding_box;
window.show_play_area = show_play_area;
window.show_edges = show_edges;
window.show_goalnets = show_goalnets;
window.show_player_lines = show_player_lines;
window.translate_position = translate_position;

// ...
export let state = {
  photo: null,
  image_x: 0,
  image_y: 0,
  image_scale: 1.0,
  prediction: null,
  show_keypoints: false,
  show_bounding_box: false,
  show_play_area: false,
  show_edges: false,
  show_goalnets: false,
  show_player_lines: false,
  show_play_area: false,
}

// ========================================================

if (sessionStorage.getItem("photo")) {
  // there is already a photo in the browser session
  const photo_url = sessionStorage.getItem("photo");

  // there is also the prediction
  state.prediction = JSON.parse(sessionStorage.getItem("prediction"));

  state.photo = new Image();
  state.photo.src = photo_url;
  state.photo.onload = () => {
    const [x, y, scale] = image_position_and_scale(state.photo.width, state.photo.height, photo_canvas.width, photo_canvas.height);
    state.image_x = x;
    state.image_y = y;
    state.image_scale = scale;
    draw(state);
  }
}


photo_input.addEventListener("change", async (e) => {
  // the user chose a photo

  const file = e.target.files[0];
  if (!file) return;

  // save the photo in the browser session
  const reader = new FileReader();
  reader.onload = () => {
    sessionStorage.setItem("photo", reader.result);
  };
  reader.readAsDataURL(file);

  // send the image to the server
  const formData = new FormData();
  formData.append("photo", file);

  // receive the prediction
  const response = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  state.prediction = await response.json();
  console.log(state.prediction);

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
    draw(state);
  };
  state.photo.src = url;
});

// --------------------------------------------------------
// BUTTON FUNCTIONS

function show_keypoints(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_keypoints) state.show_keypoints = false;
  else state.show_keypoints = true;
  draw();
}

function show_bounding_box(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_bounding_box) state.show_bounding_box = false;
  else state.show_bounding_box = true;
  draw();
}

function show_play_area(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_play_area) state.show_play_area = false;
  else state.show_play_area = true;
  draw();
}

function show_edges(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_edges) state.show_edges = false;
  else state.show_edges = true;
  draw();
}

function show_goalnets(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_goalnets) state.show_goalnets = false;
  else state.show_goalnets = true;
  draw();
}

function show_player_lines(){
  if(state.photo == null){
    // there is no image
    return;
  }
  if(state.show_player_lines) state.show_player_lines = false;
  else state.show_player_lines = true;
  draw();
}


function enable_buttons(enable){
  // enable or disable all buttons
  document.getElementById("photo_input").disabled = !enable;
  document.getElementById("show_keypoints").disabled = !enable;
  document.getElementById("show_bounding_box").disabled = !enable;
  document.getElementById("show_play_area").disabled = !enable;
  document.getElementById("show_edges").disabled = !enable;
  document.getElementById("show_goalnets").disabled = !enable;
  document.getElementById("show_player_lines").disabled = !enable;
  document.getElementById("translate_position").disabled = !enable;
}

async function translate_position(){
  if(state.photo == null){
    // there is no image
    return;
  }
  // check if the model's prediction contains keypoints
  if(state.prediction.keypoints.length == 0){
    // the model wasn't able to find any keypoint
    return;
  }

  // block all buttons
  enable_buttons(false);

  // draw the area to select the keypoint
  let delete_play_area = true;
  if(state.show_play_area){
    // the play area was already on, so this function shouldn't delete it
    delete_play_area = false;
  } else {
    // the play area was off
    state.show_play_area = true;
    draw();
  }

  function quit_translate_position(){
    if(delete_play_area){
      state.show_play_area = false;
      draw();
    }
    // enable all buttons
    enable_buttons(true);
  }
  
  // the last 4 keypoints are the ones for the play area
  const lower_keypoints = state.prediction.keypoints[0].slice(-4);

  // what's the point to translate ?
  const e = await wait_for_click_or_escape();
  if(e == null){
    quit_translate_position();  // the user pressed esc
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
    console.log(err.error, "Server error");
    quit_translate_position();
    return;
  }

  const data = await response.json();
  const translated_point = data.translated_point;
  console.log(translated_point);

  quit_translate_position();
}


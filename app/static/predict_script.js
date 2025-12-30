import {draw} from "./image_renderer.js"

const photo_input = document.getElementById("photo_input");
export const photo_canvas = document.getElementById("photo_canvas");
export const pctx = photo_canvas.getContext("2d");
// const reference_canvas = document.getElementById("reference_canvas");
// export const rctx = photo_canvas.getContext("2d");

// set canvas size
const rect = photo_canvas.getBoundingClientRect();
photo_canvas.width = rect.width;
photo_canvas.height = rect.height;

window.show_keypoints = show_keypoints;
window.show_bounding_box = show_bounding_box;
window.show_edges = show_edges;
window.show_goalnets = show_goalnets;
window.show_player_lines = show_player_lines;
window.translate_position = translate_position;

// ...
export let state = {
  photo: null,
  prediction: null,
  show_keypoints: false,
  show_bounding_box: false,
  show_edges: false,
  show_goalnets: false,
  show_player_lines: false,
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
    draw(state);
  };
  state.photo.src = url;
});


function toggle_buttons(enable){
  // enable or disable buttons
  document.getElementById("show_keypoints").disabled = !enable;
  document.getElementById("show_bounding_box").disabled = !enable;
  document.getElementById("show_edges").disabled = !enable;
  document.getElementById("show_goalnets").disabled = !enable;
  document.getElementById("show_player_lines").disabled = !enable;
  document.getElementById("translate_position").disabled = !enable;
}


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

async function translate_position(){
  if(state.photo == null){
    // there is no image
    return;
  }
  // check if there if the prediction allows to translate a point
  // ...

  // block all buttons
  document.getElementById("photo_input").disabled = true;
  document.getElementById("show_keypoints").disabled = true;
  document.getElementById("show_bounding_box").disabled = true;
  document.getElementById("show_edges").disabled = true;
  document.getElementById("show_goalnets").disabled = true;
  document.getElementById("show_player_lines").disabled = true;
  document.getElementById("translate_position").disabled = true;

  
}
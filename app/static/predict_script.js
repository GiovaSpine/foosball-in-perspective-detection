console.log("Testing predict_script.js")

const photo_input = document.getElementById("photo_input");
const photo_canvas = document.getElementById("photo_canvas");
const pctx = photo_canvas.getContext("2d");
const reference_canvas = document.getElementById("reference_canvas");
const rctx = photo_canvas.getContext("2d");

// set canvas size
const rect = photo_canvas.getBoundingClientRect();
photo_canvas.width = rect.width;
photo_canvas.height = rect.height;

// image
let photo = new Image();
let photo_x = 0;
let photo_y = 0;
let photo_scale = 1.0;

// ========================================================

if (sessionStorage.getItem("photo")) {
    // there is already a photo in the browser session
    const photo_url = sessionStorage.getItem("photo");

    // there is also the prediction
    const prediction = JSON.parse(sessionStorage.getItem("prediction"));

    photo.src = photo_url;
    photo.onload = () => {
      draw_photo();
      draw_results(prediction.keypoints);
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

  const prediction = await response.json();
  console.log(prediction);

  // save the prediction in the session
  sessionStorage.setItem("prediction", JSON.stringify(prediction));

  // show the image and the results on the canvas
  const url = URL.createObjectURL(file);
  photo.onload = () => {
      draw_photo();
      draw_results(prediction.keypoints, prediction.bounding_boxes);
  };
  photo.src = url;
});



function image_position_and_scale(){
  // determines the position and scale of the image
  if(photo.width > photo_canvas.width || photo.height > photo_canvas.height){
    // we need to scale down the image
    const photo_scale_width = photo_canvas.width / photo.width;
    const photo_scale_height = photo_canvas.height / photo.height;

    photo_scale = Math.min(photo_scale_width, photo_scale_height);

    photo_x = (photo_canvas.width / 2) - ((photo.width * photo_scale) / 2);
    photo_y = (photo_canvas.height / 2) - ((photo.height * photo_scale) / 2);
  } else {
    photo_scale = 1.0;
    
    photo_x = (photo_canvas.width / 2) - (photo.width / 2);
    photo_y = (photo_canvas.height / 2) - (photo.height / 2);
  }
}


function draw_photo(){
  image_position_and_scale();
  pctx.clearRect(0, 0, photo_canvas.width, photo_canvas.height);
  pctx.drawImage(photo, photo_x, photo_y, photo.width * photo_scale, photo.height * photo_scale);
}


function draw_results(keypoints) {
    pctx.fillStyle = "red";

    keypoints.forEach(person => {
        person.forEach(([x, y]) => {
            pctx.beginPath();
            pctx.arc(x, y, 4, 0, Math.PI * 2);
            pctx.fill();
        });
    });
}
import { photo_canvas, pctx, state } from "./predict_script.js";



export function image_position_and_scale(image_width, image_height, canvas_width, canvas_height){
    // determines the position and scale for the image
    // to be in the center of the canvas and fully visible (not cutted out)

    const image_scale_width = canvas_width / image_width;
    const image_scale_height = canvas_height / image_height;

    const image_scale = Math.min(image_scale_width, image_scale_height);

    const image_x = (canvas_width / 2) - ((image_width * image_scale) / 2);
    const image_y = (canvas_height / 2) - ((image_height * image_scale) / 2);
    return [image_x, image_y, image_scale];
}


export function draw(){

    const [x, y, scale] = image_position_and_scale(state.photo.width, state.photo.height, photo_canvas.width, photo_canvas.height);
    
    // clear the canvas
    pctx.clearRect(0, 0, photo_canvas.width, photo_canvas.height);

    // draw the photo on the canvas
    pctx.drawImage(state.photo, x, y, state.photo.width * scale, state.photo.height * scale);

    // draw the results
    draw_results(x, y, scale);
}




function draw_results(x0, y0, scale) {
    pctx.fillStyle = "red";
    console.log(scale)
    console.log()
    state.prediction.keypoints.forEach(person => {
        person.forEach(([x, y]) => {
            pctx.beginPath();
            pctx.arc((x * scale) + x0, (y * scale) + y0 , 4, 0, Math.PI * 2);
            pctx.fill();
        });
    });
}


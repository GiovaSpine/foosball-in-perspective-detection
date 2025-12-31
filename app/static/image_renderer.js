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
    
    // clear the canvas
    pctx.fillStyle = "rgb(230, 230, 230)";
    pctx.fillRect(0, 0, photo_canvas.width, photo_canvas.height);

    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    // draw the photo on the canvas
    pctx.drawImage(state.photo, x, y, state.photo.width * scale, state.photo.height * scale);

    // show play area
    if(state.show_play_area){
        // the last 4 keypoints are the ones for the play area
        const lower_keypoints = state.prediction.keypoints[0].slice(-4);

        // we have to draw the quadrilater that connects them
        pctx.beginPath();
        pctx.moveTo(lower_keypoints[0][0] * scale + x, lower_keypoints[0][1] * scale + y);
        for (let i = 1; i < lower_keypoints.length; i++) {
            pctx.lineTo(lower_keypoints[i][0] * scale + x, lower_keypoints[i][1] * scale + y);
        }
        pctx.closePath();

        pctx.fillStyle = "rgba(255, 0, 255, 0.4)";
        pctx.fill();
    }
    

    // draw the results
    draw_results(x, y, scale);
}




function draw_results(x0, y0, scale) {
    pctx.fillStyle = "red";

    const keypoints = state.prediction.keypoints[0];

    keypoints.forEach(([x, y]) => {
        pctx.beginPath();
        pctx.arc(
            x * scale + x0,
            y * scale + y0,
            4,
            0,
            Math.PI * 2
        );
        pctx.fill();
    });
}
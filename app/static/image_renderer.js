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

// --------------------------------------------------------



function draw_keypoints(color, radius){

    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    pctx.fillStyle = color;

    const keypoints = state.prediction.keypoints[0];

    for(let i = 0; i < keypoints.length; i++){
        pctx.beginPath();
        pctx.arc(
            keypoints[i][0] * scale + x,
            keypoints[i][1] * scale + y,
            radius,
            0,
            Math.PI * 2
        );
        pctx.fill();
    }
}

function draw_bounding_box(color, thickness){

    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    pctx.fillStyle = color;

    // bounding box in the form xywh
    const [cx, cy, w, h] = state.prediction.bounding_boxes[0];

    // top left corner position
    const ul_x = (cx - w / 2) * scale + x;
    const ul_y = (cy - h / 2) * scale + y;

    const width = w * scale;
    const height = h * scale;

    pctx.strokeStyle = color;
    pctx.lineWidth = thickness;

    // draw rectangle
    pctx.strokeRect(ul_x, ul_y, width, height);
}

function draw_play_area(color){

    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    // the last 4 keypoints are the ones for the play area
    const lower_keypoints = state.prediction.keypoints[0].slice(-4);

    // we have to draw the quadrilater that connects them
    pctx.beginPath();
    pctx.moveTo(lower_keypoints[0][0] * scale + x, lower_keypoints[0][1] * scale + y);
    for (let i = 1; i < lower_keypoints.length; i++) {
        pctx.lineTo(lower_keypoints[i][0] * scale + x, lower_keypoints[i][1] * scale + y);
    }
    pctx.closePath();

    pctx.fillStyle = color;
    pctx.fill();
}

function draw_edge(A, B, color, thickness){
    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    // coordinate scalate e traslate
    const x1 = A[0] * scale + x;
    const y1 = A[1] * scale + y;
    const x2 = B[0] * scale + x;
    const y2 = B[1] * scale + y;

    pctx.beginPath();
    pctx.moveTo(x1, y1);
    pctx.lineTo(x2, y2);
    pctx.strokeStyle = color;
    pctx.lineWidth = thickness;
    pctx.stroke();
}

function draw_edges(thickness){
    const kps = state.prediction.keypoints[0];
    const x_axis_color = "red";
    const y_axis_color = "rgb(0, 255, 0)";
    const z_axis_color = "blue";

    draw_edge(kps[0], kps[1], x_axis_color, thickness);
    draw_edge(kps[2], kps[3], x_axis_color, thickness);
    draw_edge(kps[1], kps[2], y_axis_color, thickness);
    draw_edge(kps[0], kps[3], y_axis_color, thickness);

    draw_edge(kps[4], kps[5], x_axis_color, thickness);
    draw_edge(kps[6], kps[7], x_axis_color, thickness);
    draw_edge(kps[5], kps[6], y_axis_color, thickness);
    draw_edge(kps[4], kps[7], y_axis_color, thickness);

    draw_edge(kps[0], kps[4], z_axis_color, thickness);
    draw_edge(kps[1], kps[5], z_axis_color, thickness);
    draw_edge(kps[2], kps[6], z_axis_color, thickness);
    draw_edge(kps[3], kps[7], z_axis_color, thickness);
}

function draw_goalnets(color, thickness){
    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;

    
}

function draw_player_lines(color, thickness){
    const x = state.image_x;
    const y = state.image_y;
    const scale = state.image_scale;
}

// --------------------------------------------------------

export function draw(){
    
    // clear the canvas
    pctx.fillStyle = "rgb(230, 230, 230)";
    pctx.fillRect(0, 0, photo_canvas.width, photo_canvas.height);

    // draw the photo on the canvas
    pctx.drawImage(
        state.photo,
        state.image_x,
        state.image_y,
        state.photo.width * state.image_scale,
        state.photo.height * state.image_scale
    );

    // show play area
    if(state.show_play_area){
        draw_play_area("rgba(255, 0, 255, 0.4)");
    }
    
    // show edges
    if(state.show_edges){
        draw_edges(3);
    }

    // show goalnets
    if(state.show_goalnets){
        draw_goalnets("yellow", 3);
    }

    // show player lines
    if(state.show_player_lines){
        draw_player_lines("gray", 3);
    }

    // show bounding box
    if(state.show_bounding_box){
        draw_bounding_box("red", 3);
    }

    // show keypoints
    if(state.show_keypoints){
        draw_keypoints("red", 5);
    }
}

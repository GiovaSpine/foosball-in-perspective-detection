import { photo_canvas, state } from "./predict_script.js";

export function page_pos_to_canvas_pos(x, y){
  // converts a postion relative to the page to a position relative to the canvas
  // it can't return null (because if the canvas was clicked in a certain position, it was clicked on the canvas)
  const rect = photo_canvas.getBoundingClientRect();

  // coordinates of x, y inside the canvas
  // because rect.top considers the viewport, we have to add window.scrollY (to obtain absolute position in the page)
  const cursur_x_canvas = x - rect.left;
  const cursur_y_canvas = y - (rect.top + window.scrollY);

  return [cursur_x_canvas, cursur_y_canvas];
}

export function canvas_pos_to_image_pos(x, y, return_anyway=false){
  // converts a position relative to canvas to a position relative to the image
  // returns null if the canvas position is outside the image and return_anyway = false
  // if return_anyway = true return the position regardless

  const cursur_x_image = (x - state.image_x) / state.image_scale;
  const cursur_y_image = (y - state.image_y) / state.image_scale;

  if(return_anyway) return [cursur_x_image, cursur_y_image];

  if(cursur_x_image < 0 || cursur_x_image >= state.photo.width || cursur_y_image < 0 || cursur_y_image >= state.photo.height) return null;
  else return [cursur_x_image, cursur_y_image];
}

export function is_image_clicked(x, y){
  const [cursur_x_canvas, cursur_y_canvas] = page_pos_to_canvas_pos(x, y);

  // x_inside = true when the x position for the cursur was inside the image during click
  // (the same for y)
  let x_inside, y_inside;

  // am i clicking on the image ?
  if(state.image_x >= 0 && state.image_x + (state.photo.width * state.image_scale) - 1 <= photo_canvas.width - 1){
    // the image is inside the canvas
    if(cursur_x_canvas >= state.image_x && cursur_x_canvas <= state.image_x + (state.photo.width * state.image_scale)){
      x_inside = true;
    } else {
      x_inside = false;
    }
  } else {
    // the image is outside the canvas (we can't trust image_x or image_x + (photo.width * image_scale))
    if(cursur_x_canvas >= Math.max(state.image_x, 0) && cursur_x_canvas <= Math.min(state.image_x + (state.photo.width * state.image_scale), photo_canvas.width)){
      x_inside = true;
    } else {
      x_inside = false;
    }
  }


  if(state.image_y >= 0 && state.image_y + (state.photo.height * state.image_scale) - 1 <= photo_canvas.height - 1){
    // the image is inside the canvas
    if(cursur_y_canvas >= state.image_y && cursur_y_canvas <= state.image_y + (state.photo.height * state.image_scale)){
      y_inside = true;
    } else {
      y_inside = false;
    }
  } else {
    // the image is outside the canvas (we can't trust image_y or image_y + (photo.height * image_scale) )
    if(cursur_y_canvas >= Math.max(state.image_y, 0) && cursur_y_canvas <= Math.min(state.image_y + (state.photo.height * state.image_scale), photo_canvas.height)){
      y_inside = true;
    } else {
      y_inside = false;
    }
  }
  
  return x_inside && y_inside;
}

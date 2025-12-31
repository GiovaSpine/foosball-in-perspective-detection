import { photo_canvas } from "./predict_script.js";
import { is_image_clicked } from "./utils.js";



export function wait_for_click(ignore_first_click = false) {
  return new Promise((resolve) => {
    let ignored = !ignore_first_click;

    function handler(e) {
      const x = e.pageX;
      const y = e.pageY;

      if (!ignored) {
        ignored = true;  // ignores the first click if requested
        return;
      }

      if (!is_image_clicked(x, y)) {
        console.log("Click ignored (outiside the image)");
        return;  // continue to listen if the click is outside the image
      }

      photo_canvas.removeEventListener("click", handler);
      resolve(e);  // return the position (you can choose which one (x, pageX, clientX))
    }

    photo_canvas.addEventListener("click", handler);
  });
}


export function wait_for_click_or_escape(ignore_first_click = false) {
  return new Promise((resolve) => {
    let ignored = !ignore_first_click;

    function click_handler(e) {
      const x = e.pageX;
      const y = e.pageY;

      if (!ignored) {
        ignored = true;
        return;
      }

      if (!is_image_clicked(x, y)) {
        console.log("Click ignored (outside the image)");
        return;
      }

      cleanup();
      resolve(e);
    }

    function key_handler(e) {
      if (e.key === "Escape") {
        cleanup();
        resolve(null);
      }
    }

    function cleanup() {
      photo_canvas.removeEventListener("click", click_handler);
      window.removeEventListener("keydown", key_handler);
    }

    photo_canvas.addEventListener("click", click_handler);
    window.addEventListener("keydown", key_handler);
  });
}
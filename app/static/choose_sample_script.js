console.log("Testing choose_sample_script");

window.choose_image = choose_image;

async function choose_image(image_url) {
  
  // load image from server
  const response = await fetch(image_url);
  const blob = await response.blob();

  const filename = image_url.split("/").pop();

  const file = new File([blob], filename, { type: blob.type });

  // save in browser session
  const reader = new FileReader();
  reader.onload = async () => {
    sessionStorage.setItem("photo", JSON.stringify({
        data: reader.result,
        name: file.name
    }));

    // send to backend
    const formData = new FormData();
    formData.append("photo", file);

    const response1 = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const prediction = await response1.json();

    // save prediction in session
    sessionStorage.setItem("prediction", JSON.stringify(prediction));

    // go to page prediction
    window.location.href = "/predict";
  };

  reader.readAsDataURL(file);
}
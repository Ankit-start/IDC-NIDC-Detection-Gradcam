const fileInput = document.getElementById("fileInput");

// Show uploaded patch immediately after selecting a file
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const uploadedImg = document.getElementById("uploadedImg");
    uploadedImg.src = URL.createObjectURL(file);
    uploadedImg.style.display = "block";
  }
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    // Show prediction text
    document.getElementById("result").innerHTML =
    `<div class="prediction-box">
        <strong>Prediction:</strong> ${data.label} 
        <span class="prob">(Prob: ${data.prob.toFixed(2)})</span>
     </div>`;

    // Show Grad-CAM overlay
    // Show Grad-CAM overlay (force refresh by adding timestamp)
    const overlayImg = document.getElementById("overlayImg");
    overlayImg.src = "http://127.0.0.1:5000" + data.overlay_url + "?t=" + new Date().getTime();
    overlayImg.style.display = "block";

  } catch (error) {
    console.error("Prediction error:", error);
    document.getElementById("result").innerText = "Error during prediction.";
  }
});
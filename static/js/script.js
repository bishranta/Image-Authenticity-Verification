document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const fileName = document.getElementById("file-name");
  const nextBtn = document.getElementById("next-btn");
  const uploadForm = document.getElementById("upload-form");
  const resultBox = document.getElementById("result-box");

  // Create an img element for thumbnail
  const thumb = document.createElement("img");
  thumb.style.maxWidth = "150px";
  thumb.style.maxHeight = "150px";
  thumb.style.marginTop = "10px";
  thumb.style.border = "1px solid #ccc";
  thumb.style.borderRadius = "5px";
  document.getElementById("upload-box").appendChild(thumb);

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileName.textContent = fileInput.files[0].name;
      nextBtn.disabled = false;

      // Show thumbnail preview
      const reader = new FileReader();
      reader.onload = (e) => {
        thumb.src = e.target.result;
      };
      reader.readAsDataURL(fileInput.files[0]);
    }
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultBox.textContent = "Processing...";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.error) {
        resultBox.textContent = "Error: " + result.error;
      } else {
        resultBox.innerHTML = `
          <strong>Prediction:</strong> ${result.pred_label}<br>
          <strong>Confidence:</strong> ${result.confidence}%
        `;
      }
    } catch (err) {
      resultBox.textContent = "An error occurred: " + err.message;
    }
  });
});

from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import gdown
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# -----------------------------
# Download + Load checkpoint
# -----------------------------
# FILE_ID = "1j0Y33f8HYS0rNUg8hU1MplWhQkgnNtlf"
# OUTPUT = "best_resnet50.pth"

# if not os.path.exists(OUTPUT):
#     url = f"https://drive.google.com/uc?id={FILE_ID}"
#     gdown.download(url, OUTPUT, quiet=False)
OUTPUT = os.path.join(os.path.dirname(__file__), "best_resnet50.pth")
# checkpoint = torch.load(OUTPUT, map_location="cpu")

checkpoint = torch.load(OUTPUT, map_location="cpu", weights_only=False)

# Define plain ResNet50 model (same as training)
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load checkpoint correctly
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        grads = self.gradients[0]
        activations = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = np.uint8(cam * 255)
        cam = Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR)
        return cam

# Grad-CAM target layer
target_layer = model.layer4
gradcam = GradCAM(model, target_layer)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------
# Flask API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    # Prediction
    output = model(input_tensor)
    prob = torch.softmax(output, dim=1)[0]
    pred_class = prob.argmax().item()
    label = "IDC" if pred_class == 1 else "NIDC"

    # Grad-CAM
    cam_mask = gradcam(input_tensor)

    # Overlay image
    fig, ax = plt.subplots()
    ax.imshow(img.resize((224,224)))
    ax.imshow(cam_mask, cmap='jet', alpha=0.4)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)

    overlay_path = os.path.join(os.path.dirname(__file__), "overlay.png")
    with open(overlay_path, "wb") as f:
        f.write(buf.read())

    return jsonify({
        "label": label,
        "prob": float(prob[pred_class]),
        "overlay_url": "/overlay"
    })


@app.route("/overlay", methods=["GET"])
def overlay():
    overlay_path = os.path.join(os.path.dirname(__file__), "overlay.png")
    return send_file(overlay_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
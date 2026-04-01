import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models import resnet50

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
        weights = grads.mean(dim=(1,2))
        cam = (weights[:, None, None] * activations).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = np.uint8(cam * 255)
        return cam

# -----------------------------
# Lazy model loader
# -----------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join("Backend", "best_resnet50.pth")  # adjust path if needed
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # IDC vs NIDC

    # Safe load with weights_only=False
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    gradcam = GradCAM(model, model.layer4)
    return model, gradcam

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("IDC vs NIDC Detection with Grad-CAM (ResNet50)")

uploaded_file = st.file_uploader("Upload histopathology image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    model, gradcam = load_model()
    img = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    # Prediction
    output = model(input_tensor)
    prob = torch.softmax(output, dim=1)[0]
    pred_class = prob.argmax().item()
    label = "IDC" if pred_class == 1 else "NIDC"

    # Grad-CAM
    cam_mask = gradcam(input_tensor)

    # Display results side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption=f"Prediction: {label} (Prob: {prob[pred_class]:.2f})")

    with col2:
        fig, ax = plt.subplots()
        ax.imshow(img.resize((224,224)))
        ax.imshow(cam_mask, cmap='jet', alpha=0.4)
        ax.axis('off')
        ax.set_title("Grad-CAM Overlay")
        st.pyplot(fig)

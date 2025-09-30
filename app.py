# app.py
import os
import io
from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import time

# ---------------------- Config ----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "runs", "casia2_resnet50", "best.pt")

ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
IMG_SIZE = 320

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload limit

# ---------------------- Model ----------------------
def build_resnet50_2class():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2-class output
    return model

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Extract state_dict if needed
    state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Build model
    model = build_resnet50_2class()

    # Remove 'module.' prefix if exists
    new_state = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state[name] = v

    # Load state dict
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model

print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded on", device)

# ---------------------- Transforms ----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------- Helpers ----------------------
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def generate_heatmap(img, pred_index):
    """Generate Grad-CAM heatmap overlay for the image."""
    model.eval()
    x = transform(img).unsqueeze(0).to(device)
    x.requires_grad = True

    # Forward pass
    logits = model(x)
    score = logits[:, pred_index]
    
    # Backward to get gradients
    model.zero_grad()
    score.backward(retain_graph=True)

    # Grab the gradients from last conv layer
    gradients = None
    activations = None
    for name, module in model.named_modules():
        if name == "layer4":
            # Register hooks dynamically
            def forward_hook(module, input, output):
                nonlocal activations
                activations = output
            def backward_hook(module, grad_in, grad_out):
                nonlocal gradients
                gradients = grad_out[0]
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    # Forward & backward again to trigger hooks
    logits = model(x)
    score = logits[:, pred_index]
    model.zero_grad()
    score.backward(retain_graph=True)

    grads = gradients.detach()
    acts = activations.detach()

    # Grad-CAM calculation
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0-1
    cam = (cam * 255).astype(np.uint8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    # Original image as BGR
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Heatmap
    heatmap_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    return overlay

# ---------------------- Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded/static images so they are accessible via URL."""
    return app.send_static_file(f"uploads/{filename}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error': 'File extension not allowed'}), 400

    try:
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)

        # ---------------- Classification ----------------
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_index = probs.argmax(dim=1).item()
            confidence = probs[0, pred_index].item()
            label = "tampered" if pred_index == 1 else "authentic"

        # ---------------- Heatmap ----------------
        overlay = generate_heatmap(img, pred_index)

        # Save to static folder
        save_dir = os.path.join("static", "uploads")
        os.makedirs(save_dir, exist_ok=True)
        orig_filename = "orig.png"
        heatmap_filename = "heatmap.png"
        orig_path = os.path.join(save_dir, "orig.png")
        heatmap_path = os.path.join(save_dir, "heatmap.png")
        img.save(orig_path)
        cv2.imwrite(heatmap_path, overlay)

        # Add timestamp to force reload
        timestamp = int(time.time() * 1000)

        return jsonify({
            'pred_label': label,
            'confidence': round(confidence, 4),
            'orig_img': url_for('static', filename=f'uploads/{orig_filename}') + f"?t={timestamp}",
            'heatmap_img': url_for('static', filename=f'uploads/{heatmap_filename}') + f"?t={timestamp}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ---------------------- Run ----------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

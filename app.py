# app.py
import os
import io
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

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

    checkpoint = torch.load(path, map_location='cpu')

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

# ---------------------- Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')

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

        with torch.no_grad():
            logits = model(x)
            if logits.shape[1] == 2:  # 2-class
                probs = torch.softmax(logits, dim=1)
                pred_index = probs.argmax(dim=1).item()
                confidence = probs[0, pred_index].item()
                label = "tampered" if pred_index == 1 else "authentic"
            else:  # fallback
                prob = torch.sigmoid(logits).item()
                label = "tampered" if prob >= 0.5 else "authentic"
                confidence = prob if label=="tampered" else 1-prob

        return jsonify({'pred_label': label, 'confidence': round(confidence, 4)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ---------------------- Run ----------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

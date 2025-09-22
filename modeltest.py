from PIL import Image
import torch
from torchvision import transforms
from app import build_resnet50_binary, load_model, device

model = load_model("D:\\Projects\\Image-Authenticity-Verification\\image_forgery\\runs\\casia2_resnet50\\best.pt")

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

img = Image.open("D:\\Bishranta\\chitwan\\100D7500\\DSC_1537.JPG").convert("RGB")
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)  # 2-class
    print("Probabilities:", probs)
    label = "tampered" if probs[0,1] > 0.5 else "authentic"
    print("Predicted label:", label)

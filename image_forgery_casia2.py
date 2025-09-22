#!/usr/bin/env python3
"""
Image Forgery (Tampering) Detection – CASIA v2 (or similar) Trainer
------------------------------------------------------------------

This script trains a binary classifier (authentic vs. tampered) for image forgery detection
using PyTorch. It is designed to work out‑of‑the‑box with CASIA v2 from the
"Image-Forgery-Datasets-List" you shared, but can also work with any dataset laid out as:

    dataset_root/
        train/
            authentic/
            tampered/
        val/
            authentic/
            tampered/
        test/
            authentic/
            tampered/

If you download CASIA v2 in its original layout (Au for authentic and Tp for tampered),
you can use the built‑in `--prepare-casia2` utility to reorganize it into the above split
(70/15/15 by default). The script also supports training with class weights, AMP, and
simple Grad-CAM‑like saliency for quick visual inspection.

Dependencies
------------
Python >=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (or cpu)
pip install opencv-python albumentations tqdm scikit-learn matplotlib

Quickstart
----------
# 1) Prepare CASIA v2 (from the raw CASIA v2 root that contains Au and Tp folders)
python image_forgery_casia2.py --prepare-casia2 \
    --casia2-root /path/to/CASIA2 \
    --output-root /path/to/datasets/casia2_splits \
    --val-ratio 0.15 --test-ratio 0.15 --seed 42

# 2) Train
python image_forgery_casia2.py --train \
    --data-root /path/to/datasets/casia2_splits \
    --model resnet50 --epochs 20 --batch-size 32 --lr 3e-4 \
    --img-size 320 --mixup 0.0 --cutmix 0.0 \
    --out-dir ./runs/casia2_resnet50

# 3) Evaluate on test split
python image_forgery_casia2.py --eval --ckpt ./runs/casia2_resnet50/best.pt --data-root /path/to/datasets/casia2_splits

# 4) Predict on a folder of images
python image_forgery_casia2.py --predict --ckpt ./runs/casia2_resnet50/best.pt --input /path/to/images_or_dir --output ./predictions.csv

Note: This is a solid baseline. For localization (tamper mask) tasks, consider extending the
DataSet to also read masks (available in some datasets like COVERAGE or Nimble) and train a
segmentation network (e.g., U-Net). 
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data
# -----------------------------

class ForgeryDataset(Dataset):
    """Generic folder‑based dataset.

    Expects directory layout with two class folders inside split folders:
      split/{authentic,tampered}/image.jpg
    """

    def __init__(self, root: Path, split: str, img_size: int = 320, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.samples = []  # list of (path, label)
        for cls_name, label in [("authentic", 0), ("tampered", 1)]:
            cls_dir = self.root / split / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.is_file() and is_image_file(p):
                    self.samples.append((p, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root}/{split}. Expected 'authentic' and 'tampered' subfolders.")

        # Albumentations pipeline
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1)
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
                    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                ], p=0.5),
                A.ColorJitter(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101),
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.tf(image=img)
        x = out['image']
        y = torch.tensor(label, dtype=torch.long)
        return x, y, str(img_path)


# -----------------------------
# Model
# -----------------------------

def build_model(name: str = "resnet50", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model: {name}")


# -----------------------------
# Training / Evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        n += x.size(0)
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}", "acc": f"{correct/n:.3f}"})
    return total_loss / n, correct / n


def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    ys = []
    ps = []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            n += x.size(0)
            ys.append(y.detach().cpu().numpy())
            ps.append(probs.detach().cpu().numpy())
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    auc = roc_auc_score(y_all, p_all)
    loss = total_loss / n if criterion is not None else float('nan')
    acc = correct / n
    return loss, acc, auc, y_all, p_all


def save_checkpoint(state: dict, path: Path):
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location=None):
    return torch.load(path, map_location=map_location)


# -----------------------------
# CASIA v2 preparation utility
# -----------------------------

def prepare_casia2(casia_root: Path, out_root: Path, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Convert the original CASIA v2 layout into train/val/test with authentic/tampered.

    Expected input layout (common):
        casia_root/
            Au/  # authentic (possibly with nested category folders)
            Tp/  # tampered (splicing/copy-move/etc.)
    """
    set_seed(seed)
    au_dir = casia_root / "Au"
    tp_dir = casia_root / "Tp"
    assert au_dir.exists() and tp_dir.exists(), "CASIA2 root must contain 'Au' and 'Tp' directories"

    def collect_images(root: Path) -> List[Path]:
        return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]

    au_imgs = collect_images(au_dir)
    tp_imgs = collect_images(tp_dir)

    random.shuffle(au_imgs)
    random.shuffle(tp_imgs)

    def split(lst: List[Path]):
        n = len(lst)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    au_train, au_val, au_test = split(au_imgs)
    tp_train, tp_val, tp_test = split(tp_imgs)

    for split_name, au_list, tp_list in [
        ("train", au_train, tp_train),
        ("val", au_val, tp_val),
        ("test", au_test, tp_test),
    ]:
        for cls in ["authentic", "tampered"]:
            ensure_dir(out_root / split_name / cls)
        for src in au_list:
            dst = out_root / split_name / "authentic" / src.name
            shutil.copy2(src, dst)
        for src in tp_list:
            dst = out_root / split_name / "tampered" / src.name
            shutil.copy2(src, dst)

    print(f"Prepared CASIA2 splits at: {out_root}")
    print(f"Counts -> train: au={len(au_train)}, tp={len(tp_train)} | val: au={len(au_val)}, tp={len(tp_val)} | test: au={len(au_test)}, tp={len(tp_test)}")


# -----------------------------
# Saliency (Grad-CAM-ish) for quick inspection
# -----------------------------
class SimpleGradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.features = None
        self.gradients = None
        self.hook_f = self.target_layer.register_forward_hook(self._forward_hook)
        self.hook_b = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.features = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        one_hot = torch.zeros_like(logits)
        one_hot[range(logits.size(0)), class_idx] = 1
        logits.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.features).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam

    def close(self):
        self.hook_f.remove()
        self.hook_b.remove()


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Image Forgery (Tampering) Detection – CASIA v2 Baseline")

    # Modes
    parser.add_argument('--prepare-casia2', action='store_true', help='Prepare CASIA v2 into train/val/test folders')
    parser.add_argument('--train', action='store_true', help='Train a classifier')
    parser.add_argument('--eval', action='store_true', help='Evaluate on test split')
    parser.add_argument('--predict', action='store_true', help='Predict on a folder or image file')

    # Data
    parser.add_argument('--casia2-root', type=str, default=None, help='Path to raw CASIA v2 root (contains Au/ and Tp/)')
    parser.add_argument('--output-root', type=str, default=None, help='Where to write prepared splits')
    parser.add_argument('--data-root', type=str, default=None, help='Root of prepared dataset with train/val/test splits')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    # Training params
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=320)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--class-weight', type=float, nargs=2, default=None, help='Weight for [authentic, tampered] (optional)')
    parser.add_argument('--out-dir', type=str, default='./runs/exp')

    # Inference
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint for eval/predict')
    parser.add_argument('--input', type=str, default=None, help='Path to image or folder for prediction')
    parser.add_argument('--output', type=str, default='./predictions.csv')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mode: prepare CASIA v2
    if args.prepare_casia2:
        assert args.casia2_root and args.output_root, "--casia2-root and --output-root are required"
        prepare_casia2(Path(args.casia2_root), Path(args.output_root), args.val_ratio, args.test_ratio, args.seed)
        return

    # Mode: train
    if args.train:
        assert args.data_root, "--data-root is required for training"
        data_root = Path(args.data_root)
        train_ds = ForgeryDataset(data_root, 'train', img_size=args.img_size, augment=True)
        val_ds = ForgeryDataset(data_root, 'val', img_size=args.img_size, augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.model, num_classes=2, pretrained=not args.no_pretrained).to(device)

        if args.class_weight is not None:
            cw = torch.tensor(args.class_weight, dtype=torch.float32, device=device)
        else:
            cw = None
        criterion = nn.CrossEntropyLoss(weight=cw)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        best_acc = 0.0
        out_dir = Path(args.out_dir)
        ensure_dir(out_dir)

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
            val_loss, val_acc, val_auc, y_true, y_prob = evaluate(model, val_loader, device, criterion)
            print(f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f}")

            # Save latest
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, out_dir / 'last.pt')

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, out_dir / 'best.pt')
                print(f"Saved best checkpoint (acc={best_acc:.4f})")
        return

    # Mode: eval
    if args.eval:
        assert args.data_root and args.ckpt, "--data-root and --ckpt are required for eval"
        data_root = Path(args.data_root)
        test_ds = ForgeryDataset(data_root, 'test', img_size=args.img_size, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # Build model and load
        model = build_model(args.model, num_classes=2, pretrained=False).to(device)
        ckpt = load_checkpoint(Path(args.ckpt), map_location=device)
        model.load_state_dict(ckpt['model'])

        criterion = nn.CrossEntropyLoss()
        loss, acc, auc, y_true, y_prob = evaluate(model, test_loader, device, criterion)
        y_pred = (y_prob >= 0.5).astype(int)
        print(f"Test: loss={loss:.4f} acc={acc:.4f} auc={auc:.4f}")
        print("Classification report:\n", classification_report(y_true, y_pred, target_names=["authentic", "tampered"]))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        return

    # Mode: predict
    if args.predict:
        assert args.ckpt and args.input, "--ckpt and --input are required for prediction"
        input_path = Path(args.input)
        model = build_model(args.model, num_classes=2, pretrained=False).to(device)
        ckpt = load_checkpoint(Path(args.ckpt), map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        tf = A.Compose([
            A.LongestMaxSize(max_size=args.img_size),
            A.PadIfNeeded(min_height=args.img_size, min_width=args.img_size, border_mode=cv2.BORDER_REFLECT_101),
            A.Normalize(),
            ToTensorV2(),
        ])

        imgs = []
        if input_path.is_dir():
            for p in sorted(input_path.rglob('*')):
                if p.is_file() and is_image_file(p):
                    imgs.append(p)
        elif input_path.is_file() and is_image_file(input_path):
            imgs.append(input_path)
        else:
            raise RuntimeError("--input must be an image file or a directory containing images")

        rows = ["path,prob_tampered,pred_label"]
        for p in tqdm(imgs, desc="predict"):
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = tf(image=img)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), dim=1)[0, 1].item()
            pred = int(prob >= 0.5)
            rows.append(f"{p},{prob:.6f},{pred}")
        out_csv = Path(args.output)
        ensure_dir(out_csv.parent)
        out_csv.write_text("\n".join(rows))
        print(f"Wrote predictions to {out_csv}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

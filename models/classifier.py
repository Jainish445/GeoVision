"""
models/classifier.py — Fine-grained warship type classification.

After YOLOv8 detects a ship, this module crops the detection region
and classifies it into one of the WARSHIP_CLASSES using a ResNet-50
backbone fine-tuned on naval vessel imagery.

Training data recommendation:
  • HRSC2016 dataset (High-Resolution Ship Collections)
  • Military ship datasets from Roboflow Universe
  • xView dataset (subset: warship classes)

Usage:
    from models.classifier import WarshipClassifier
    clf = WarshipClassifier()
    detections = clf.classify_detections(detections, scene_image)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLASSIFIER_WEIGHTS, WARSHIP_CLASSES,
    CLASSIFIER_IMG_SIZE, CLASSIFIER_CONF_MIN,
)
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("classifier")

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Model Definition ───────────────────────────────────────────────────────────

def build_classifier(num_classes: int, pretrained: bool = True) -> "nn.Module":
    """
    Build a ResNet-50 classifier with a custom head for warship type prediction.
    The backbone is frozen for the first few epochs (fine-tuning regime).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed. Run: pip install torch torchvision")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT if pretrained else None
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    # Unfreeze last residual block for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model


# ── Preprocessing Transform ────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_inference_transform(img_size: int = CLASSIFIER_IMG_SIZE):
    if not HAS_TORCH:
        return None
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

def get_train_transform(img_size: int = CLASSIFIER_IMG_SIZE):
    if not HAS_TORCH:
        return None
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── Classifier ─────────────────────────────────────────────────────────────────

class WarshipClassifier:
    """
    Crop-and-classify warship type from satellite image detections.

    Falls back to "Unknown Warship" with zero confidence if:
      • Weights file is not found.
      • PyTorch / PIL are not installed.
      • The crop region is too small (<20×20 pixels).
    """

    MIN_CROP_PX = 20    # minimum bbox size to attempt classification

    def __init__(
        self,
        weights:       Optional[Path] = None,
        device:        str = "cpu",
        conf_min:      float = CLASSIFIER_CONF_MIN,
        classes:       List[str] = WARSHIP_CLASSES,
    ):
        self.weights   = weights or CLASSIFIER_WEIGHTS
        self.device    = torch.device(device) if HAS_TORCH else None
        self.conf_min  = conf_min
        self.classes   = classes
        self._model    = None
        self._transform = get_inference_transform()

    def _load_model(self):
        if self._model is not None:
            return

        if not HAS_TORCH:
            log.warning("PyTorch not installed — skipping classification.")
            return

        model = build_classifier(num_classes=len(self.classes), pretrained=False)

        if self.weights.exists():
            log.info("Loading classifier weights: %s", self.weights)
            state = torch.load(str(self.weights), map_location=self.device)
            model.load_state_dict(state)
        else:
            log.warning(
                "Classifier weights not found at %s — "
                "all ships will be tagged as 'Unknown Warship'.",
                self.weights,
            )

        model.eval()
        model.to(self.device)
        self._model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify_crop(
        self, crop: "Image.Image"
    ) -> Tuple[str, float]:
        """
        Classify a single PIL crop image.

        Returns:
            (warship_type, confidence)
        """
        self._load_model()

        if self._model is None or not HAS_TORCH or not HAS_PIL:
            return "Unknown Warship", 0.0

        tensor = self._transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        conf, idx = probs.max(0)
        conf_val  = conf.item()
        cls_name  = self.classes[idx.item()]

        if conf_val < self.conf_min:
            return "Unknown Warship", conf_val

        return cls_name, conf_val

    def classify_detections(
        self,
        detections: List[Detection],
        scene_image: Optional[np.ndarray],     # Full H×W×3 scene array
        tile_size: int = 640,
        padding: int = 10,
    ) -> List[Detection]:
        """
        For each detection, crop its bounding box from the scene image
        and classify the warship type.

        Args:
            detections:   Detections from the detector (with bbox_px).
            scene_image:  The full satellite image tile as H×W×3 numpy array.
            tile_size:    Tile size (default 640px).
            padding:      Extra context pixels around the bbox.

        Returns:
            Detections with warship_type and warship_conf populated.
        """
        self._load_model()

        if scene_image is None:
            log.warning("No scene image provided — skipping classifier.")
            for d in detections:
                d.warship_type = "Unknown Warship"
                d.warship_conf = 0.0
            return detections

        if not HAS_PIL:
            log.warning("Pillow not installed — skipping classification.")
            return detections

        H, W = scene_image.shape[:2]

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox_px]

            # Expand bbox with padding
            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(W, x2 + padding)
            y2p = min(H, y2 + padding)

            if (x2p - x1p) < self.MIN_CROP_PX or (y2p - y1p) < self.MIN_CROP_PX:
                d.warship_type = "Unknown Warship"
                d.warship_conf = 0.0
                continue

            crop_arr = scene_image[y1p:y2p, x1p:x2p]
            crop_img = Image.fromarray(crop_arr.astype(np.uint8), mode="RGB")
            wtype, wconf = self.classify_crop(crop_img)
            d.warship_type = wtype
            d.warship_conf = round(wconf, 4)
            log.debug("Track %s → %s (%.2f)", d.track_id, wtype, wconf)

        return detections


# ── Training Script ────────────────────────────────────────────────────────────

class WarshipDataset:
    """
    Simple folder-based dataset: data/classifier/<class_name>/<image.jpg>
    """

    def __init__(self, root: Path, transform=None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training.")

        from torch.utils.data import Dataset

        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for idx, cls_name in enumerate(WARSHIP_CLASSES):
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.[jp][pn]g"):
                self.samples.append((img_path, idx))

        log.info("Dataset: %d samples, %d classes.", len(self.samples),
                 len(WARSHIP_CLASSES))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from torch.utils.data import Dataset
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test: classify a single image file
    import argparse

    parser = argparse.ArgumentParser(description="Classify a warship crop image")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Pillow required: pip install Pillow")
        sys.exit(1)

    clf  = WarshipClassifier(device=args.device)
    crop = Image.open(args.image_path).convert("RGB")
    wtype, conf = clf.classify_crop(crop)
    print(f"Classification: {wtype}  (confidence: {conf:.3f})")

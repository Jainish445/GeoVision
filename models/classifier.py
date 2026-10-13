"""
models/classifier.py — Warship type classification.

After YOLOv8 detects a ship, this module crops the detection from its tile
and assigns one of the WARSHIP_CLASSES. Two methods are available:

  1. CNN   — ResNet-50 fine-tuned on naval vessel crops. Used when trained
             weights exist at CLASSIFIER_WEIGHTS *and* the imagery is sharp
             enough for hull detail to be visible (≤ CLASSIFIER_CNN_MAX_GSD_M
             metres per pixel, i.e. commercial / aerial imagery).
  2. Size  — used otherwise (no weights, or coarse imagery such as Sentinel's
             10 m pixels). It does NOT identify the ship: the detection is
             labelled "Unidentified Vessel" and gets a *size class* (which
             warship type has a hull this size). A warship type is only
             asserted later if AIS backs it (see geo/ais_fusion.py).
             The hull is segmented from the
             crop, its length and beam are measured in metres (using the
             tile's ground sample distance) and mapped to a class by typical
             hull dimensions. This is a coarse estimate: it cannot tell a
             warship from a merchant ship of the same size (AIS fusion flags
             those afterwards), and never predicts "Submarine (surfaced)".

Training data recommendation for the CNN:
  • HRSC2016 dataset (High-Resolution Ship Collections)
  • Military ship datasets from Roboflow Universe
  • xView dataset (subset: warship classes)

Usage:
    from models.classifier import WarshipClassifier
    clf = WarshipClassifier()
    detections = clf.classify_detections(detections)

    # Train the CNN (folder layout: <data>/<class name>/<image>.jpg):
    python models/classifier.py train --data data/classifier/train \
        --val data/classifier/test --epochs 12

    # Per-class accuracy of the trained weights on a labelled folder:
    python models/classifier.py eval --data data/classifier/test

    # Classify a single crop:
    python models/classifier.py predict crop.png --gsd 10
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLASSIFIER_WEIGHTS, WARSHIP_CLASSES,
    CLASSIFIER_IMG_SIZE, CLASSIFIER_CONF_MIN,
    CLASSIFIER_CLASSES, CLASSIFIER_CNN_MAX_GSD_M, UNIDENTIFIED_LABEL,
    SIZE_CLASS_BANDS, SIZE_WIDE_BEAM_M, DEFAULT_GSD_M,
)
from models.detector import Detection, _geo_distance_km
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

UNKNOWN = "Unknown Warship"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Model Definition ───────────────────────────────────────────────────────────

def build_classifier(num_classes: int, pretrained: bool = True) -> "nn.Module":
    """
    Build a ResNet-50 classifier with a custom head for warship type prediction.
    The backbone is frozen except for the last residual block (fine-tuning regime).
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


# ── Size-based fallback ────────────────────────────────────────────────────────

def measure_ship(crop: np.ndarray, gsd_m: float) -> Optional[Tuple[float, float]]:
    """
    Measure hull length and beam (metres) of the ship in a crop.

    The ship is segmented as the connected region nearest the crop centre that
    stands out from the background (Otsu threshold on brightness contrast,
    floored at 3σ of the background so speckle is ignored; works for bright SAR
    returns and for light or dark hulls in optical imagery). Length is the extent along the hull's principal axis; beam is
    the segmented area divided by that length.

    Returns None when nothing ship-like can be segmented.
    """
    from scipy import ndimage

    gray = crop.max(axis=2).astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return None

    # Background statistics from the crop's border ring (robust to outliers)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    sigma = 1.4826 * float(np.median(np.abs(border - bg)))

    # Is the hull brighter (SAR, light hulls) or darker than the water?
    centre = gray[h // 3: h - h // 3, w // 3: w - w // 3]
    hi_dev, lo_dev = float(centre.max()) - bg, bg - float(centre.min())
    contrast = (gray - bg) if hi_dev >= lo_dev else (bg - gray)
    contrast = np.clip(contrast, 0, None)
    if contrast.max() < 8:          # flat crop — nothing to segment
        return None

    # Otsu threshold on the contrast image …
    hist, edges = np.histogram(contrast, bins=64)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (m0[-1] - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = 0
    # … but never below 3σ of the background, so speckle isn't segmented
    threshold = max(float(mids[int(np.argmax(between))]), 3.0 * sigma)
    # Real radar returns are surrounded by a glow (blur + sidelobes) that Otsu
    # counts as hull. Tighten to half the hull's own peak contrast.
    peak = float(np.percentile(contrast[contrast > threshold], 90))         if (contrast > threshold).any() else 0.0
    threshold = max(threshold, 0.5 * peak)
    mask = contrast > threshold
    if not mask.any():
        return None

    # 8-connectivity: thin diagonal hulls are only joined corner-to-corner
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return None

    # Pick the component closest to the centre (ties → larger area)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    best, best_key = None, None
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = np.nonzero(labels[sl] == i)
        if ys.size < 2:
            continue
        ys, xs = ys + sl[0].start, xs + sl[1].start
        dist = np.min(np.hypot(ys - cy, xs - cx))
        key = (round(dist / 3), -ys.size)
        if best_key is None or key < best_key:
            best, best_key = (ys, xs), key
    if best is None:
        return None

    ys, xs = best
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    if len(pts) >= 3:
        # Principal axes of the pixel cloud = hull orientation
        _, vecs = np.linalg.eigh(np.cov(pts.T))
        proj = pts @ vecs
    else:
        proj = pts
    ext = proj.max(axis=0) - proj.min(axis=0) + 1.0    # +1: pixel centres → edges
    length_px = float(ext.max())
    # Beam from area / length: the raw cross-axis extent over-reads by about a
    # pixel on rotated hulls, which is a large error for a 2-pixel-wide ship.
    beam_px = min(float(ext.min()), len(pts) / length_px)
    return length_px * gsd_m, beam_px * gsd_m


def classify_by_size(length_m: float, beam_m: float) -> Tuple[str, float]:
    """
    Map measured hull dimensions to a warship class.

    Confidence reflects how centrally the length sits inside its class band
    and is capped at 0.75 — size alone is never conclusive.
    """
    for name, lo, hi in SIZE_CLASS_BANDS:
        if lo <= length_m < hi:
            # 0 at a band edge, 1 in the middle
            centrality = 1.0 - abs((length_m - (lo + hi) / 2) / ((hi - lo) / 2))
            conf = 0.45 + 0.30 * centrality

            # Cruisers vs. amphibious ships overlap in length; beam separates them
            if name == "Cruiser" and beam_m >= SIZE_WIDE_BEAM_M:
                name, conf = "Amphibious Assault", min(conf, 0.5)
            elif name == "Amphibious Assault" and beam_m < SIZE_WIDE_BEAM_M * 0.6:
                name, conf = "Cruiser", min(conf, 0.5)
            return name, round(conf, 4)

    # Too small to be a ship, or bigger than any warship
    return UNKNOWN, 0.0


def tile_gsd_m(geo: Optional[dict], default: float = DEFAULT_GSD_M) -> float:
    """Ground sample distance (metres/pixel) of a tile from its sidecar JSON."""
    if not geo:
        return default
    if geo.get("gsd_m"):
        return float(geo["gsd_m"])
    if all(k in geo for k in ("lat_tl", "lon_tl", "lat_br", "lon_br")):
        size = geo.get("tile_size", 640)
        diag_km = _geo_distance_km(geo["lat_tl"], geo["lon_tl"],
                                   geo["lat_br"], geo["lon_br"])
        gsd = float(diag_km) * 1000.0 / (size * 2 ** 0.5)
        if gsd > 0:
            return gsd
    return default


# ── Classifier ─────────────────────────────────────────────────────────────────

class WarshipClassifier:
    """
    Crop-and-classify warship type from satellite image detections.

    Uses the trained CNN on sharp imagery. On coarse imagery (or without
    weights) the detection is labelled "Unidentified Vessel" and given a size
    class from its measured hull — size alone is not treated as evidence of a
    warship type.
    """

    MIN_CROP_PX = 8     # minimum padded crop size to attempt classification

    def __init__(
        self,
        weights:       Optional[Path] = None,
        device:        str = "cpu",
        conf_min:      float = CLASSIFIER_CONF_MIN,
        classes:       List[str] = WARSHIP_CLASSES,
        cnn_max_gsd_m: float = CLASSIFIER_CNN_MAX_GSD_M,
    ):
        self.weights   = Path(weights or CLASSIFIER_WEIGHTS)
        self.conf_min  = conf_min
        self.cnn_max_gsd_m = cnn_max_gsd_m
        self.classes   = list(classes)
        self._model    = None
        self._loaded   = False
        self._transform = get_inference_transform()

        if HAS_TORCH:
            # Accept ultralytics-style device strings ("0", "0,1") too
            if device and device[0].isdigit():
                device = f"cuda:{device.split(',')[0]}"
            if device.startswith("cuda") and not torch.cuda.is_available():
                log.warning("CUDA not available — classifier running on CPU.")
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = None

    @property
    def method(self) -> str:
        """'cnn' if trained weights are loaded, else 'size'."""
        self._load_model()
        return "cnn" if self._model is not None else "size"

    def _load_model(self):
        if self._loaded:
            return
        self._loaded = True

        if not HAS_TORCH:
            log.warning("PyTorch not installed — using size-based classification.")
            return

        if not self.weights.exists():
            log.warning(
                "Classifier weights not found at %s — falling back to "
                "size-based classification (hull length / beam).",
                self.weights,
            )
            return

        log.info("Loading classifier weights: %s", self.weights)
        try:
            state = torch.load(str(self.weights), map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                # Checkpoint written by train_classifier()
                self.classes = list(state.get("classes", self.classes))
                state = state["state_dict"]
            model = build_classifier(num_classes=len(self.classes), pretrained=False)
            model.load_state_dict(state)
        except Exception as e:
            log.error("Could not load classifier weights (%s) — falling back "
                      "to size-based classification.", e)
            return

        model.eval()
        model.to(self.device)
        self._model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify_crop(
        self,
        crop: "Image.Image",
        gsd_m: float = DEFAULT_GSD_M,
    ) -> Tuple[str, float]:
        """
        Classify a single PIL crop image.

        Returns:
            (label, confidence) — on coarse imagery the label is
            "Unidentified Vessel" with confidence None; see _classify() for
            the size class.
        """
        r = self._classify(np.asarray(crop.convert("RGB")), gsd_m)
        return r["type"], r["conf"]

    def _classify(self, crop_arr: np.ndarray, gsd_m: float,
                  dims: Optional[Tuple[float, float]] = None) -> dict:
        """
        Classify one crop. Returns a dict with:
            type, conf      — the label the evidence supports
            source          — "cnn" or "size"
            size_class      — warship type with a hull this size (or None)
            size_conf       — how centrally the hull sits in that size band
            length_m, beam_m
        """
        self._load_model()

        # Dimensions from an oriented-box detector beat segmenting the crop
        if dims is None:
            dims = measure_ship(crop_arr, gsd_m)
        length_m, beam_m = dims if dims else (None, None)

        size_class, size_conf = None, None
        if dims is not None:
            name, conf = classify_by_size(length_m, beam_m)
            if name != UNKNOWN:
                size_class, size_conf = name, conf

        out = {"length_m": length_m, "beam_m": beam_m,
               "size_class": size_class, "size_conf": size_conf}

        if self._model is not None and gsd_m <= self.cnn_max_gsd_m:
            tensor = self._transform(Image.fromarray(crop_arr)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self._model(tensor), dim=1)[0]
            conf, idx = probs.max(0)
            conf_val = float(conf.item())
            wtype = self.classes[idx.item()] if conf_val >= self.conf_min else UNIDENTIFIED_LABEL
            return {**out, "type": wtype, "conf": conf_val, "source": "cnn"}

        # Hull size is not evidence of a warship: report the size class, but
        # leave the vessel unidentified until AIS or context says more.
        return {**out, "type": UNIDENTIFIED_LABEL, "conf": None, "source": "size"}

    def classify_detections(
        self,
        detections: List[Detection],
        scene_image: Optional[np.ndarray] = None,
        padding: int = 10,
        gsd_m: Optional[float] = None,
    ) -> List[Detection]:
        """
        For each detection, crop its bounding box and classify the warship type.

        Detection boxes are in the pixel coordinates of their own tile, so by
        default each detection is cropped from the PNG at `d.tile_path` (and
        the ground sample distance read from the tile's sidecar JSON).

        Args:
            detections:   Detections from the detector (with bbox_px).
            scene_image:  Optional H×W×3 array to crop from instead of the
                          tile files (all detections must share its pixel frame).
            padding:      Extra context pixels around the bbox.
            gsd_m:        Override metres/pixel (otherwise from the sidecar).

        Returns:
            Detections with warship_type, warship_conf, length_m and beam_m
            populated.
        """
        if not detections:
            return detections

        if not HAS_PIL:
            log.warning("Pillow not installed — skipping classification.")
            return detections

        self._load_model()

        # Group by tile so each PNG is read only once
        by_tile: Dict[Path, List[Detection]] = defaultdict(list)
        for d in detections:
            by_tile[Path(d.tile_path)].append(d)

        n_missing = 0
        for tile_path, dets in by_tile.items():
            if scene_image is not None:
                image, tile_gsd = scene_image, gsd_m or DEFAULT_GSD_M
            else:
                image, tile_gsd = self._load_tile(tile_path)
                if gsd_m:
                    tile_gsd = gsd_m

            for d in dets:
                if image is None:
                    n_missing += 1
                    d.warship_type, d.warship_conf = UNIDENTIFIED_LABEL, None
                    continue
                self._classify_one(d, image, tile_gsd, padding)

        if n_missing:
            log.warning("%d detection(s) left unclassified — tile image missing.",
                        n_missing)

        counts: Dict[str, int] = defaultdict(int)
        sizes: Dict[str, int] = defaultdict(int)
        for d in detections:
            counts[d.warship_type or UNIDENTIFIED_LABEL] += 1
            if d.size_class:
                sizes[d.size_class] += 1
        log.info("Classified %d detection(s): %s", len(detections),
                 ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if sizes:
            log.info("Size classes (hull dimensions only, not identification): %s",
                     ", ".join(f"{k}-sized={v}" for k, v in sorted(sizes.items())))
        return detections

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_tile(tile_path: Path) -> Tuple[Optional[np.ndarray], float]:
        """Read a tile PNG and its ground sample distance."""
        if not tile_path.exists():
            return None, DEFAULT_GSD_M
        try:
            image = np.asarray(Image.open(tile_path).convert("RGB"))
        except Exception as e:
            log.warning("Could not read tile %s: %s", tile_path, e)
            return None, DEFAULT_GSD_M

        geo = None
        sidecar = tile_path.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar) as f:
                geo = json.load(f)
        return image, tile_gsd_m(geo)

    def _classify_one(self, d: Detection, image: np.ndarray,
                      gsd_m: float, padding: int) -> None:
        H, W = image.shape[:2]
        x1, y1, x2, y2 = d.bbox_px

        # Expand bbox with padding
        x1p = max(0, int(np.floor(x1)) - padding)
        y1p = max(0, int(np.floor(y1)) - padding)
        x2p = min(W, int(np.ceil(x2)) + padding)
        y2p = min(H, int(np.ceil(y2)) + padding)

        if (x2p - x1p) < self.MIN_CROP_PX or (y2p - y1p) < self.MIN_CROP_PX:
            d.warship_type, d.warship_conf = UNIDENTIFIED_LABEL, None
            return

        crop = np.ascontiguousarray(image[y1p:y2p, x1p:x2p], dtype=np.uint8)
        known = (d.length_m, d.beam_m) if d.length_m and d.beam_m else None
        r = self._classify(crop, gsd_m, dims=known)
        d.warship_type = r["type"]
        d.warship_conf = round(r["conf"], 4) if r["conf"] is not None else None
        d.type_source  = r["source"]
        d.size_class   = r["size_class"]
        d.size_conf    = r["size_conf"]
        d.length_m = round(r["length_m"], 1) if r["length_m"] is not None else None
        d.beam_m   = round(r["beam_m"], 1) if r["beam_m"] is not None else None
        log.debug("Track %s → %s (size class %s)", d.track_id, d.warship_type,
                  d.size_class)


# ── Training ───────────────────────────────────────────────────────────────────

_TorchDataset = torch.utils.data.Dataset if HAS_TORCH else object


class WarshipDataset(_TorchDataset):
    """
    Simple folder-based dataset: <root>/<class_name>/<image>

    Class folders are matched to WARSHIP_CLASSES case-insensitively, ignoring
    punctuation and spacing ("patrol_corvette" == "Patrol / Corvette").
    """

    def __init__(self, root: Path, transform=None, classes: List[str] = CLASSIFIER_CLASSES):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training.")

        self.transform = transform
        self.classes = list(classes)
        self.samples: List[Tuple[Path, int]] = []

        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        index = {norm(c): i for i, c in enumerate(self.classes)}
        root = Path(root)
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            idx = index.get(norm(cls_dir.name))
            if idx is None:
                log.warning("Ignoring folder '%s' — not a known class.", cls_dir.name)
                continue
            for img_path in sorted(cls_dir.rglob("*")):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, idx))

        log.info("Dataset: %d samples across %d class(es).", len(self.samples),
                 len({i for _, i in self.samples}))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _resolve_device(device: str) -> "torch.device":
    if device and device[0].isdigit():
        device = f"cuda:{device.split(',')[0]}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available — using CPU.")
        device = "cpu"
    return torch.device(device)


def train_classifier(
    data_dir:   Path,
    val_dir:    Optional[Path] = None,
    epochs:     int = 20,
    batch_size: int = 32,
    lr:         float = 1e-3,
    val_split:  float = 0.2,
    device:     str = "cpu",
    pretrained: bool = True,
    output:     Path = CLASSIFIER_WEIGHTS,
    seed:       int = 42,
    classes:    List[str] = CLASSIFIER_CLASSES,
) -> Path:
    """
    Fine-tune the ResNet-50 classifier and save the best checkpoint.

    If val_dir is given it is used for validation; otherwise val_split of
    data_dir is held out. Classes are weighted inversely to their frequency.
    """
    if not (HAS_TORCH and HAS_PIL):
        raise ImportError("PyTorch, torchvision and Pillow are required for training.")
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(seed)
    full = WarshipDataset(data_dir, get_train_transform(), classes)
    if len(full) < 2:
        raise RuntimeError(
            f"No training images found in {data_dir}. Expected one folder per "
            f"class, e.g. {data_dir}/Destroyer/*.jpg. Classes: {classes}"
        )

    if val_dir is not None:
        train_ds = full
        val_ds = WarshipDataset(val_dir, get_inference_transform(), classes)
        train_labels = [label for _, label in full.samples]
        if len(val_ds) == 0:
            raise RuntimeError(f"No validation images found in {val_dir}.")
    else:
        # Train / val split over shuffled indices, each with its own transform
        order = np.random.default_rng(seed).permutation(len(full))
        n_val = max(1, int(len(full) * val_split))
        train_ds = Subset(full, order[n_val:].tolist())
        val_ds = Subset(WarshipDataset(data_dir, get_inference_transform(), classes),
                        order[:n_val].tolist())
        train_labels = [full.samples[i][1] for i in order[n_val:]]

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)

    dev = _resolve_device(device)
    model = build_classifier(len(classes), pretrained=pretrained).to(dev)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, total_steps=max(1, epochs * len(train_dl)))

    # Inverse-frequency class weights (classes with no samples get weight 0)
    counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    weights = weights / weights[weights > 0].mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(dev))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            sched.step()
            running += loss.item() * len(y)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        acc = correct / len(val_ds)
        log.info("Epoch %d/%d — train loss %.4f, val acc %.3f",
                 epoch, epochs, running / max(1, len(train_ds)), acc)

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": list(classes)}, str(output))

    log.info("Best val accuracy %.3f — weights saved to %s", best_acc, output)
    return output


def evaluate_classifier(
    data_dir: Path,
    weights:  Path = CLASSIFIER_WEIGHTS,
    device:   str = "cpu",
    batch_size: int = 32,
) -> dict:
    """
    Per-class accuracy and confusion matrix of trained weights on a labelled
    folder (same layout as for training).
    """
    from torch.utils.data import DataLoader

    clf = WarshipClassifier(weights=weights, device=device)
    if clf.method != "cnn":
        raise RuntimeError(f"No usable classifier weights at {weights}.")
    classes = clf.classes

    ds = WarshipDataset(data_dir, get_inference_transform(), classes)
    if len(ds) == 0:
        raise RuntimeError(f"No labelled images found in {data_dir}.")

    n = len(classes)
    confusion = np.zeros((n, n), dtype=int)       # rows = truth, cols = predicted
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=batch_size):
            pred = clf._model(x.to(clf.device)).argmax(1).cpu().numpy()
            for t, p_ in zip(y.numpy(), pred):
                confusion[t, p_] += 1

    support = confusion.sum(axis=1)
    per_class = {
        classes[i]: {"n": int(support[i]),
                     "accuracy": float(confusion[i, i] / support[i])}
        for i in range(n) if support[i]
    }
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "balanced_accuracy": float(np.mean([v["accuracy"] for v in per_class.values()])),
        "per_class": per_class,
        "classes": classes,
        "confusion": confusion.tolist(),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warship type classifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Classify a single ship crop image")
    p_pred.add_argument("image_path", type=Path)
    p_pred.add_argument("--gsd", type=float, default=DEFAULT_GSD_M,
                        help="Metres per pixel of the crop (default: 10 = Sentinel; "
                             "the CNN is only used at or below 4)")
    p_pred.add_argument("--device", default="cpu")

    p_train = sub.add_parser("train", help="Fine-tune the ResNet-50 classifier")
    p_train.add_argument("--data", type=Path, required=True,
                         help="Folder with one sub-folder of images per class")
    p_train.add_argument("--val", type=Path, default=None,
                         help="Separate validation folder (default: hold out 20%%)")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--no-pretrained", action="store_true",
                         help="Do not start from ImageNet weights")
    p_train.add_argument("--output", type=Path, default=CLASSIFIER_WEIGHTS)

    p_eval = sub.add_parser("eval", help="Per-class accuracy on a labelled folder")
    p_eval.add_argument("--data", type=Path, required=True)
    p_eval.add_argument("--weights", type=Path, default=CLASSIFIER_WEIGHTS)
    p_eval.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.cmd == "train":
        train_classifier(args.data, val_dir=args.val, epochs=args.epochs,
                         batch_size=args.batch, lr=args.lr, device=args.device,
                         pretrained=not args.no_pretrained, output=args.output)
    elif args.cmd == "eval":
        r = evaluate_classifier(args.data, weights=args.weights, device=args.device)
        print(f"\nOverall accuracy:  {r['accuracy']:.3f}")
        print(f"Balanced accuracy: {r['balanced_accuracy']:.3f}\n")
        print(f"{'Class':<22} {'n':>5} {'acc':>6}   most confused with")
        for i, name in enumerate(r["classes"]):
            if name not in r["per_class"]:
                continue
            row = list(r["confusion"][i])
            row[i] = -1
            j = int(np.argmax(row))
            other = f"{r['classes'][j]} ({row[j]})" if row[j] > 0 else "—"
            pc = r["per_class"][name]
            print(f"{name:<22} {pc['n']:>5} {pc['accuracy']:>6.2f}   {other}")
    else:
        if not HAS_PIL:
            print("Pillow required: pip install Pillow")
            sys.exit(1)
        clf  = WarshipClassifier(device=args.device)
        crop = np.asarray(Image.open(args.image_path).convert("RGB"))
        r = clf._classify(crop, args.gsd)
        conf = f"{r['conf']:.3f}" if r["conf"] is not None else "n/a"
        print(f"Classification: {r['type']}  (confidence: {conf}, method: {r['source']})")
        if r["length_m"] is not None:
            print(f"Measured hull:  {r['length_m']:.0f} m × {r['beam_m']:.0f} m")
        if r["source"] == "size":
            print(f"Size class:     {r['size_class'] or 'none'}"
                  "  (hull dimensions only — not an identification)")

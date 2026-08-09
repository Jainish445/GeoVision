"""
train/finetune.py — Fine-tune YOLOv8 on ship / warship detection datasets.

Supported datasets (any YOLO-format dataset with 'ship' class):
  • SAR-Ship (Sentinel-1 SAR, 43,000+ chips)  → https://github.com/CAESAR-Radi/SAR-Ship-Dataset
  • DOTA v1.5 (aerial, ship class subset)      → https://captain-whu.github.io/DOTA/
  • xView (satellite, ship class)              → http://xviewdataset.org/
  • Custom Roboflow export (YOLO format)

Usage:
    # Basic fine-tuning:
    python train/finetune.py --data data.yaml --epochs 80

    # Resume from checkpoint:
    python train/finetune.py --data data.yaml --resume runs/detect/exp/weights/last.pt

    # Export to ONNX after training:
    python train/finetune.py --data data.yaml --export onnx
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR0, TRAIN_LRF,
    TRAIN_IMG_SIZE, TRAIN_PATIENCE, TRAIN_WORKERS, TRAIN_DEVICE,
    YOLO_MODEL_SIZE, MODELS_DIR, RESULTS_DIR,
)
from utils.logger import get_logger

log = get_logger("finetune")


def build_data_yaml(
    train_dir: Path,
    val_dir:   Path,
    output:    Path,
) -> Path:
    """
    Write a YOLO-format data.yaml pointing to training/validation image dirs.
    Assumes YOLO annotation format (one .txt per image).
    """
    content = f"""# Warship Detection — YOLO Dataset Config
path: {train_dir.parent.resolve()}
train: {train_dir.name}
val:   {val_dir.name}

nc: 1        # number of classes
names:
  - ship     # single class; warship type added by classifier post-detection
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    log.info("data.yaml written to %s", output)
    return output


def train(args: argparse.Namespace):
    """Launch YOLOv8 fine-tuning with all configured hyperparameters."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load base model
    if args.resume and Path(args.resume).exists():
        log.info("Resuming from checkpoint: %s", args.resume)
        model = YOLO(args.resume)
    else:
        weights = f"{YOLO_MODEL_SIZE}.pt"
        log.info("Fine-tuning from pretrained: %s", weights)
        model = YOLO(weights)

    # Hyperparameter overrides (Ultralytics supports keyword args)
    train_kwargs = {
        "data":      args.data,
        "epochs":    args.epochs,
        "imgsz":     TRAIN_IMG_SIZE,
        "batch":     args.batch,
        "lr0":       TRAIN_LR0,
        "lrf":       TRAIN_LRF,
        "patience":  TRAIN_PATIENCE,
        "workers":   TRAIN_WORKERS,
        "device":    args.device,
        "project":   str(RESULTS_DIR / "runs"),
        "name":      "warship_detection",
        "exist_ok":  True,
        "augment":   True,
        "save":      True,
        "plots":     True,
        "val":       True,
        # Data augmentation for satellite imagery
        "hsv_h":     0.015,
        "hsv_s":     0.7,
        "hsv_v":     0.4,
        "degrees":   180.0,     # full rotation (ships have no fixed orientation)
        "translate": 0.1,
        "scale":     0.5,
        "shear":     0.0,
        "perspective": 0.0,
        "flipud":    0.5,       # flip vertically (top-down view)
        "fliplr":    0.5,
        "mosaic":    1.0,
        "mixup":     0.2,
        "copy_paste": 0.1,
    }

    log.info("Starting training with kwargs: %s", train_kwargs)
    results = model.train(**train_kwargs)
    log.info("Training complete.")

    # Save best weights to our models dir
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "warship_yolov8m.pt"
        import shutil
        shutil.copy2(best_weights, dest)
        log.info("Best weights copied to %s", dest)

    return results


def evaluate(args: argparse.Namespace):
    """Run validation metrics on the best saved model."""
    from ultralytics import YOLO

    weights = args.weights or str(MODELS_DIR / "warship_yolov8m.pt")
    if not Path(weights).exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    model = YOLO(weights)
    metrics = model.val(data=args.data, imgsz=TRAIN_IMG_SIZE, device=args.device)

    print("\n=== Validation Metrics ===")
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    return metrics


def export_model(args: argparse.Namespace):
    """Export the trained model to ONNX / TensorRT / CoreML etc."""
    from ultralytics import YOLO

    weights = args.weights or str(MODELS_DIR / "warship_yolov8m.pt")
    model   = YOLO(weights)
    path    = model.export(format=args.export, imgsz=TRAIN_IMG_SIZE)
    log.info("Model exported to %s", path)


def prepare_sar_ship_dataset(raw_dir: Path, output_dir: Path):
    """
    Convert SAR-Ship dataset (HRSC-like format) to YOLO annotation format.
    
    SAR-Ship structure:
        raw_dir/
          images/    *.png
          labels/    *.xml  (PASCAL VOC format)
    
    Output structure (YOLO format):
        output_dir/
          images/train/  *.png
          images/val/    *.png
          labels/train/  *.txt
          labels/val/    *.txt
    """
    import random
    import xml.etree.ElementTree as ET

    log.info("Preparing SAR-Ship dataset: %s → %s", raw_dir, output_dir)

    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"

    images = sorted(img_dir.glob("*.png"))
    random.shuffle(images)

    split = int(0.8 * len(images))
    splits = {"train": images[:split], "val": images[split:]}

    for split_name, split_images in splits.items():
        out_img = output_dir / "images" / split_name
        out_lbl = output_dir / "labels" / split_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            xml_path = lbl_dir / img_path.with_suffix(".xml").name
            if not xml_path.exists():
                continue

            # Parse PASCAL VOC XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size  = root.find("size")
            W = int(size.find("width").text)
            H = int(size.find("height").text)

            yolo_lines = []
            for obj in root.findall("object"):
                bnd = obj.find("bndbox")
                x1 = float(bnd.find("xmin").text)
                y1 = float(bnd.find("ymin").text)
                x2 = float(bnd.find("xmax").text)
                y2 = float(bnd.find("ymax").text)

                cx = (x1 + x2) / (2 * W)
                cy = (y1 + y2) / (2 * H)
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            import shutil
            shutil.copy2(img_path, out_img / img_path.name)
            (out_lbl / img_path.with_suffix(".txt").name).write_text(
                "\n".join(yolo_lines)
            )

    # Write data.yaml
    build_data_yaml(
        train_dir = output_dir / "images" / "train",
        val_dir   = output_dir / "images" / "val",
        output    = output_dir / "data.yaml",
    )
    log.info("Dataset preparation complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Warship Detection Fine-tuner")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="Fine-tune YOLOv8")
    p_train.add_argument("--data",    required=True, help="Path to data.yaml")
    p_train.add_argument("--epochs",  type=int, default=TRAIN_EPOCHS)
    p_train.add_argument("--batch",   type=int, default=TRAIN_BATCH_SIZE)
    p_train.add_argument("--device",  default=TRAIN_DEVICE)
    p_train.add_argument("--resume",  help="Resume from .pt checkpoint")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate saved model")
    p_eval.add_argument("--data",    required=True)
    p_eval.add_argument("--weights", default=None)
    p_eval.add_argument("--device",  default=TRAIN_DEVICE)

    # export
    p_exp = sub.add_parser("export", help="Export model")
    p_exp.add_argument("--export",  default="onnx",
                       choices=["onnx", "torchscript", "tflite", "coreml"])
    p_exp.add_argument("--weights", default=None)
    p_exp.add_argument("--device",  default="cpu")

    # prepare
    p_prep = sub.add_parser("prepare", help="Prepare SAR-Ship dataset")
    p_prep.add_argument("--raw-dir",    type=Path, required=True)
    p_prep.add_argument("--output-dir", type=Path, default=Path("data/dataset"))

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "export":
        export_model(args)
    elif args.cmd == "prepare":
        prepare_sar_ship_dataset(args.raw_dir, args.output_dir)
    else:
        parser.print_help()

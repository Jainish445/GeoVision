"""
train/finetune.py — Fine-tune YOLOv8 on ship / warship detection datasets.

Supported datasets (any YOLO-format dataset with 'ship' class):
  • SAR-Ship (Sentinel-1 SAR, 43,000+ chips)  → https://github.com/CAESAR-Radi/SAR-Ship-Dataset
  • DOTA v1.5 (aerial, ship class subset)      → https://captain-whu.github.io/DOTA/
  • xView (satellite, ship class)              → http://xviewdataset.org/
  • Custom Roboflow export (YOLO format)

Usage:
    # Split the raw SAR-Ship dataset into train/val:
    python train/finetune.py prepare --raw-dir data/SAR-Ship-Dataset --output-dir data/dataset

    # Basic fine-tuning:
    python train/finetune.py train --data data/dataset/data.yaml --epochs 80

    # Resume from checkpoint:
    python train/finetune.py train --data data/dataset/data.yaml --resume results/runs/warship_detection/weights/last.pt

    # Export to ONNX after training:
    python train/finetune.py export --export onnx
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR0, TRAIN_LRF,
    TRAIN_IMG_SIZE, TRAIN_PATIENCE, TRAIN_WORKERS, TRAIN_DEVICE,
    YOLO_BASE_WEIGHTS, YOLO_WEIGHTS, MODELS_DIR, RESULTS_DIR,
)
from utils.logger import get_logger

log = get_logger("finetune")


def resolve_device(device: str) -> str:
    """Map "auto" to GPU 0 when CUDA is available, and never request a
    GPU that isn't there (ultralytics raises on an invalid device)."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if device == "auto":
        return "0" if has_cuda else "cpu"
    if device != "cpu" and device != "mps" and not has_cuda:
        log.warning("CUDA device '%s' requested but not available — using CPU.",
                    device)
        return "cpu"
    return device


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
        if args.resume:
            log.warning("Checkpoint %s not found — starting from pretrained.",
                        args.resume)
        # Absolute path: downloaded here if missing, independent of the cwd
        weights = str(YOLO_BASE_WEIGHTS)
        log.info("Fine-tuning from pretrained: %s", weights)
        model = YOLO(weights)

    device = resolve_device(args.device)

    # Hyperparameter overrides (Ultralytics supports keyword args)
    train_kwargs = {
        "data":      str(Path(args.data).resolve()),
        "epochs":    args.epochs,
        "imgsz":     args.imgsz,
        "batch":     args.batch,
        "fraction":  args.fraction,
        "lr0":       TRAIN_LR0,
        "lrf":       TRAIN_LRF,
        "patience":  TRAIN_PATIENCE,
        "workers":   args.workers,
        "device":    device,
        "project":   str(RESULTS_DIR / "runs"),
        "name":      "warship_detection",
        "exist_ok":  True,
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
        dest = YOLO_WEIGHTS
        import shutil
        shutil.copy2(best_weights, dest)
        log.info("Best weights copied to %s", dest)

    return results


def evaluate(args: argparse.Namespace):
    """Run validation metrics on the best saved model."""
    from ultralytics import YOLO

    weights = args.weights or str(YOLO_WEIGHTS)
    if not Path(weights).exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    model = YOLO(weights)
    metrics = model.val(data=str(Path(args.data).resolve()), imgsz=args.imgsz,
                        device=resolve_device(args.device))

    print("\n=== Validation Metrics ===")
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    return metrics


def export_model(args: argparse.Namespace):
    """Export the trained model to ONNX / TensorRT / CoreML etc."""
    from ultralytics import YOLO

    weights = args.weights or str(YOLO_WEIGHTS)
    if not Path(weights).exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)
    model   = YOLO(weights)
    path    = model.export(format=args.export, imgsz=TRAIN_IMG_SIZE,
                           device=resolve_device(args.device))
    log.info("Model exported to %s", path)


def prepare_sar_ship_dataset(raw_dir: Path, output_dir: Path, seed: int = 42):
    """
    Split the SAR-Ship dataset into YOLO train/val folders.

    Accepts either layout:
      • YOLO labels  — *.jpg / *.png with a same-named *.txt (the layout of
                       https://github.com/CAESAR-Radi/SAR-Ship-Dataset), either
                       flat in raw_dir or under images/ + labels/.
      • PASCAL VOC   — images under JPEGImages/ (or images/) with *.xml under
                       Annotations/ (or labels/).

    Output structure (YOLO format):
        output_dir/
          images/train/  images/val/
          labels/train/  labels/val/
          data.yaml
    """
    from data.dataset import DatasetManager

    log.info("Preparing SAR-Ship dataset: %s → %s", raw_dir, output_dir)
    DatasetManager(output_dir).prepare_sar_ship(raw_dir, output_dir, seed=seed)
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
    p_train.add_argument("--imgsz",   type=int, default=TRAIN_IMG_SIZE,
                         help="Training image size. SAR-Ship chips are 256 px at "
                              "Sentinel-1 resolution, so 256 keeps ships at the "
                              "scale they have in pipeline tiles (and is ~6× faster)")
    p_train.add_argument("--fraction", type=float, default=1.0,
                         help="Fraction of the training set to use (e.g. 0.2 on CPU)")
    p_train.add_argument("--workers", type=int, default=TRAIN_WORKERS)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate saved model")
    p_eval.add_argument("--data",    required=True)
    p_eval.add_argument("--weights", default=None)
    p_eval.add_argument("--device",  default=TRAIN_DEVICE)
    p_eval.add_argument("--imgsz",   type=int, default=TRAIN_IMG_SIZE)

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

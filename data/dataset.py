"""
data/dataset.py — Dataset download and management utilities.

Handles:
  • SAR-Ship dataset (Sentinel-1 chips, PASCAL VOC labels)
  • HRSC2016 (optical, fine-grained warship types)
  • Roboflow project download (YOLO format)
  • Dataset statistics and class distribution

Usage:
    from data.dataset import DatasetManager
    dm = DatasetManager()
    dm.download_roboflow("your_workspace/warship-detect/3", api_key="xxx")
    dm.stats("data/dataset")
"""

import os
import sys
import json
import shutil
import zipfile
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

log = get_logger("dataset")

# Default output directory for all dataset operations
DEFAULT_DATASET_DIR = Path("data/dataset")


class DatasetManager:
    """
    Utilities for downloading, validating, and preparing datasets
    for YOLOv8 training.
    """

    def __init__(self, dataset_dir: Path = DEFAULT_DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # ── Roboflow ───────────────────────────────────────────────────────────────

    def download_roboflow(
        self,
        project_version: str,    # e.g. "workspace/project/version_number"
        api_key: str,
        fmt: str = "yolov8",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download a dataset from Roboflow in YOLO format.

        Args:
            project_version: Roboflow project path: "workspace/project/version"
            api_key:         Your Roboflow API key.
            fmt:             Export format (yolov8, yolov5, coco, etc.)
            output_dir:      Where to save the dataset.

        Returns:
            Path to the downloaded and extracted dataset directory.
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")

        parts = project_version.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(
                "project_version must be 'workspace/project/version' "
                f"(got: {project_version})"
            )
        workspace, project, version = parts
        output_dir = Path(output_dir or self.dataset_dir / project)

        url = (f"https://api.roboflow.com/{workspace}/{project}/{version}/"
               f"{fmt}?api_key={api_key}")

        log.info("Fetching Roboflow download URL …")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        dl_url = data.get("export", {}).get("link") or data.get("link")
        if not dl_url:
            log.error("No download link in Roboflow response: %s", data)
            raise RuntimeError("Could not obtain Roboflow download link.")

        zip_path = output_dir.parent / f"{project}_v{version}.zip"
        log.info("Downloading %s …", dl_url)

        with requests.get(dl_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        log.info("Extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()

        log.info("Dataset saved to: %s", output_dir)
        return output_dir

    # ── SAR-Ship ───────────────────────────────────────────────────────────────

    def prepare_sar_ship(
        self,
        raw_dir: Path,
        output_dir: Optional[Path] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Path:
        """
        Convert SAR-Ship dataset from PASCAL VOC XML format to YOLO format.

        Expected input structure:
            raw_dir/
              Annotations/   *.xml
              JPEGImages/    *.jpg  or  *.png

        Output structure (YOLO):
            output_dir/
              images/train/
              images/val/
              labels/train/
              labels/val/
              data.yaml
        """
        import xml.etree.ElementTree as ET

        raw_dir    = Path(raw_dir)
        output_dir = Path(output_dir or self.dataset_dir / "sar_ship")

        ann_dir = raw_dir / "Annotations"
        img_dir = raw_dir / "JPEGImages"

        if not ann_dir.exists():
            # Try alternative layout (images + labels at root)
            ann_dir = raw_dir / "labels"
            img_dir = raw_dir / "images"

        xml_files = sorted(ann_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML annotation files found in {ann_dir}")

        log.info("Found %d annotation files in %s", len(xml_files), ann_dir)

        random.seed(seed)
        random.shuffle(xml_files)
        split_idx = int(len(xml_files) * train_ratio)
        splits = {
            "train": xml_files[:split_idx],
            "val":   xml_files[split_idx:],
        }

        converted = {"train": 0, "val": 0}

        for split_name, xml_list in splits.items():
            out_img = output_dir / "images" / split_name
            out_lbl = output_dir / "labels" / split_name
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for xml_path in xml_list:
                stem = xml_path.stem

                # Find the image
                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".tif"]:
                    candidate = img_dir / (stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break

                if img_path is None:
                    log.warning("Image not found for %s — skipping.", stem)
                    continue

                # Parse XML
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find("size")
                    W    = int(size.find("width").text)
                    H    = int(size.find("height").text)
                except Exception as e:
                    log.warning("Bad XML %s: %s", xml_path.name, e)
                    continue

                lines = []
                for obj in root.findall("object"):
                    bnd = obj.find("bndbox")
                    if bnd is None:
                        continue
                    x1 = float(bnd.find("xmin").text)
                    y1 = float(bnd.find("ymin").text)
                    x2 = float(bnd.find("xmax").text)
                    y2 = float(bnd.find("ymax").text)

                    cx = (x1 + x2) / (2 * W)
                    cy = (y1 + y2) / (2 * H)
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H

                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))

                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if not lines:
                    continue

                shutil.copy2(img_path, out_img / img_path.name)
                (out_lbl / (stem + ".txt")).write_text("\n".join(lines))
                converted[split_name] += 1

        # Write data.yaml
        yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  - ship
"""
        (output_dir / "data.yaml").write_text(yaml_content)

        log.info(
            "SAR-Ship conversion complete: %d train, %d val images.",
            converted["train"], converted["val"],
        )
        return output_dir

    # ── HRSC2016 ───────────────────────────────────────────────────────────────

    def prepare_hrsc2016(
        self,
        raw_dir: Path,
        output_dir: Optional[Path] = None,
        use_subtypes: bool = False,
    ) -> Path:
        """
        Convert HRSC2016 dataset to YOLO format.
        If use_subtypes=True, maps ship_type XML field to WARSHIP_CLASSES.
        Otherwise, all ships are class 0 (generic ship).

        Download from: https://www.kaggle.com/datasets/guofeng/hrsc2016
        """
        import xml.etree.ElementTree as ET
        from config import WARSHIP_CLASSES

        raw_dir    = Path(raw_dir)
        output_dir = Path(output_dir or self.dataset_dir / "hrsc2016")

        # HRSC has FullDataSet/AllImages and FullDataSet/Annotations
        img_dir = raw_dir / "FullDataSet" / "AllImages"
        ann_dir = raw_dir / "FullDataSet" / "Annotations"

        if not img_dir.exists():
            img_dir = raw_dir / "AllImages"
            ann_dir = raw_dir / "Annotations"

        if not ann_dir.exists():
            raise FileNotFoundError(f"HRSC2016 Annotations dir not found in {raw_dir}")

        xml_files = sorted(ann_dir.glob("*.xml"))
        log.info("HRSC2016: %d annotation files found.", len(xml_files))

        random.seed(42)
        random.shuffle(xml_files)
        split_idx = int(len(xml_files) * 0.8)
        splits = {"train": xml_files[:split_idx], "val": xml_files[split_idx:]}

        # HRSC ship type code → WARSHIP_CLASSES index (rough mapping)
        HRSC_TYPE_MAP = {
            "100000001": 0,  # Aircraft Carrier
            "100000002": 1,  # Destroyer
            "100000003": 2,  # Frigate
            "100000004": 3,  # Cruiser
            "100000009": 6,  # Patrol
        }

        nc = len(WARSHIP_CLASSES) if use_subtypes else 1

        for split_name, xml_list in splits.items():
            out_img = output_dir / "images" / split_name
            out_lbl = output_dir / "labels" / split_name
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for xml_path in xml_list:
                stem = xml_path.stem
                img_path = None
                for ext in [".bmp", ".jpg", ".png"]:
                    c = img_dir / (stem + ext)
                    if c.exists():
                        img_path = c
                        break
                if not img_path:
                    continue

                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    imgsz = root.find("Img_SizeWidth"), root.find("Img_SizeHeight")
                    W = int(imgsz[0].text)
                    H = int(imgsz[1].text)
                except Exception:
                    continue

                lines = []
                for obj in root.findall(".//HRSC_Object"):
                    try:
                        x1 = float(obj.find("box_xmin").text)
                        y1 = float(obj.find("box_ymin").text)
                        x2 = float(obj.find("box_xmax").text)
                        y2 = float(obj.find("box_ymax").text)
                        cls_code = (obj.find("Class_ID").text or "").strip()
                    except (AttributeError, ValueError):
                        continue

                    cls_id = 0
                    if use_subtypes:
                        cls_id = HRSC_TYPE_MAP.get(cls_code, 7)  # default Unknown

                    cx = max(0, min(1, (x1 + x2) / (2 * W)))
                    cy = max(0, min(1, (y1 + y2) / (2 * H)))
                    bw = max(0, min(1, (x2 - x1) / W))
                    bh = max(0, min(1, (y2 - y1) / H))
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if not lines:
                    continue

                shutil.copy2(img_path, out_img / img_path.name)
                (out_lbl / (stem + ".txt")).write_text("\n".join(lines))

        # data.yaml
        class_names = [f"  - {c}" for c in (WARSHIP_CLASSES if use_subtypes else ["ship"])]
        yaml_str = (f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\n\n"
                    f"nc: {nc}\nnames:\n" + "\n".join(class_names) + "\n")
        (output_dir / "data.yaml").write_text(yaml_str)
        log.info("HRSC2016 conversion complete → %s", output_dir)
        return output_dir

    # ── Dataset Statistics ─────────────────────────────────────────────────────

    def stats(self, dataset_dir: Path) -> Dict:
        """
        Compute class distribution and image count statistics
        for a YOLO-format dataset directory.

        Returns:
            {
                "train": {"images": N, "labels": N, "class_counts": {0: N}},
                "val":   { ... },
                "total_instances": N,
            }
        """
        dataset_dir = Path(dataset_dir)
        result: Dict = {}
        total_instances = 0

        for split in ["train", "val"]:
            img_dir = dataset_dir / "images" / split
            lbl_dir = dataset_dir / "labels" / split

            n_images = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
            class_counts: Dict[int, int] = {}
            n_labels = 0

            if lbl_dir.exists():
                for lbl in lbl_dir.glob("*.txt"):
                    lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
                    n_labels += 1
                    for line in lines:
                        parts = line.split()
                        if parts:
                            cls = int(parts[0])
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                            total_instances += 1

            result[split] = {
                "images":       n_images,
                "labels":       n_labels,
                "class_counts": class_counts,
            }

        result["total_instances"] = total_instances
        return result

    def print_stats(self, dataset_dir: Path):
        s = self.stats(dataset_dir)
        print(f"\n📊 Dataset Statistics: {dataset_dir}")
        print(f"{'─' * 50}")
        for split in ["train", "val"]:
            d = s.get(split, {})
            print(f"  {split.upper():<8}  images={d.get('images', 0):>5}  "
                  f"labels={d.get('labels', 0):>5}  "
                  f"instances={sum(d.get('class_counts', {}).values()):>6}")
        print(f"  {'TOTAL':<8}  instances={s['total_instances']:>6}")
        print()

    # ── Augmentation helpers ───────────────────────────────────────────────────

    @staticmethod
    def verify_labels(dataset_dir: Path) -> List[str]:
        """
        Verify all label files have valid YOLO format.
        Returns list of problematic file paths.
        """
        issues = []
        for lbl in Path(dataset_dir).rglob("labels/**/*.txt"):
            for i, line in enumerate(lbl.read_text().splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"{lbl}:{i+1} — expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    if not all(0.0 <= v <= 1.0 for v in [cx, cy, bw, bh]):
                        issues.append(f"{lbl}:{i+1} — values out of [0,1] range")
                except ValueError as e:
                    issues.append(f"{lbl}:{i+1} — {e}")
        return issues


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset management CLI")
    sub = parser.add_subparsers(dest="cmd")

    # stats
    p_stats = sub.add_parser("stats", help="Print dataset statistics")
    p_stats.add_argument("dataset_dir", type=Path)

    # verify
    p_verify = sub.add_parser("verify", help="Verify YOLO label files")
    p_verify.add_argument("dataset_dir", type=Path)

    # prepare-sar
    p_sar = sub.add_parser("prepare-sar", help="Prepare SAR-Ship dataset")
    p_sar.add_argument("--raw-dir",    type=Path, required=True)
    p_sar.add_argument("--output-dir", type=Path, default=None)

    # prepare-hrsc
    p_hrsc = sub.add_parser("prepare-hrsc", help="Prepare HRSC2016 dataset")
    p_hrsc.add_argument("--raw-dir",       type=Path, required=True)
    p_hrsc.add_argument("--output-dir",    type=Path, default=None)
    p_hrsc.add_argument("--use-subtypes",  action="store_true")

    # roboflow
    p_rf = sub.add_parser("roboflow", help="Download from Roboflow")
    p_rf.add_argument("--project", required=True, help="workspace/project/version")
    p_rf.add_argument("--api-key", required=True)
    p_rf.add_argument("--format",  default="yolov8")

    args = parser.parse_args()
    dm   = DatasetManager()

    if args.cmd == "stats":
        dm.print_stats(args.dataset_dir)
    elif args.cmd == "verify":
        issues = DatasetManager.verify_labels(args.dataset_dir)
        if issues:
            print(f"⚠️  {len(issues)} issue(s) found:")
            for iss in issues[:20]:
                print(" ", iss)
        else:
            print("✅ All labels valid.")
    elif args.cmd == "prepare-sar":
        dm.prepare_sar_ship(args.raw_dir, args.output_dir)
    elif args.cmd == "prepare-hrsc":
        dm.prepare_hrsc2016(args.raw_dir, args.output_dir,
                            use_subtypes=args.use_subtypes)
    elif args.cmd == "roboflow":
        dm.download_roboflow(args.project, args.api_key, fmt=args.format)
    else:
        parser.print_help()

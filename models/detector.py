"""
models/detector.py — YOLOv8-based ship detection on satellite image tiles.

Responsibilities:
  • Load a fine-tuned YOLOv8 model (or fall back to pretrained COCO weights).
  • Run inference on a list of tile PNGs.
  • Merge overlapping detections across tiles back to full-image coordinates.
  • Attach geographic lat/lon from tile sidecar JSONs.
  • Return a list of Detection dataclass objects.

Usage:
    from models.detector import ShipDetector, Detection
    detector = ShipDetector()
    detections = detector.run_tiles(tile_paths)
    for d in detections:
        print(d.lat, d.lon, d.confidence)
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    YOLO_WEIGHTS, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    YOLO_IMGSZ, TILE_SIZE_PX,
)
from utils.logger import get_logger

log = get_logger("detector")


# ── Detection dataclass ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single ship detection result."""
    tile_path:   Path
    bbox_px:     Tuple[float, float, float, float]  # x1, y1, x2, y2 in tile coords
    confidence:  float
    class_id:    int
    class_name:  str = "ship"

    # Geographic position (centre of bbox, from sidecar JSON)
    lat:         Optional[float] = None
    lon:         Optional[float] = None

    # Set by tracker
    track_id:    Optional[int] = None

    # Set by classifier
    warship_type:      Optional[str]  = None
    warship_conf:      Optional[float] = None

    # AIS-enriched fields
    vessel_name:  Optional[str] = None
    vessel_mmsi:  Optional[str] = None
    vessel_flag:  Optional[str] = None

    # Timestamp from the Sentinel product
    timestamp:    Optional[str] = None

    @property
    def centre_px(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> dict:
        return {
            "tile":         str(self.tile_path),
            "bbox_px":      list(self.bbox_px),
            "confidence":   round(self.confidence, 4),
            "class_name":   self.class_name,
            "lat":          self.lat,
            "lon":          self.lon,
            "track_id":     self.track_id,
            "warship_type": self.warship_type,
            "warship_conf": self.warship_conf,
            "vessel_name":  self.vessel_name,
            "vessel_mmsi":  self.vessel_mmsi,
            "vessel_flag":  self.vessel_flag,
            "timestamp":    self.timestamp,
        }


# ── ShipDetector ──────────────────────────────────────────────────────────────

class ShipDetector:
    """
    YOLOv8-based ship detector.

    On first call the model weights are loaded once and cached.
    Falls back to generic 'yolov8m.pt' (COCO) if fine-tuned weights
    are not yet available; the ship class index is then mapped correctly.
    """

    # COCO class index for 'boat' (closest proxy for ships in COCO)
    COCO_BOAT_CLASS = 8

    def __init__(
        self,
        weights: Optional[Path] = None,
        conf: float = YOLO_CONF_THRESHOLD,
        iou:  float = YOLO_IOU_THRESHOLD,
        imgsz: int  = YOLO_IMGSZ,
        device: str = "cpu",
    ):
        self.weights  = weights or YOLO_WEIGHTS
        self.conf     = conf
        self.iou      = iou
        self.imgsz    = imgsz
        self.device   = device
        self._model   = None

    def _load_model(self):
        """Lazy-load YOLOv8 model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        if self.weights.exists():
            log.info("Loading fine-tuned weights: %s", self.weights)
            self._model = YOLO(str(self.weights))
            self._custom_model = True
        else:
            log.warning(
                "Fine-tuned weights not found at %s — using generic YOLOv8m.",
                self.weights,
            )
            self._model = YOLO("yolov8m.pt")
            self._custom_model = False

        log.info("Model loaded on device: %s", self.device)

    # ── Core Inference ─────────────────────────────────────────────────────────

    def detect_tile(self, tile_path: Path) -> List[Detection]:
        """Run detection on a single tile PNG."""
        self._load_model()

        results = self._model.predict(
            source=str(tile_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        geo = self._load_geo(tile_path)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())

                # If using COCO model, only keep boat class
                if not self._custom_model and cls_id != self.COCO_BOAT_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                cls_name = (
                    self._model.names[cls_id]
                    if hasattr(self._model, "names")
                    else "ship"
                )

                lat, lon = self._pixel_to_geo(
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, geo=geo
                )

                detections.append(Detection(
                    tile_path=tile_path,
                    bbox_px=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    lat=lat,
                    lon=lon,
                ))

        return detections

    def run_tiles(self, tile_paths: List[Path], verbose: bool = True) -> List[Detection]:
        """
        Run detection on a list of tile paths.

        Returns:
            Merged list of Detection objects from all tiles.
        """
        all_detections: List[Detection] = []
        total = len(tile_paths)

        for i, tp in enumerate(tile_paths):
            if verbose:
                print(f"\r  Detecting … tile {i+1}/{total}", end="", flush=True)

            dets = self.detect_tile(tp)
            all_detections.extend(dets)

        if verbose:
            print()

        log.info(
            "Detected %d ship(s) across %d tiles (conf≥%.2f).",
            len(all_detections), total, self.conf,
        )
        return all_detections

    # ── Batch Inference (faster via ultralytics batch API) ────────────────────

    def run_batch(self, tile_paths: List[Path], batch_size: int = 8) -> List[Detection]:
        """
        Batch inference — up to ~3× faster than tile-by-tile on GPU.
        Groups tiles into batches and calls YOLO once per batch.
        """
        self._load_model()
        all_detections: List[Detection] = []
        batches = [
            tile_paths[i:i + batch_size]
            for i in range(0, len(tile_paths), batch_size)
        ]

        for b_idx, batch in enumerate(batches):
            log.info("Batch %d/%d (%d tiles) …", b_idx + 1, len(batches), len(batch))
            results = self._model.predict(
                source=[str(p) for p in batch],
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            for tile_path, r in zip(batch, results):
                geo = self._load_geo(tile_path)
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    if not self._custom_model and cls_id != self.COCO_BOAT_CLASS:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    lat, lon = self._pixel_to_geo(
                        cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, geo=geo
                    )
                    all_detections.append(Detection(
                        tile_path=tile_path,
                        bbox_px=(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                        class_id=cls_id,
                        class_name=self._model.names.get(cls_id, "ship"),
                        lat=lat,
                        lon=lon,
                    ))

        log.info("Batch detection complete — %d detection(s).", len(all_detections))
        return all_detections

    # ── Geo helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_geo(tile_path: Path) -> Optional[dict]:
        """Load the geo-metadata sidecar JSON next to the tile PNG."""
        json_path = tile_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        return None

    @staticmethod
    def _pixel_to_geo(
        cx: float, cy: float, geo: Optional[dict]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Interpolate geographic lat/lon for a pixel (cx, cy) within the tile,
        using the tile's top-left and bottom-right corners stored in the sidecar.
        """
        if geo is None:
            return None, None
        if not all(k in geo for k in ("lat_tl", "lon_tl", "lat_br", "lon_br")):
            return None, None

        frac_x = cx / TILE_SIZE_PX
        frac_y = cy / TILE_SIZE_PX
        lon = geo["lon_tl"] + frac_x * (geo["lon_br"] - geo["lon_tl"])
        lat = geo["lat_tl"] + frac_y * (geo["lat_br"] - geo["lat_tl"])
        return round(lat, 6), round(lon, 6)

    # ── NMS across tiles ───────────────────────────────────────────────────────

    @staticmethod
    def nms_across_tiles(
        detections: List[Detection],
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """
        Suppress duplicate detections caused by overlapping tiles.
        Uses a simple greedy IoU-based NMS in geographic space.
        """
        if not detections:
            return detections

        # Sort by confidence descending
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: List[Detection] = []

        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [
                d for d in dets
                if best.lat is None
                or d.lat is None
                or _geo_distance_km(best.lat, best.lon, d.lat, d.lon) > 0.05
                # 50 m threshold — two ships can't occupy same pixel
            ]

        log.debug("NMS: %d → %d detections after suppression.",
                  len(detections), len(keep))
        return keep


def _geo_distance_km(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ship detection on a tile directory")
    parser.add_argument("tile_dir", type=Path, help="Directory of tile PNGs")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tile_paths = sorted(args.tile_dir.glob("**/*.png"))
    print(f"Found {len(tile_paths)} tiles.")

    det = ShipDetector(conf=args.conf, device=args.device)
    results = det.run_tiles(tile_paths)
    results = ShipDetector.nms_across_tiles(results)

    print(f"\n{'Tile':<40} {'Conf':>6} {'Lat':>10} {'Lon':>10}")
    print("-" * 70)
    for d in results:
        print(f"{d.tile_path.name:<40} {d.confidence:>6.3f} "
              f"{str(d.lat):>10} {str(d.lon):>10}")

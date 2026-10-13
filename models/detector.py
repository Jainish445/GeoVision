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
    YOLO_WEIGHTS, YOLO_BASE_WEIGHTS, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    YOLO_IMGSZ, TILE_SIZE_PX, DETECT_MAX_SHIP_M,
)

# COCO has no "ship"; "boat" is the closest proxy when no fine-tuned weights exist
COCO_FALLBACK_CLASSES = {"boat"}
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

    # Hull outline as four (lat, lon) corners — set by oriented-box detectors
    footprint:   Optional[list] = None

    # Set by tracker
    track_id:    Optional[int] = None

    # Set by classifier
    warship_type:      Optional[str]  = None
    warship_conf:      Optional[float] = None
    length_m:          Optional[float] = None   # measured hull length
    beam_m:            Optional[float] = None   # measured hull width
    # Which warship type has a hull this size (a size class, not an identification)
    size_class:        Optional[str]  = None
    size_conf:         Optional[float] = None
    # What the warship_type rests on: "cnn", "ais", "ais+size", "context", "size"
    type_source:       Optional[str]  = None

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
            "length_m":     self.length_m,
            "beam_m":       self.beam_m,
            "size_class":   self.size_class,
            "type_source":  self.type_source,
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
    Falls back to the generic COCO base weights if fine-tuned weights
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
        keep_classes: Optional[set] = None,
    ):
        """
        Args:
            keep_classes: Class names to keep (e.g. {"ship"} for a general
                          aerial-imagery model). None keeps every class, which
                          is right for weights fine-tuned on ships only.
        """
        self.weights  = weights or YOLO_WEIGHTS
        self.conf     = conf
        self.iou      = iou
        self.imgsz    = imgsz
        self.device   = device
        self.keep_classes = keep_classes
        self._model   = None
        self._custom_model = False

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

        if self.weights.exists() or self.weights != YOLO_WEIGHTS:
            # Explicitly requested weights that are missing are official
            # ultralytics releases, which YOLO() downloads to this path.
            log.info("Loading weights: %s", self.weights)
            self._model = YOLO(str(self.weights))
            self._custom_model = True
        else:
            log.warning(
                "Fine-tuned weights not found at %s — using generic YOLOv8m "
                "(COCO 'boat' class only; expect poor recall on SAR imagery).",
                self.weights,
            )
            # Absolute path: ultralytics downloads here if missing, and a stale
            # file in the current working directory can't shadow it.
            try:
                self._model = YOLO(str(YOLO_BASE_WEIGHTS))
            except Exception as e:
                raise RuntimeError(
                    f"Could not load base weights {YOLO_BASE_WEIGHTS} ({e}). "
                    "If the file is corrupt (e.g. an interrupted download), "
                    "delete it and re-run so it is downloaded again."
                ) from e
            self._custom_model = False
            self.keep_classes = COCO_FALLBACK_CLASSES

        log.info("Model loaded on device: %s", self.device)

    # ── Core Inference ─────────────────────────────────────────────────────────

    def detect_tile(self, tile_path: Path) -> List[Detection]:
        """Run detection on a single tile PNG."""
        return self.run_batch([tile_path], batch_size=1, quiet=True)

    def run_tiles(self, tile_paths: List[Path], verbose: bool = True) -> List[Detection]:
        """
        Run detection on a list of tile paths, one at a time.

        Returns:
            Merged list of Detection objects from all tiles.
        """
        all_detections: List[Detection] = []
        total = len(tile_paths)

        for i, tp in enumerate(tile_paths):
            if verbose:
                print(f"\r  Detecting … tile {i+1}/{total}", end="", flush=True)
            all_detections.extend(self.detect_tile(tp))

        if verbose:
            print()

        log.info(
            "Detected %d ship(s) across %d tiles (conf≥%.2f).",
            len(all_detections), total, self.conf,
        )
        return all_detections

    # ── Batch Inference (faster via ultralytics batch API) ────────────────────

    def run_batch(self, tile_paths: List[Path], batch_size: int = 8,
                  quiet: bool = False) -> List[Detection]:
        """
        Batch inference — up to ~3× faster than tile-by-tile on GPU.
        Groups tiles into batches and calls YOLO once per batch.
        Handles both axis-aligned and oriented-box (OBB) models.
        """
        self._load_model()
        all_detections: List[Detection] = []
        batches = [
            tile_paths[i:i + batch_size]
            for i in range(0, len(tile_paths), batch_size)
        ]

        for b_idx, batch in enumerate(batches):
            if not quiet:
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
                all_detections.extend(self._parse_result(Path(tile_path), r))

        if not quiet:
            log.info("Batch detection complete — %d detection(s).", len(all_detections))
        return all_detections

    def _parse_result(self, tile_path: Path, r) -> List[Detection]:
        """Turn one ultralytics result into Detection objects."""
        geo = self._load_geo(tile_path)
        names = self._model.names
        out: List[Detection] = []

        def keep(cls_id: int) -> bool:
            return self.keep_classes is None or names.get(cls_id) in self.keep_classes

        if getattr(r, "obb", None) is not None:
            # Oriented boxes: the hull outline and its true length / beam
            gsd_m = (geo or {}).get("gsd_m")
            for corners, xyxy, xywhr, conf, cls in zip(
                r.obb.xyxyxyxy.tolist(), r.obb.xyxy.tolist(), r.obb.xywhr.tolist(),
                r.obb.conf.tolist(), r.obb.cls.tolist(),
            ):
                cls_id = int(cls)
                if not keep(cls_id):
                    continue
                lat, lon = self._pixel_to_geo(xywhr[0], xywhr[1], geo)
                footprint = [self._pixel_to_geo(x, y, geo) for x, y in corners]
                d = Detection(
                    tile_path=tile_path,
                    bbox_px=tuple(xyxy),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=names.get(cls_id, "ship"),
                    lat=lat,
                    lon=lon,
                    footprint=footprint if lat is not None else None,
                )
                if gsd_m:
                    d.length_m = round(max(xywhr[2], xywhr[3]) * gsd_m, 1)
                    d.beam_m   = round(min(xywhr[2], xywhr[3]) * gsd_m, 1)
                out.append(d)
            return out

        if r.boxes is None:
            return out
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if not keep(cls_id):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            lat, lon = self._pixel_to_geo(
                cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, geo=geo
            )
            out.append(Detection(
                tile_path=tile_path,
                bbox_px=(x1, y1, x2, y2),
                confidence=float(box.conf.item()),
                class_id=cls_id,
                class_name=names.get(cls_id, "ship"),
                lat=lat,
                lon=lon,
            ))
        return out

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
        Interpolate geographic lat/lon for a pixel (cx, cy) within the tile.

        Uses bilinear interpolation over all four tile corners when the
        sidecar has them (needed for rotated grids such as Sentinel-1 GRD),
        otherwise falls back to the top-left / bottom-right corners.
        """
        if geo is None:
            return None, None
        if not all(k in geo for k in ("lat_tl", "lon_tl", "lat_br", "lon_br")):
            return None, None

        size = geo.get("tile_size", TILE_SIZE_PX)
        fx = cx / size
        fy = cy / size

        if all(k in geo for k in ("lat_tr", "lon_tr", "lat_bl", "lon_bl")):
            def bilinear(key: str) -> float:
                top = geo[f"{key}_tl"] + fx * (geo[f"{key}_tr"] - geo[f"{key}_tl"])
                bot = geo[f"{key}_bl"] + fx * (geo[f"{key}_br"] - geo[f"{key}_bl"])
                return top + fy * (bot - top)
            return round(bilinear("lat"), 6), round(bilinear("lon"), 6)

        lon = geo["lon_tl"] + fx * (geo["lon_br"] - geo["lon_tl"])
        lat = geo["lat_tl"] + fy * (geo["lat_br"] - geo["lat_tl"])
        return round(lat, 6), round(lon, 6)

    # ── Size sanity filter ─────────────────────────────────────────────────────

    @staticmethod
    def filter_by_size(
        detections: List[Detection],
        max_length_m: float = DETECT_MAX_SHIP_M,
        min_length_m: float = 0.0,
    ) -> List[Detection]:
        """
        Drop detections whose box is longer than any ship (on real coastal
        scenes these are islets, breakwaters and piers) or shorter than
        min_length_m (boats, when only ships are of interest).
        """
        geo_cache: dict = {}
        keep = []
        for d in detections:
            key = str(d.tile_path)
            if key not in geo_cache:
                geo_cache[key] = ShipDetector._load_geo(Path(d.tile_path)) or {}
            gsd_m = geo_cache[key].get("gsd_m", 10.0)
            x1, y1, x2, y2 = d.bbox_px
            length = d.length_m or max(x2 - x1, y2 - y1) * gsd_m
            if min_length_m <= length <= max_length_m:
                keep.append(d)
        if len(keep) < len(detections):
            log.info("Dropped %d detection(s) outside %.0f–%.0f m (not ships).",
                     len(detections) - len(keep), min_length_m, max_length_m)
        return keep

    # ── NMS across tiles ───────────────────────────────────────────────────────

    @staticmethod
    def nms_across_tiles(
        detections: List[Detection],
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """
        Suppress duplicate detections caused by overlapping tiles.

        Greedy NMS in geographic space. A ship in the overlap strip is seen by
        two tiles, and one of them often sees only part of it, so the two box
        centres can be far more than a few pixels apart. Detections from
        different tiles are therefore merged when their centres are closer
        than half the larger box; boxes cut by a tile edge lose to complete ones.
        """
        if not detections:
            return detections

        geo_cache: dict = {}

        def info(d: Detection):
            """(half box length in km, True if the box touches its tile edge)."""
            key = str(d.tile_path)
            if key not in geo_cache:
                geo_cache[key] = ShipDetector._load_geo(Path(d.tile_path)) or {}
            geo = geo_cache[key]
            size = geo.get("tile_size", TILE_SIZE_PX)
            gsd_m = geo.get("gsd_m", 10.0)
            x1, y1, x2, y2 = d.bbox_px
            radius_km = 0.5 * max(x2 - x1, y2 - y1) * gsd_m / 1000.0
            truncated = min(x1, y1) <= 1 or max(x2, y2) >= size - 1
            return radius_km, truncated

        meta = {id(d): info(d) for d in detections}

        # Complete boxes first, then by confidence descending
        dets = sorted(detections,
                      key=lambda d: (meta[id(d)][1], -d.confidence))
        keep: List[Detection] = []

        def duplicate(best: Detection, d: Detection) -> bool:
            if best.lat is None or d.lat is None:
                return False
            if best.footprint and d.footprint:
                # Hull outlines known: duplicates overlap, neighbours moored
                # side by side do not
                return _footprint_overlap(best.footprint, d.footprint) > 0.5
            threshold = 0.05      # 50 m — two ships can't occupy the same spot
            if str(best.tile_path) != str(d.tile_path):
                threshold = max(threshold, meta[id(best)][0], meta[id(d)][0])
            return _geo_distance_km(best.lat, best.lon, d.lat, d.lon) <= threshold

        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if not duplicate(best, d)]

        log.debug("NMS: %d → %d detections after suppression.",
                  len(detections), len(keep))
        return keep


def _footprint_overlap(a: list, b: list) -> float:
    """Intersection area of two (lat, lon) outlines over the smaller one's area."""
    from shapely.geometry import Polygon

    lat0 = a[0][0]
    kx = 111_320.0 * float(np.cos(np.radians(lat0)))     # metres per degree lon

    def poly(corners):
        return Polygon([((lon - a[0][1]) * kx, (lat - lat0) * 111_320.0)
                        for lat, lon in corners]).buffer(0)

    pa, pb = poly(a), poly(b)
    smaller = min(pa.area, pb.area)
    return pa.intersection(pb).area / smaller if smaller > 0 else 0.0


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

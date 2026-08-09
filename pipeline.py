"""
pipeline.py — End-to-end orchestration of the Warship Detection & Tracking System.

Executes the full pipeline for one or more regions:
  1. Query Copernicus for the latest satellite products.
  2. Download the product(s).
  3. Preprocess into 640×640 tiles.
  4. Run YOLOv8 ship detection on every tile.
  5. Apply NMS across overlapping tiles.
  6. Run DeepSORT tracking across acquisitions.
  7. Classify warship types (ResNet-50 classifier).
  8. Enrich with AIS vessel data.
  9. Store results in SQLite.
 10. Optionally export to GeoJSON.

Usage:
    # Process one region with Sentinel-1 SAR:
    python pipeline.py --region South_China_Sea --sensor S1

    # Process all regions (full world scan):
    python pipeline.py --all --sensor S1 --days 7

    # Dry-run on a local product directory (skip download):
    python pipeline.py --local-dir data/raw/S1_product.SAFE --sensor S1

    # Use demo mode (synthetic detections, no download required):
    python pipeline.py --demo
"""

import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REGIONS_OF_INTEREST, RESULTS_DIR,
    YOLO_CONF_THRESHOLD, YOLO_IMGSZ,
)
from data.ingestion     import SentinelIngester
from data.preprocessing import Preprocessor
from models.detector    import ShipDetector, Detection
from models.tracker     import ShipTracker
from models.classifier  import WarshipClassifier
from geo.ais_fusion     import AISFusion
from geo.geodb          import GeoDB
from utils.logger       import get_logger

log = get_logger("pipeline")


class WarshipPipeline:
    """
    Top-level pipeline class.  Owns one instance of each component
    so models are loaded once and reused across multiple regions / acquisitions.
    """

    def __init__(
        self,
        enable_ais:        bool = True,
        enable_classifier: bool = True,
        ais_backend:       str  = "marinetraffic",
        ais_file:          Optional[Path] = None,
        device:            str  = "cpu",
        conf_threshold:    float = YOLO_CONF_THRESHOLD,
        export_geojson:    bool = True,
    ):
        log.info("Initialising Warship Detection Pipeline …")

        self.enable_ais        = enable_ais
        self.enable_classifier = enable_classifier
        self.export_geojson    = export_geojson

        # Component instances (lazy model loading inside each)
        self.ingester   = SentinelIngester()
        self.preprocessor = Preprocessor()
        self.detector   = ShipDetector(conf=conf_threshold, device=device)
        self.tracker    = ShipTracker()
        self.db         = GeoDB()

        if enable_classifier:
            self.classifier = WarshipClassifier(device=device)
        else:
            self.classifier = None

        if enable_ais:
            self.ais = AISFusion(
                backend=ais_backend,
                ais_file=ais_file,
            )
        else:
            self.ais = None

        log.info("Pipeline ready.")

    # ── Full pipeline for one region ───────────────────────────────────────────

    def run_region(
        self,
        region_name: str,
        sensor:      str = "S1",
        days_back:   int = 7,
        max_products: int = 1,
    ) -> List[Detection]:
        """
        Download the latest satellite data for a region and run the full pipeline.

        Returns:
            List of Detection objects with all fields populated.
        """
        log.info("=" * 60)
        log.info("REGION: %s  |  SENSOR: %s  |  DAYS_BACK: %d",
                 region_name, sensor, days_back)

        # 1. Download
        log.info("[1/7] Querying and downloading satellite imagery …")
        products = self.ingester.query_region(region_name, days_back=days_back,
                                              sensor=sensor)
        if not products:
            log.warning("No products found for region '%s'. Skipping.", region_name)
            return []

        product_paths = self.ingester.download(products, max_items=max_products)
        if not product_paths:
            return []

        all_detections: List[Detection] = []

        for product_path in product_paths:
            acq_ts = _extract_timestamp(products[0])
            dets = self._process_product(
                product_path=product_path,
                region=region_name,
                sensor=sensor,
                acq_ts=acq_ts,
            )
            all_detections.extend(dets)

        return all_detections

    def run_local(
        self,
        product_dir: Path,
        region:      str,
        sensor:      str = "S1",
        acq_ts:      str = "",
    ) -> List[Detection]:
        """
        Run the pipeline on a locally stored Sentinel product (no download).
        Useful for testing or reprocessing.
        """
        log.info("Running on local product: %s", product_dir)
        return self._process_product(
            product_path=product_dir,
            region=region,
            sensor=sensor,
            acq_ts=acq_ts or datetime.now(timezone.utc).isoformat(),
        )

    # ── Internal pipeline stages ───────────────────────────────────────────────

    def _process_product(
        self,
        product_path: Path,
        region:       str,
        sensor:       str,
        acq_ts:       str,
    ) -> List[Detection]:
        """Stages 2–9 for a single satellite product."""
        product_name = product_path.name
        acq_id = f"{region}_{product_name[:20]}_{uuid.uuid4().hex[:6]}"

        # 2. Preprocess → tiles
        log.info("[2/7] Preprocessing → tiles …")
        tile_paths = self.preprocessor.process_product(product_path, sensor=sensor)
        log.info("  %d tile(s) generated.", len(tile_paths))

        if not tile_paths:
            return []

        self.db.insert_acquisition(
            acq_id=acq_id,
            region=region,
            sensor=sensor,
            product_name=product_name,
            acq_ts=acq_ts,
            n_tiles=len(tile_paths),
        )

        # 3. Detection
        log.info("[3/7] Running YOLOv8 ship detection …")
        detections = self.detector.run_batch(tile_paths)

        # Set timestamp on every detection
        for d in detections:
            d.timestamp = acq_ts

        # 4. NMS across tile overlaps
        log.info("[4/7] Applying cross-tile NMS …")
        detections = ShipDetector.nms_across_tiles(detections)
        log.info("  %d detection(s) after NMS.", len(detections))

        if not detections:
            self.db.insert_detections(detections, region, acq_id)
            return detections

        # 5. Tracking
        log.info("[5/7] Updating DeepSORT tracker …")
        detections = self.tracker.update(detections, timestamp=acq_ts)

        # 6. Warship classification
        if self.classifier is not None:
            log.info("[6/7] Classifying warship types …")
            detections = self.classifier.classify_detections(
                detections, scene_image=None
            )
        else:
            log.info("[6/7] Classifier disabled — skipping.")

        # 7. AIS enrichment
        if self.ais is not None:
            log.info("[7/7] Enriching with AIS data …")
            detections = self.ais.enrich(detections)
        else:
            log.info("[7/7] AIS disabled — skipping.")

        # 8. Persist to DB
        self.db.insert_detections(detections, region, acq_id)

        # Update track records
        for det in detections:
            if det.track_id is not None:
                trail = self.tracker.get_track_trail(det.track_id)
                self.db.upsert_track(
                    track_id=det.track_id,
                    region=region,
                    detections=[d for d in detections if d.track_id == det.track_id],
                    trail=trail,
                )

        # 9. GeoJSON export
        if self.export_geojson:
            out = RESULTS_DIR / f"{region}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.geojson"
            self.db.export_geojson(out, region=region)

        log.info(
            "Region '%s' complete: %d ship(s) detected, %d track(s) active.",
            region, len(detections), len(self.tracker.active_track_ids()),
        )
        return detections

    # ── Demo mode ─────────────────────────────────────────────────────────────

    def run_demo(self) -> List[Detection]:
        """
        Generate synthetic detections for demonstration (no satellite download).
        Useful for testing the tracker, DB, and dashboard without real data.
        """
        import random, math

        log.info("Running in DEMO mode with synthetic detections …")
        random.seed(42)

        demo_region = "South_China_Sea"
        acq_id = f"DEMO_{uuid.uuid4().hex[:8]}"
        self.db.insert_acquisition(
            acq_id=acq_id, region=demo_region, sensor="DEMO",
            acq_ts=datetime.utcnow().isoformat(), n_tiles=0,
        )

        # Simulate 3 acquisitions 6 hours apart
        base_ships = [
            {"lat": 15.2, "lon": 114.5, "wtype": "Aircraft Carrier"},
            {"lat": 12.8, "lon": 110.3, "wtype": "Destroyer"},
            {"lat": 18.5, "lon": 118.2, "wtype": "Frigate"},
            {"lat":  9.1, "lon": 107.6, "wtype": "Patrol / Corvette"},
        ]

        all_dets = []
        for frame in range(3):
            ts = f"2024-03-{15 + frame:02d}T06:00:00Z"
            acq_id_f = f"DEMO_{frame}"
            self.db.insert_acquisition(
                acq_id=acq_id_f, region=demo_region, sensor="DEMO",
                acq_ts=ts, n_tiles=0,
            )
            dets = []
            for i, ship in enumerate(base_ships):
                # Simulate slight movement
                lat = ship["lat"] + random.uniform(-0.05, 0.05)
                lon = ship["lon"] + random.uniform(-0.05, 0.05)
                d = Detection(
                    tile_path   = Path(f"demo_tile_{i}.png"),
                    bbox_px     = (100.0, 100.0, 200.0, 200.0),
                    confidence  = round(random.uniform(0.72, 0.97), 3),
                    class_id    = 0,
                    class_name  = "ship",
                    lat         = round(lat, 5),
                    lon         = round(lon, 5),
                    track_id    = i + 1,
                    warship_type = ship["wtype"],
                    warship_conf = round(random.uniform(0.60, 0.92), 3),
                    timestamp   = ts,
                )
                dets.append(d)

            self.db.insert_detections(dets, demo_region, acq_id_f)
            for d in dets:
                self.db.upsert_track(
                    track_id=d.track_id,
                    region=demo_region,
                    detections=[d],
                    trail=[{"lat": d.lat, "lon": d.lon, "timestamp": ts}],
                )
            all_dets.extend(dets)

        log.info("Demo complete. %d synthetic detections inserted.", len(all_dets))
        return all_dets


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_timestamp(product: dict) -> str:
    """Try to extract acquisition timestamp from CDSE product metadata."""
    try:
        raw = product.get("ContentDate", {}).get("Start", "")
        return raw if raw else datetime.utcnow().isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Warship Detection & Tracking Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--region", choices=list(REGIONS_OF_INTEREST.keys()),
                       help="Single region to process")
    group.add_argument("--all", action="store_true",
                       help="Process all configured regions")
    group.add_argument("--local-dir", type=Path,
                       help="Path to local .SAFE product directory (skip download)")
    group.add_argument("--demo", action="store_true",
                       help="Run with synthetic demo data (no download)")

    parser.add_argument("--sensor",  default="S1", choices=["S1", "S2"],
                        help="Satellite sensor (default: S1 SAR)")
    parser.add_argument("--days",    type=int, default=7,
                        help="Days of history to search (default: 7)")
    parser.add_argument("--device",  default="cpu",
                        help="Inference device: cpu / 0 / 0,1 (default: cpu)")
    parser.add_argument("--no-ais",  action="store_true",
                        help="Disable AIS enrichment")
    parser.add_argument("--no-clf",  action="store_true",
                        help="Disable warship classifier")
    parser.add_argument("--conf",    type=float, default=YOLO_CONF_THRESHOLD,
                        help="Detection confidence threshold")
    parser.add_argument("--export-geojson", action="store_true", default=True,
                        help="Export detections to GeoJSON after each run")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = WarshipPipeline(
        enable_ais        = not args.no_ais,
        enable_classifier = not args.no_clf,
        device            = args.device,
        conf_threshold    = args.conf,
        export_geojson    = args.export_geojson,
    )

    if args.demo:
        dets = pipeline.run_demo()

    elif args.local_dir:
        dets = pipeline.run_local(
            product_dir = args.local_dir,
            region      = "custom",
            sensor      = args.sensor,
        )

    elif args.all:
        dets = []
        for region in REGIONS_OF_INTEREST:
            try:
                d = pipeline.run_region(
                    region_name  = region,
                    sensor       = args.sensor,
                    days_back    = args.days,
                )
                dets.extend(d)
            except Exception as e:
                log.error("Region '%s' failed: %s", region, e)

    else:
        dets = pipeline.run_region(
            region_name = args.region,
            sensor      = args.sensor,
            days_back   = args.days,
        )

    print(f"\n{'═' * 50}")
    print(f"  Pipeline complete — {len(dets)} detection(s)")
    print(f"  DB: {pipeline.db.db_path}")
    print(f"  Summary: {json.dumps(pipeline.db.summary(), indent=2)}")


if __name__ == "__main__":
    main()

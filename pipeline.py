"""
pipeline.py — End-to-end orchestration of the Warship Detection & Tracking System.

Executes the full pipeline for one or more regions:
  1. Query Copernicus for the latest satellite products.
  2. Download the product(s).
  3. Preprocess into 640×640 tiles.
  4. Run YOLOv8 ship detection on every tile.
  5. Apply NMS across overlapping tiles.
  6. Classify warship types (CNN, or hull-size fallback).
  7. Associate detections with tracks across acquisitions.
  8. Enrich with AIS vessel data.
  9. Store results in SQLite.
 10. Optionally export to GeoJSON.

Usage:
    # Process one region with Sentinel-1 SAR:
    python pipeline.py --region South_China_Sea --sensor S1

    # Process all regions (full world scan):
    python pipeline.py --all --sensor S1 --days 7

    # Dry-run on a local product directory or GeoTIFF (skip download):
    python pipeline.py --local-dir data/raw/S1_product.SAFE --sensor S1

    # Use demo mode (synthetic detections, no download required):
    python pipeline.py --demo
"""

import re
import sys
import json
import math
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REGIONS_OF_INTEREST, RESULTS_DIR, DATA_SOURCE,
    CIVILIAN_LABEL, UNIDENTIFIED_LABEL,
    HR_TILE_SIZE_PX, HR_TILE_STRIDE, HR_CONF_THRESHOLD, HR_MIN_SHIP_M, OBB_WEIGHTS,
    YOLO_CONF_THRESHOLD, YOLO_IMGSZ,
)
from data.ingestion     import make_ingester, PlanetaryComputerIngester
from data.preprocessing import Preprocessor
from models.detector    import ShipDetector, Detection
from models.tracker     import ShipTracker
from models.classifier  import WarshipClassifier
from geo.ais_fusion     import AISFusion, label_crowded_waters
from geo.geodb          import GeoDB, utc_now_iso
from utils.geo_utils    import destination_point
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
        source:            str  = DATA_SOURCE,
    ):
        log.info("Initialising Warship Detection Pipeline …")

        self.enable_ais        = enable_ais
        self.enable_classifier = enable_classifier
        self.export_geojson    = export_geojson

        # Component instances (lazy model loading inside each)
        self.source     = source
        self._ingester  = None          # created on first download
        self.preprocessor = Preprocessor()
        self.detector   = ShipDetector(conf=conf_threshold, device=device)
        self.tracker    = ShipTracker()
        self.db         = GeoDB()

        # High-resolution optical mode (sensor "NAIP"): bigger tiles and an
        # oriented-box detector trained on aerial photos
        self.preprocessor_hr = Preprocessor(tile_size=HR_TILE_SIZE_PX,
                                            stride=HR_TILE_STRIDE)
        self.detector_hr = ShipDetector(weights=OBB_WEIGHTS, conf=HR_CONF_THRESHOLD,
                                        imgsz=HR_TILE_SIZE_PX, device=device,
                                        keep_classes={"ship"})

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

    @property
    def ingester(self):
        if self._ingester is None:
            self._ingester = make_ingester(self.source)
        return self._ingester

    @ingester.setter
    def ingester(self, value):
        self._ingester = value

    @property
    def db(self) -> GeoDB:
        return self._db

    @db.setter
    def db(self, value: GeoDB):
        # The tracker seeds its state (and ID counter) from the database
        self._db = value
        self.tracker.db = value
        self.tracker.reset()

    # ── Full pipeline for one region ───────────────────────────────────────────

    def run_region(
        self,
        region_name: str,
        sensor:      str = "S1",
        days_back:   int = 7,
        max_products: int = 1,
        bbox:        Optional[tuple] = None,
        date:        Optional[str] = None,
    ) -> List[Detection]:
        """
        Download the latest satellite data for a region and run the full pipeline.

        Args:
            region_name:  Key of REGIONS_OF_INTEREST, or any label when `bbox`
                          is given.
            bbox:         Optional (lat_min, lon_min, lat_max, lon_max) to
                          search instead of a named region.
            date:         Optional YYYY-MM-DD: use the imagery acquired on that
                          UTC day instead of the newest (reproducible runs).

        Returns:
            List of Detection objects with all fields populated.
        """
        log.info("=" * 60)
        log.info("REGION: %s  |  SENSOR: %s  |  DAYS_BACK: %d",
                 region_name, sensor, days_back)

        # 1. Download
        log.info("[1/7] Querying and downloading satellite imagery …")
        # NAIP only exists on Planetary Computer
        ingester = PlanetaryComputerIngester() if sensor == "NAIP" else self.ingester
        window = {}
        if date:
            day = datetime.strptime(date, "%Y-%m-%d")
            window = {
                "date_start": day.strftime("%Y-%m-%dT00:00:00Z"),
                "date_end": (day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
            }
        if bbox is not None:
            products = ingester.query_bbox(tuple(bbox), days_back=days_back,
                                           sensor=sensor, **window)
        else:
            products = ingester.query_region(region_name, days_back=days_back,
                                             sensor=sensor, **window)
        if not products:
            log.warning("No products found for region '%s'. Skipping.", region_name)
            return []

        if sensor == "NAIP":
            max_products = len(products)     # one flight, several image quads
        products = products[:max_products]
        product_paths = ingester.download(products, max_items=max_products)
        products = products[:len(product_paths)] if sensor != "NAIP" else products
        if not product_paths:
            return []

        all_detections: List[Detection] = []

        if sensor == "NAIP":
            # One flight split over adjoining, slightly overlapping image
            # quads: treat them as a single acquisition so that ships on a
            # seam are not counted twice.
            return self._process_product(
                product_path=product_paths[0],
                region=region_name,
                sensor=sensor,
                acq_ts=_extract_timestamp(products[0]),
                extra_paths=product_paths[1:],
            )

        # Oldest first so tracks are extended in chronological order
        pairs = sorted(zip(products, product_paths),
                       key=lambda pp: _extract_timestamp(pp[0]))
        for product, product_path in pairs:
            dets = self._process_product(
                product_path=product_path,
                region=region_name,
                sensor=sensor,
                acq_ts=_extract_timestamp(product),
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
        product_dir = Path(product_dir)
        if not product_dir.exists():
            raise FileNotFoundError(f"Local product not found: {product_dir}")

        log.info("Running on local product: %s", product_dir)
        return self._process_product(
            product_path=product_dir,
            region=region,
            sensor=sensor,
            acq_ts=acq_ts or _timestamp_from_name(product_dir.name) or utc_now_iso(),
        )

    # ── Internal pipeline stages ───────────────────────────────────────────────

    def _process_product(
        self,
        product_path: Path,
        region:       str,
        sensor:       str,
        acq_ts:       str,
        extra_paths:  tuple = (),
    ) -> List[Detection]:
        """
        Stages 2–9 for a single satellite product. `extra_paths` are further
        pieces of the same acquisition, tiled and detected together with it.
        """
        product_name = product_path.name
        acq_id = f"{region}_{product_name[:20]}_{uuid.uuid4().hex[:6]}"

        # 2. Preprocess → tiles
        log.info("[2/7] Preprocessing → tiles …")
        hr = sensor == "NAIP"
        preprocessor = self.preprocessor_hr if hr else self.preprocessor
        detector     = self.detector_hr if hr else self.detector
        tile_paths = preprocessor.process_product(product_path, sensor=sensor)
        for extra in extra_paths:
            tile_paths += preprocessor.process_product(extra, sensor=sensor)
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
        detections = detector.run_batch(tile_paths, batch_size=2 if hr else 8)

        # Set timestamp on every detection
        for d in detections:
            d.timestamp = acq_ts

        # 4. NMS across tile overlaps
        log.info("[4/7] Applying cross-tile NMS …")
        detections = ShipDetector.nms_across_tiles(detections)
        detections = ShipDetector.filter_by_size(
            detections, min_length_m=HR_MIN_SHIP_M if hr else 0.0)
        log.info("  %d detection(s) after NMS.", len(detections))

        if not detections:
            return detections

        detections = self._track_classify_store(detections, region, acq_id, acq_ts)

        # 9. GeoJSON export
        if self.export_geojson:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
            self.db.export_geojson(RESULTS_DIR / f"{region}_{stamp}.geojson",
                                   region=region)

        log.info(
            "Region '%s' complete: %d ship(s) detected, %d track(s) known.",
            region, len(detections), len(self.tracker.active_track_ids()),
        )
        return detections

    def _track_classify_store(
        self,
        detections: List[Detection],
        region:     str,
        acq_id:     str,
        acq_ts:     str,
    ) -> List[Detection]:
        """Stages 5–8: classification, tracking, AIS fusion, persistence."""
        # Classification runs first so the tracker can use measured hull size
        # (each detection is cropped from its own tile)
        if self.classifier is not None:
            log.info("[5/7] Classifying warship types …")
            detections = self.classifier.classify_detections(detections)
        else:
            log.info("[5/7] Classifier disabled — skipping.")

        log.info("[6/7] Associating detections with tracks …")
        detections = self.tracker.update(detections, timestamp=acq_ts, region=region)

        # 7. AIS enrichment
        if self.ais is not None:
            log.info("[7/7] Enriching with AIS data …")
            detections = self.ais.enrich(detections)
        else:
            log.info("[7/7] AIS disabled — skipping.")

        # Ships no evidence has identified, sitting in a crowd of other ships,
        # are commercial traffic
        detections = label_crowded_waters(detections)

        # 8. Persist to DB and refresh the affected track records
        self.db.insert_detections(detections, region, acq_id)
        self.db.rebuild_tracks(d.track_id for d in detections)
        return detections

    # ── Demo mode ─────────────────────────────────────────────────────────────

    def run_demo(self) -> List[Detection]:
        """
        Generate synthetic detections for demonstration (no satellite download).

        Simulates three acquisitions six hours apart, ending now, of a naval
        group and two merchant ships. Everything downstream of the detector
        runs for real: a synthetic SAR tile is rendered for every ship and a
        synthetic AIS feed is written for every pass, so the classifier, AIS
        fusion, tracker, DB and dashboard are all exercised.

        The ships show each way a label can come about:
          • warship broadcasting military AIS  → warship type (AIS + hull size)
          • warship running dark (no AIS)      → "Unidentified Vessel" + size class
          • merchant ship broadcasting AIS     → "Civilian Vessel"

        Re-running replaces the previous demo data.
        """
        import random

        log.info("Running in DEMO mode with synthetic detections …")
        rng = random.Random(42)

        removed = self.db.clear_demo()
        if removed:
            log.info("Removed %d detection(s) from a previous demo run.", removed)
            self.tracker.reset()

        demo_region = "South_China_Sea"
        # start position, hull length × beam (m), heading (°), speed (kn),
        # AIS broadcast (ship-type code, name, flag) or None if running dark,
        # and the label the pipeline should arrive at
        base_ships = [
            {"lat": 15.2, "lon": 114.5, "length": 333, "beam": 76, "heading":  45,
             "speed": 18, "ais": (35, "DEMO CARRIER", "US"),
             "expect": "Aircraft Carrier"},
            {"lat": 12.8, "lon": 110.3, "length": 155, "beam": 20, "heading":  80,
             "speed": 22, "ais": (35, "DEMO DESTROYER", "US"),
             "expect": "Destroyer"},
            {"lat": 18.5, "lon": 118.2, "length": 125, "beam": 16, "heading": 200,
             "speed": 16, "ais": None,                       # frigate, dark
             "expect": UNIDENTIFIED_LABEL},
            {"lat":  9.1, "lon": 107.6, "length":  80, "beam": 12, "heading": 320,
             "speed": 12, "ais": None,                       # patrol vessel, dark
             "expect": UNIDENTIFIED_LABEL},
            {"lat": 16.4, "lon": 112.0, "length": 240, "beam": 34, "heading": 130,
             "speed": 14, "ais": (35, "DEMO AMPHIB", "US"),
             "expect": "Amphibious Assault"},
            {"lat": 11.0, "lon": 113.0, "length": 250, "beam": 44, "heading":  60,
             "speed": 13, "ais": (80, "DEMO TANKER", "PA"),
             "expect": CIVILIAN_LABEL},
            {"lat": 14.0, "lon": 117.0, "length": 190, "beam": 30, "heading": 250,
             "speed": 15, "ais": (70, "DEMO CARGO", "LR"),
             "expect": CIVILIAN_LABEL},
        ]

        n_frames, step_h = 3, 6
        now = datetime.now(timezone.utc).replace(microsecond=0)
        tile_dir = self.preprocessor.out_dir / "DEMO"

        all_dets = []
        for frame in range(n_frames):
            ts = (now - timedelta(hours=step_h * (n_frames - 1 - frame))
                  ).strftime("%Y-%m-%dT%H:%M:%SZ")
            acq_id = f"DEMO_{frame}"
            self.db.insert_acquisition(
                acq_id=acq_id, region=demo_region, sensor="DEMO",
                product_name="synthetic", acq_ts=ts, n_tiles=len(base_ships),
            )
            dets, ais_records = [], []
            for i, ship in enumerate(base_ships):
                # Dead-reckon along the ship's heading, with a little jitter
                dist_km = ship["speed"] * 1.852 * step_h * frame
                lat, lon = destination_point(
                    (ship["lat"], ship["lon"]),
                    ship["heading"] + rng.uniform(-4, 4), dist_km,
                )
                conf = round(rng.uniform(0.72, 0.97), 3)

                if self.classifier is not None:
                    d = _render_demo_tile(
                        tile_dir / f"frame{frame}_ship{i}.png", ship, lat, lon, rng,
                    )
                    d.confidence = conf
                    if ship["ais"]:
                        code, name, flag = ship["ais"]
                        ais_records.append({
                            "MMSI": str(366000000 + i), "SHIPNAME": name,
                            "FLAG": flag, "SHIPTYPE": code, "LAT": lat, "LON": lon,
                        })
                else:
                    # No classifier → use the expected label directly
                    d = Detection(
                        tile_path   = Path(f"demo_tile_{i}.png"),
                        bbox_px     = (100.0, 100.0, 200.0, 200.0),
                        confidence  = conf,
                        class_id    = 0,
                        lat         = round(lat, 5),
                        lon         = round(lon, 5),
                        warship_type = ship["expect"],
                    )
                d.timestamp = ts
                dets.append(d)

            # Same stages as a real run, with this pass's synthetic AIS feed
            real_ais, self.ais = self.ais, None
            if self.classifier is not None:
                tile_dir.mkdir(parents=True, exist_ok=True)
                ais_file = tile_dir / f"ais_frame{frame}.json"
                ais_file.write_text(json.dumps(ais_records))
                self.ais = AISFusion(backend="file", ais_file=ais_file)
            try:
                self._track_classify_store(dets, demo_region, acq_id, ts)
            finally:
                self.ais = real_ais

            if self.classifier is not None:
                for d, ship in zip(dets, base_ships):
                    log.info("  %-14s → %-19s size class %-18s evidence %-8s "
                             "hull %s × %s m",
                             (ship["ais"] or (0, "(no AIS)"))[1], d.warship_type,
                             d.size_class, d.type_source, d.length_m, d.beam_m)
            all_dets.extend(dets)

        log.info("Demo complete. %d synthetic detections inserted.", len(all_dets))
        return all_dets


# ── Helpers ────────────────────────────────────────────────────────────────────

def _render_demo_tile(png_path: Path, ship: dict, lat: float, lon: float,
                      rng, size: int = 256, gsd_m: float = 10.0) -> Detection:
    """
    Render a synthetic SAR tile (sea clutter + one bright hull at the centre)
    with a geo sidecar, and return the Detection a detector would produce.
    """
    import numpy as np
    from PIL import Image

    nprng = np.random.default_rng(rng.randrange(2 ** 32))
    sea = nprng.gamma(shape=4.0, scale=7.0, size=(size, size))    # speckle

    # Hull footprint: rectangle rotated to the ship's heading (north is up)
    theta = math.radians(ship["heading"])
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - size / 2, yy - size / 2
    along  = dx * math.sin(theta) - dy * math.cos(theta)
    across = dx * math.cos(theta) + dy * math.sin(theta)
    hull = ((np.abs(along) <= ship["length"] / gsd_m / 2)
            & (np.abs(across) <= max(ship["beam"] / gsd_m / 2, 0.5)))
    sea[hull] = nprng.normal(225, 12, size=int(hull.sum()))

    img = np.clip(sea, 0, 255).astype(np.uint8)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([img] * 3, axis=-1)).save(png_path)

    half_lat = size * gsd_m / 2 / 111_320.0
    half_lon = half_lat / math.cos(math.radians(lat))
    geo = {
        "row": 0, "col": 0, "product": "DEMO", "tile_size": size, "gsd_m": gsd_m,
        "lat_tl": lat + half_lat, "lon_tl": lon - half_lon,
        "lat_tr": lat + half_lat, "lon_tr": lon + half_lon,
        "lat_bl": lat - half_lat, "lon_bl": lon - half_lon,
        "lat_br": lat - half_lat, "lon_br": lon + half_lon,
    }
    with open(png_path.with_suffix(".json"), "w") as f:
        json.dump(geo, f)

    ys, xs = np.nonzero(hull)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    d_lat, d_lon = ShipDetector._pixel_to_geo(
        (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, geo)
    return Detection(tile_path=png_path, bbox_px=bbox, confidence=0.9,
                     class_id=0, lat=d_lat, lon=d_lon)


def _extract_timestamp(product: dict) -> str:
    """Try to extract acquisition timestamp from CDSE product metadata."""
    try:
        raw = product.get("ContentDate", {}).get("Start", "")
    except AttributeError:
        raw = ""
    return raw or _timestamp_from_name(product.get("Name", "")) or utc_now_iso()


def _timestamp_from_name(name: str) -> Optional[str]:
    """Sentinel product names embed the sensing start as YYYYMMDDTHHMMSS."""
    m = re.search(r"(\d{8})T(\d{6})", name or "")
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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
    group.add_argument("--bbox", type=float, nargs=4,
                       metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
                       help="Custom area to process (label it with --local-region)")
    group.add_argument("--local-dir", type=Path,
                       help="Path to a local .SAFE product directory or GeoTIFF "
                            "(skip download)")
    group.add_argument("--demo", action="store_true",
                       help="Run with synthetic demo data (no download)")

    parser.add_argument("--sensor",  default="S1", choices=["S1", "S2", "NAIP"],
                        help="S1 = Sentinel-1 radar (default), S2 = Sentinel-2 "
                             "optical,\nNAIP = 0.6 m aerial imagery (US only) — sharp "
                             "enough for the CNN to type warships")
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
    parser.add_argument("--local-region", default="custom",
                        help="Region label stored for --local-dir / --bbox runs")
    parser.add_argument("--source", default=DATA_SOURCE,
                        choices=["auto", "copernicus", "planetary"],
                        help="Imagery source (default: auto — Copernicus if "
                             "credentials are set, else Planetary Computer)")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="Use the imagery acquired on this UTC day instead of "
                             "the newest (makes a run reproducible)")
    parser.add_argument("--max-products", type=int, default=1,
                        help="Products to process per region, newest first")
    parser.add_argument("--no-geojson", dest="export_geojson", action="store_false",
                        help="Do not export detections to GeoJSON after each run")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = WarshipPipeline(
        enable_ais        = not args.no_ais,
        enable_classifier = not args.no_clf,
        device            = args.device,
        conf_threshold    = args.conf,
        export_geojson    = args.export_geojson,
        source            = args.source,
    )

    failed = []

    if args.demo:
        dets = pipeline.run_demo()

    elif args.local_dir:
        dets = pipeline.run_local(
            product_dir = args.local_dir,
            region      = args.local_region,
            sensor      = args.sensor,
        )

    else:
        dets = []
        if args.bbox:
            regions = [args.local_region]
        else:
            regions = list(REGIONS_OF_INTEREST) if args.all else [args.region]
        for region in regions:
            try:
                dets.extend(pipeline.run_region(
                    region_name  = region,
                    sensor       = args.sensor,
                    days_back    = args.days,
                    max_products = args.max_products,
                    bbox         = args.bbox,
                    date         = args.date,
                ))
            except Exception as e:
                log.error("Region '%s' failed: %s", region, e)
                failed.append(region)

    print(f"\n{'═' * 50}")
    print(f"  Pipeline complete — {len(dets)} detection(s)")
    if failed:
        print(f"  FAILED regions: {', '.join(failed)}")
    print(f"  DB: {pipeline.db.db_path}")
    print(f"  Summary: {json.dumps(pipeline.db.summary(), indent=2)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

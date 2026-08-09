"""
tests/test_pipeline.py — Unit & integration tests for the Warship Tracking System.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q        # quiet mode
    pytest tests/test_pipeline.py::TestGeoUtils -v   # single class
"""

import json
import math
import sqlite3
import tempfile
import sys
from pathlib import Path
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# Geo Utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestGeoUtils:
    """Tests for utils/geo_utils.py"""

    def setup_method(self):
        from utils.geo_utils import (
            haversine_km, haversine_nm, bearing_degrees,
            destination_point, speed_knots, estimate_next_position,
            bbox_expand, bbox_to_wkt, split_bbox_grid,
            point_in_bbox, bbox_area_km2,
        )
        self.haversine_km        = haversine_km
        self.haversine_nm        = haversine_nm
        self.bearing_degrees     = bearing_degrees
        self.destination_point   = destination_point
        self.speed_knots         = speed_knots
        self.estimate_next_pos   = estimate_next_position
        self.bbox_expand         = bbox_expand
        self.bbox_to_wkt         = bbox_to_wkt
        self.split_bbox_grid     = split_bbox_grid
        self.point_in_bbox       = point_in_bbox
        self.bbox_area_km2       = bbox_area_km2

    def test_haversine_known_distance(self):
        # London to Paris ≈ 340 km
        dist = self.haversine_km((51.5, -0.1), (48.9, 2.3))
        assert 335 < dist < 345, f"Expected ~340 km, got {dist:.1f}"

    def test_haversine_zero_distance(self):
        assert self.haversine_km((10.0, 20.0), (10.0, 20.0)) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_nm_conversion(self):
        km = self.haversine_km((0, 0), (0, 1))
        nm = self.haversine_nm((0, 0), (0, 1))
        assert nm == pytest.approx(km / 1.852, rel=1e-4)

    def test_bearing_east(self):
        # Moving east along equator → bearing 90°
        assert self.bearing_degrees((0, 0), (0, 1)) == pytest.approx(90.0, abs=0.1)

    def test_bearing_north(self):
        # Moving north → bearing 0°
        assert self.bearing_degrees((0, 0), (1, 0)) == pytest.approx(0.0, abs=0.1)

    def test_bearing_south(self):
        assert self.bearing_degrees((1, 0), (0, 0)) == pytest.approx(180.0, abs=0.1)

    def test_destination_point_roundtrip(self):
        origin = (15.0, 114.0)
        dist_km = 100.0
        bearing = 45.0
        dest = self.destination_point(origin, bearing, dist_km)
        back_dist = self.haversine_km(origin, dest)
        assert back_dist == pytest.approx(dist_km, rel=1e-3)

    def test_speed_knots(self):
        # 1 nautical mile in 1 hour = 1 knot
        p1 = (0.0, 0.0)
        p2 = (0.0, 1.0 / 60.0)   # ≈ 1 nm east
        spd = self.speed_knots(p1, p2, elapsed_hours=1.0)
        assert 0.9 < spd < 1.1

    def test_speed_knots_zero_elapsed(self):
        assert self.speed_knots((0, 0), (1, 1), elapsed_hours=0) == 0.0

    def test_estimate_next_position_basic(self):
        positions = [
            {"lat": 15.0, "lon": 114.0, "timestamp": "2024-01-15T00:00:00Z"},
            {"lat": 15.1, "lon": 114.1, "timestamp": "2024-01-15T06:00:00Z"},
        ]
        proj = self.estimate_next_pos(positions, hours_ahead=6.0)
        assert proj is not None
        lat, lon = proj
        assert 15.0 < lat < 16.0
        assert 114.0 < lon < 115.0

    def test_estimate_next_position_insufficient_data(self):
        assert self.estimate_next_pos([{"lat": 0, "lon": 0, "timestamp": ""}]) is None

    def test_bbox_expand(self):
        bbox = (10.0, 20.0, 30.0, 40.0)
        expanded = self.bbox_expand(bbox, margin_deg=1.0)
        assert expanded[0] == pytest.approx(9.0)
        assert expanded[1] == pytest.approx(19.0)
        assert expanded[2] == pytest.approx(31.0)
        assert expanded[3] == pytest.approx(41.0)

    def test_bbox_expand_clamps_poles(self):
        # Should not go past ±90 lat or ±180 lon
        expanded = self.bbox_expand((-89.0, -179.0, 89.0, 179.0), margin_deg=5.0)
        assert expanded[0] >= -90
        assert expanded[2] <= 90

    def test_bbox_to_wkt_format(self):
        wkt = self.bbox_to_wkt((10.0, 20.0, 30.0, 40.0))
        assert wkt.startswith("POLYGON((")
        assert "20.0 10.0" in wkt

    def test_split_bbox_grid_count(self):
        # 10°×10° box split into 5° cells → 4 cells
        cells = self.split_bbox_grid((0.0, 0.0, 10.0, 10.0), cell_deg=5.0)
        assert len(cells) == 4

    def test_split_bbox_grid_coverage(self):
        bbox = (0.0, 0.0, 10.0, 10.0)
        cells = self.split_bbox_grid(bbox, cell_deg=5.0)
        all_lats = [c[0] for c in cells] + [c[2] for c in cells]
        assert min(all_lats) == pytest.approx(0.0)
        assert max(all_lats) == pytest.approx(10.0)

    def test_point_in_bbox(self):
        bbox = (10.0, 20.0, 30.0, 40.0)
        assert self.point_in_bbox(20.0, 30.0, bbox) is True
        assert self.point_in_bbox(5.0, 30.0, bbox) is False
        assert self.point_in_bbox(20.0, 50.0, bbox) is False

    def test_bbox_area_positive(self):
        area = self.bbox_area_km2((0.0, 99.0, 25.0, 125.0))
        assert area > 0


# ══════════════════════════════════════════════════════════════════════════════
# Detection Dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestDetection:
    """Tests for models/detector.py Detection dataclass"""

    def _make_det(self, **kwargs):
        from models.detector import Detection
        defaults = {
            "tile_path": Path("tile.png"),
            "bbox_px": (100.0, 100.0, 200.0, 200.0),
            "confidence": 0.85,
            "class_id": 0,
        }
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_centre_px(self):
        d = self._make_det(bbox_px=(100.0, 100.0, 200.0, 200.0))
        assert d.centre_px == (150.0, 150.0)

    def test_to_dict_keys(self):
        d = self._make_det(lat=15.0, lon=114.0, track_id=1)
        data = d.to_dict()
        for key in ("lat", "lon", "confidence", "track_id", "class_name"):
            assert key in data

    def test_to_dict_values(self):
        d = self._make_det(lat=15.123, lon=114.456, confidence=0.92)
        data = d.to_dict()
        assert data["lat"]        == pytest.approx(15.123)
        assert data["lon"]        == pytest.approx(114.456)
        assert data["confidence"] == pytest.approx(0.92, abs=0.001)

    def test_defaults_none(self):
        d = self._make_det()
        assert d.track_id    is None
        assert d.warship_type is None
        assert d.lat         is None


# ══════════════════════════════════════════════════════════════════════════════
# Detector — pixel to geo interpolation
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectorGeo:
    """Tests for ShipDetector geo helpers (no model needed)."""

    def setup_method(self):
        from models.detector import ShipDetector
        self.ShipDetector = ShipDetector

    def test_pixel_to_geo_centre(self):
        geo = {"lat_tl": 20.0, "lon_tl": 114.0,
               "lat_br": 19.9, "lon_br": 114.1}
        lat, lon = self.ShipDetector._pixel_to_geo(320, 320, geo)
        # Centre of 640px tile → halfway between TL and BR
        assert lat == pytest.approx(19.95, abs=0.01)
        assert lon == pytest.approx(114.05, abs=0.01)

    def test_pixel_to_geo_tl_corner(self):
        geo = {"lat_tl": 20.0, "lon_tl": 114.0,
               "lat_br": 19.9, "lon_br": 114.1}
        lat, lon = self.ShipDetector._pixel_to_geo(0, 0, geo)
        assert lat == pytest.approx(20.0, abs=1e-4)
        assert lon == pytest.approx(114.0, abs=1e-4)

    def test_pixel_to_geo_no_sidecar(self):
        lat, lon = self.ShipDetector._pixel_to_geo(100, 100, None)
        assert lat is None
        assert lon is None

    def test_nms_across_tiles_empty(self):
        from models.detector import ShipDetector
        result = ShipDetector.nms_across_tiles([])
        assert result == []

    def test_nms_removes_duplicates(self):
        from models.detector import Detection, ShipDetector
        # Two detections very close together → only one survives NMS
        d1 = Detection(Path("t.png"), (0, 0, 100, 100), 0.9, 0, lat=15.0, lon=114.0)
        d2 = Detection(Path("t.png"), (0, 0, 100, 100), 0.7, 0, lat=15.00001, lon=114.00001)
        result = ShipDetector.nms_across_tiles([d1, d2])
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_nms_keeps_distant(self):
        from models.detector import Detection, ShipDetector
        # Two detections far apart → both survive
        d1 = Detection(Path("t.png"), (0, 0, 100, 100), 0.9, 0, lat=15.0, lon=114.0)
        d2 = Detection(Path("t.png"), (0, 0, 100, 100), 0.8, 0, lat=16.0, lon=115.0)
        result = ShipDetector.nms_across_tiles([d1, d2])
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Tests for data/preprocessing.py"""

    def setup_method(self):
        from data.preprocessing import Preprocessor
        self.Preprocessor = Preprocessor

    def test_normalize_band_range(self):
        from data.preprocessing import Preprocessor
        arr = np.random.rand(100, 100).astype(np.float32)
        result = Preprocessor._normalize_band(arr)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_sar_to_uint8_range(self):
        from data.preprocessing import Preprocessor
        arr = np.random.exponential(scale=0.1, size=(100, 100)).astype(np.float32)
        result = Preprocessor._sar_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_tile_image_count(self):
        """Test that tiling a 1280×1280 image with stride 576 produces the right tile count."""
        from data.preprocessing import Preprocessor
        with tempfile.TemporaryDirectory() as tmpdir:
            pre = Preprocessor(tile_size=640, stride=576, out_dir=Path(tmpdir))
            image = np.zeros((1280, 1280, 3), dtype=np.uint8)
            tiles = pre._tile_image(image, transform=None, product_name="test_prod")
            assert len(tiles) > 0
            for t in tiles:
                assert t.exists()
                assert t.suffix == ".png"

    def test_tile_image_sidecar_json(self):
        """Each tile PNG should have a companion .json sidecar."""
        from data.preprocessing import Preprocessor
        with tempfile.TemporaryDirectory() as tmpdir:
            pre = Preprocessor(tile_size=640, stride=576, out_dir=Path(tmpdir))
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            tiles = pre._tile_image(image, transform=None, product_name="test_prod")
            for t in tiles:
                json_path = t.with_suffix(".json")
                assert json_path.exists(), f"Missing sidecar for {t.name}"
                with open(json_path) as f:
                    meta = json.load(f)
                assert "row" in meta and "col" in meta


# ══════════════════════════════════════════════════════════════════════════════
# GeoDB
# ══════════════════════════════════════════════════════════════════════════════

class TestGeoDB:
    """Tests for geo/geodb.py"""

    def setup_method(self):
        from geo.geodb import GeoDB
        from models.detector import Detection

        # Use a temp file per test
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self._tmpdir) / "test.db"
        self.db = GeoDB(db_path=self.db_path)
        self.Detection = Detection

    def _make_det(self, track_id=1, lat=15.0, lon=114.0):
        return self.Detection(
            tile_path=Path("tile.png"),
            bbox_px=(100.0, 100.0, 200.0, 200.0),
            confidence=0.87,
            class_id=0,
            lat=lat,
            lon=lon,
            track_id=track_id,
            warship_type="Destroyer",
            warship_conf=0.75,
            timestamp="2024-01-15T06:00:00Z",
        )

    def test_insert_acquisition(self):
        self.db.insert_acquisition("ACQ1", "South_China_Sea", "S1")
        # Should not raise
        self.db.insert_acquisition("ACQ1", "South_China_Sea", "S1")  # duplicate → no error

    def test_insert_and_query_detections(self):
        self.db.insert_acquisition("ACQ2", "Persian_Gulf", "S1")
        dets = [self._make_det(track_id=i, lat=22.0 + i * 0.1) for i in range(3)]
        n = self.db.insert_detections(dets, "Persian_Gulf", "ACQ2")
        assert n == 3

        rows = self.db.query_detections(region="Persian_Gulf")
        assert len(rows) == 3

    def test_query_detections_region_filter(self):
        self.db.insert_acquisition("ACQ_SCS", "South_China_Sea", "S1")
        self.db.insert_acquisition("ACQ_PG",  "Persian_Gulf",    "S1")

        self.db.insert_detections([self._make_det()], "South_China_Sea", "ACQ_SCS")
        self.db.insert_detections([self._make_det()], "Persian_Gulf",    "ACQ_PG")

        scs = self.db.query_detections(region="South_China_Sea")
        pg  = self.db.query_detections(region="Persian_Gulf")
        assert len(scs) == 1
        assert len(pg)  == 1

    def test_upsert_and_query_tracks(self):
        self.db.insert_acquisition("ACQ3", "Mediterranean_Sea", "S1")
        det = self._make_det(track_id=42)
        self.db.insert_detections([det], "Mediterranean_Sea", "ACQ3")
        self.db.upsert_track(
            track_id=42,
            region="Mediterranean_Sea",
            detections=[det],
            trail=[{"lat": 15.0, "lon": 114.0, "timestamp": "2024-01-15T06:00:00Z"}],
        )

        tracks = self.db.query_tracks(region="Mediterranean_Sea")
        assert len(tracks) == 1
        assert tracks[0]["track_id"] == 42
        assert tracks[0]["warship_type"] == "Destroyer"
        assert isinstance(tracks[0]["trail"], list)

    def test_upsert_track_updates(self):
        """Upserting the same track twice should update, not duplicate."""
        self.db.insert_acquisition("ACQ4", "Black_Sea", "S1")
        det = self._make_det(track_id=7)
        self.db.insert_detections([det], "Black_Sea", "ACQ4")
        trail = [{"lat": 40.0, "lon": 30.0, "timestamp": "2024-01-15T06:00:00Z"}]
        self.db.upsert_track(7, "Black_Sea", [det], trail)
        self.db.upsert_track(7, "Black_Sea", [det], trail + [{"lat": 40.1, "lon": 30.1, "timestamp": "2024-01-15T12:00:00Z"}])

        tracks = self.db.query_tracks(region="Black_Sea")
        assert sum(1 for t in tracks if t["track_id"] == 7) == 1

    def test_summary_counts(self):
        self.db.insert_acquisition("ACQ5", "Baltic_Sea", "S1")
        self.db.insert_detections([self._make_det()], "Baltic_Sea", "ACQ5")
        s = self.db.summary()
        assert s["acquisitions"] >= 1
        assert s["detections"]   >= 1

    def test_export_geojson(self):
        self.db.insert_acquisition("ACQ6", "Arabian_Sea", "S1")
        self.db.insert_detections([self._make_det(lat=18.0, lon=65.0)],
                                  "Arabian_Sea", "ACQ6")
        out = Path(self._tmpdir) / "export.geojson"
        self.db.export_geojson(out, region="Arabian_Sea")
        assert out.exists()
        with open(out) as f:
            gj = json.load(f)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) >= 1
        feat = gj["features"][0]
        assert feat["geometry"]["type"] == "Point"
        assert len(feat["geometry"]["coordinates"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# AIS Fusion
# ══════════════════════════════════════════════════════════════════════════════

class TestAISFusion:
    """Tests for geo/ais_fusion.py"""

    def setup_method(self):
        from geo.ais_fusion import AISFusion, AISRecord, _haversine_km
        self.AISFusion    = AISFusion
        self.AISRecord    = AISRecord
        self._haversine   = _haversine_km

    def test_ais_record_fields(self):
        r = self.AISRecord({
            "MMSI": "123456789", "SHIPNAME": "USS Test",
            "FLAG": "US", "SHIPTYPE": 35,
            "LAT": 15.0, "LON": 114.0,
            "SPEED": 12.5,
        })
        assert r.mmsi == "123456789"
        assert r.name == "USS Test"
        assert r.is_military is True
        assert r.speed == pytest.approx(12.5)

    def test_ais_record_non_military(self):
        r = self.AISRecord({"SHIPTYPE": 70, "LAT": 0, "LON": 0})
        assert r.is_military is False

    def test_haversine_basic(self):
        dist = self._haversine(0, 0, 0, 1)
        assert 110 < dist < 112

    def test_nearest_vessel_found(self):
        fusion = self.AISFusion(backend="file", search_radius_km=5.0)
        vessels = [
            self.AISRecord({"MMSI": "111", "SHIPNAME": "Alpha", "FLAG": "US",
                            "SHIPTYPE": 35, "LAT": 15.01, "LON": 114.01}),
            self.AISRecord({"MMSI": "222", "SHIPNAME": "Beta",  "FLAG": "RU",
                            "SHIPTYPE": 35, "LAT": 20.0,  "LON": 120.0}),
        ]
        match = fusion._nearest_vessel(15.0, 114.0, vessels)
        assert match is not None
        assert match.mmsi == "111"

    def test_nearest_vessel_none_in_radius(self):
        fusion = self.AISFusion(backend="file", search_radius_km=1.0)
        vessels = [
            self.AISRecord({"MMSI": "333", "SHIPNAME": "Far Ship",
                            "SHIPTYPE": 35, "LAT": 30.0, "LON": 130.0}),
        ]
        match = fusion._nearest_vessel(15.0, 114.0, vessels)
        assert match is None

    def test_enrich_populates_fields(self):
        from models.detector import Detection

        fusion = self.AISFusion(backend="file", search_radius_km=10.0)
        vessel = self.AISRecord({
            "MMSI": "999888777", "SHIPNAME": "CNS Test",
            "FLAG": "CN", "SHIPTYPE": 35, "LAT": 15.0, "LON": 114.0,
        })

        # Monkey-patch _fetch_vessels to return our test vessel
        fusion._cache      = [vessel]
        fusion._cache_time = datetime.utcnow()

        det = Detection(Path("t.png"), (0, 0, 100, 100), 0.9, 0,
                        lat=15.0, lon=114.0)
        enriched = fusion.enrich([det])
        assert enriched[0].vessel_name == "CNS Test"
        assert enriched[0].vessel_flag == "CN"
        assert enriched[0].vessel_mmsi == "999888777"

    def test_enrich_no_position_skips(self):
        from models.detector import Detection
        fusion = self.AISFusion(backend="file")
        det = Detection(Path("t.png"), (0, 0, 100, 100), 0.9, 0)   # no lat/lon
        result = fusion.enrich([det])
        assert result[0].vessel_name is None


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline — Demo Mode (integration test, no network/GPU)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineDemo:
    """Integration test: run_demo() inserts synthetic detections into a temp DB."""

    def test_demo_inserts_detections(self):
        import tempfile
        from pipeline import WarshipPipeline
        from geo.geodb import GeoDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "demo.db"

            pipe = WarshipPipeline(
                enable_ais=False,
                enable_classifier=False,
            )
            # Override DB path so we don't touch the real one
            pipe.db = GeoDB(db_path=db_path)

            dets = pipe.run_demo()
            assert len(dets) > 0

            summary = pipe.db.summary()
            assert summary["detections"] > 0
            assert summary["acquisitions"] > 0

    def test_demo_creates_tracks(self):
        import tempfile
        from pipeline import WarshipPipeline
        from geo.geodb import GeoDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "demo_tracks.db"
            pipe = WarshipPipeline(enable_ais=False, enable_classifier=False)
            pipe.db = GeoDB(db_path=db_path)
            pipe.run_demo()

            tracks = pipe.db.query_tracks()
            assert len(tracks) > 0
            # Each track should have a warship type
            for t in tracks:
                assert t.get("warship_type") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Conftest / fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_detection():
    from models.detector import Detection
    return Detection(
        tile_path   = Path("sample_tile.png"),
        bbox_px     = (50.0, 80.0, 150.0, 180.0),
        confidence  = 0.91,
        class_id    = 0,
        lat         = 15.2342,
        lon         = 114.5678,
        track_id    = 3,
        warship_type = "Destroyer",
        warship_conf = 0.82,
        timestamp   = "2024-03-15T08:30:00Z",
    )


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        from geo.geodb import GeoDB
        yield GeoDB(db_path=Path(tmpdir) / "test.db")


def test_detection_fixture_sanity(sample_detection):
    assert sample_detection.confidence > 0.5
    assert sample_detection.lat is not None
    assert sample_detection.to_dict()["warship_type"] == "Destroyer"


def test_db_fixture_empty(temp_db):
    s = temp_db.summary()
    assert s["detections"]   == 0
    assert s["acquisitions"] == 0
    assert s["tracks"]       == 0

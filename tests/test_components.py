"""
tests/test_components.py — Tests for tracking, classification, georeferencing,
dataset preparation, the dashboard and the end-to-end pipeline.

No network, GPU or model weights are needed: the YOLO detector is replaced by
a stub wherever a full pipeline run is exercised.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import Detection, ShipDetector


def _det(lat, lon, conf=0.9, **kw):
    return Detection(Path("t.png"), (0, 0, 20, 20), conf, 0, lat=lat, lon=lon, **kw)


def _ship_tile(path, length_m, beam_m, heading=30, lat=15.0, lon=114.0, seed=0):
    """Synthetic SAR tile with one hull; returns the matching Detection."""
    import random
    from pipeline import _render_demo_tile
    return _render_demo_tile(
        path, {"length": length_m, "beam": beam_m, "heading": heading},
        lat, lon, random.Random(seed),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TestTracker:

    def setup_method(self):
        from models.tracker import ShipTracker
        self.ShipTracker = ShipTracker

    def test_every_detection_gets_a_unique_id(self):
        tr = self.ShipTracker()
        dets = tr.update([_det(15.0, 114.0), _det(16.0, 115.0), _det(17.0, 116.0)],
                         timestamp="2024-01-15T06:00:00Z")
        ids = [d.track_id for d in dets]
        assert None not in ids
        assert len(set(ids)) == 3

    def test_track_persists_across_acquisitions(self):
        tr = self.ShipTracker()
        a = tr.update([_det(15.0, 114.0), _det(18.0, 118.0)],
                      timestamp="2024-01-15T06:00:00Z")
        # 6 h later both ships have moved ~100 km; order is swapped
        b = tr.update([_det(18.5, 118.6), _det(15.6, 114.7)],
                      timestamp="2024-01-15T12:00:00Z")
        assert b[1].track_id == a[0].track_id
        assert b[0].track_id == a[1].track_id
        assert len(tr.get_track_trail(a[0].track_id)) == 2

    def test_crossing_ships_keep_identity(self):
        """Two ships sailing through one another's positions must not swap IDs."""
        tr = self.ShipTracker()
        # A heads north-east, B heads south-west along the same line
        a_path = [(15.0, 114.0), (16.0, 115.0), (17.0, 116.0), (18.0, 117.0)]
        b_path = [(18.0, 117.2), (17.0, 116.2), (16.0, 115.2), (15.0, 114.2)]
        ids = []
        for hour, (a, b) in zip((0, 6, 12, 18), zip(a_path, b_path)):
            dets = tr.update([_det(*a), _det(*b)],
                             timestamp=f"2024-01-15T{hour:02d}:00:00Z")
            ids.append((dets[0].track_id, dets[1].track_id))
        assert len(set(ids)) == 1

    def test_hull_length_breaks_ties(self):
        tr = self.ShipTracker()
        a = tr.update([_det(15.0, 114.0, length_m=330.0), _det(15.0, 114.1, length_m=90.0)],
                      timestamp="2024-01-15T06:00:00Z")
        # Both ships are now midway between the old fixes: only size tells them apart
        b = tr.update([_det(15.0, 114.05, length_m=92.0), _det(15.001, 114.05, length_m=325.0)],
                      timestamp="2024-01-15T07:00:00Z")
        assert b[1].track_id == a[0].track_id
        assert b[0].track_id == a[1].track_id

    def test_implausible_jump_starts_new_track(self):
        tr = self.ShipTracker()
        a = tr.update([_det(15.0, 114.0)], timestamp="2024-01-15T06:00:00Z")
        # 1 hour later, ~780 km away — no ship is that fast
        b = tr.update([_det(22.0, 114.0)], timestamp="2024-01-15T07:00:00Z")
        assert b[0].track_id != a[0].track_id

    def test_stale_track_not_continued(self):
        tr = self.ShipTracker(max_gap_hours=72)
        a = tr.update([_det(15.0, 114.0)], timestamp="2024-01-01T00:00:00Z")
        b = tr.update([_det(15.0, 114.0)], timestamp="2024-01-10T00:00:00Z")
        assert b[0].track_id != a[0].track_id

    def test_regions_do_not_share_tracks(self):
        tr = self.ShipTracker()
        a = tr.update([_det(15.0, 114.0)], "2024-01-15T06:00:00Z", region="A")
        b = tr.update([_det(15.0, 114.0)], "2024-01-15T06:00:00Z", region="B")
        assert b[0].track_id != a[0].track_id

    def test_two_detections_never_share_a_track(self):
        tr = self.ShipTracker()
        tr.update([_det(15.0, 114.0)], timestamp="2024-01-15T06:00:00Z")
        b = tr.update([_det(15.01, 114.0), _det(15.0, 114.01)],
                      timestamp="2024-01-15T06:30:00Z")
        assert b[0].track_id != b[1].track_id

    def test_detection_without_position_is_left_alone(self):
        tr = self.ShipTracker()
        d = Detection(Path("t.png"), (0, 0, 10, 10), 0.9, 0)
        assert tr.update([d], "2024-01-15T06:00:00Z")[0].track_id is None

    def test_ids_persist_across_runs_via_db(self, tmp_path):
        from geo.geodb import GeoDB
        db = GeoDB(db_path=tmp_path / "t.db")

        run1 = self.ShipTracker(db=db)
        dets = run1.update([_det(15.0, 114.0, timestamp="2024-01-15T06:00:00Z")],
                           "2024-01-15T06:00:00Z", region="R")
        db.insert_acquisition("A1", "R", "S1")
        db.insert_detections(dets, "R", "A1")
        db.rebuild_tracks([dets[0].track_id])

        # A fresh process: continues the same track, and new ships get new IDs
        run2 = self.ShipTracker(db=db)
        later = run2.update([_det(15.3, 114.3), _det(30.0, 130.0)],
                            "2024-01-15T12:00:00Z", region="R")
        assert later[0].track_id == dets[0].track_id
        assert later[1].track_id > dets[0].track_id


# ══════════════════════════════════════════════════════════════════════════════
# Classifier
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifier:

    def setup_method(self):
        from models.classifier import (
            WarshipClassifier, classify_by_size, measure_ship, tile_gsd_m,
        )
        self.WarshipClassifier = WarshipClassifier
        self.classify_by_size  = classify_by_size
        self.measure_ship      = measure_ship
        self.tile_gsd_m        = tile_gsd_m

    def _clf(self, tmp_path):
        # Point at a weights file that doesn't exist → size-based method
        return self.WarshipClassifier(weights=tmp_path / "none.pt")

    @pytest.mark.parametrize("length,beam,expected", [
        (333, 76, "Aircraft Carrier"),
        (257, 32, "Amphibious Assault"),
        (173, 17, "Cruiser"),
        (155, 20, "Destroyer"),
        (125, 16, "Frigate"),
        (80,  12, "Patrol / Corvette"),
    ])
    def test_size_bands(self, length, beam, expected):
        wtype, conf = self.classify_by_size(length, beam)
        assert wtype == expected
        assert 0.4 <= conf <= 0.75

    def test_size_out_of_range_is_unknown(self):
        assert self.classify_by_size(8, 3)[0] == "Unknown Warship"
        assert self.classify_by_size(450, 60)[0] == "Unknown Warship"

    def test_measure_ship_recovers_dimensions(self, tmp_path):
        from PIL import Image
        d = _ship_tile(tmp_path / "t.png", 250, 40, heading=60)
        img = np.asarray(Image.open(d.tile_path))
        length, beam = self.measure_ship(img, gsd_m=10.0)
        assert length == pytest.approx(250, abs=20)
        assert beam == pytest.approx(40, abs=12)

    def test_measure_ship_flat_crop(self):
        assert self.measure_ship(np.full((40, 40, 3), 30, np.uint8), 10.0) is None

    @pytest.mark.parametrize("length,beam,expected", [
        (333, 76, "Aircraft Carrier"),
        (155, 20, "Destroyer"),
        (125, 16, "Frigate"),
        (80,  12, "Patrol / Corvette"),
    ])
    @pytest.mark.parametrize("heading", [0, 35, 90, 140])
    def test_classify_detections_from_tile(self, tmp_path, length, beam,
                                           expected, heading):
        d = _ship_tile(tmp_path / "t.png", length, beam, heading=heading)
        self._clf(tmp_path).classify_detections([d])
        # Hull size yields a size class, never a warship identification
        assert d.size_class == expected
        assert d.warship_type == "Unidentified Vessel"
        assert d.warship_conf is None
        assert d.type_source == "size"
        assert d.length_m == pytest.approx(length, abs=25)

    def test_method_is_size_without_weights(self, tmp_path):
        assert self._clf(tmp_path).method == "size"

    def test_missing_tile_gives_unknown(self, tmp_path):
        d = Detection(tmp_path / "missing.png", (10, 10, 50, 50), 0.9, 0)
        self._clf(tmp_path).classify_detections([d])
        assert d.warship_type == "Unidentified Vessel"
        assert d.warship_conf is None and d.size_class is None

    def test_gsd_scales_measurement(self, tmp_path):
        # Same pixels at 3 m/px are a third of the size → a smaller class
        d = _ship_tile(tmp_path / "t.png", 333, 76)
        self._clf(tmp_path).classify_detections([d], gsd_m=3.0)
        assert d.length_m == pytest.approx(100, abs=10)
        assert d.size_class == "Patrol / Corvette"

    def test_tile_gsd_from_sidecar(self, tmp_path):
        d = _ship_tile(tmp_path / "t.png", 150, 20)
        geo = json.loads(d.tile_path.with_suffix(".json").read_text())
        assert self.tile_gsd_m(geo) == pytest.approx(10.0)
        del geo["gsd_m"]      # legacy sidecar: derive from the corners
        assert self.tile_gsd_m(geo) == pytest.approx(10.0, rel=0.05)
        assert self.tile_gsd_m(None) == pytest.approx(10.0)

    def test_corrupt_weights_fall_back_to_size(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        clf = self.WarshipClassifier(weights=bad)
        assert clf.method == "size"

    def test_cnn_train_and_predict_roundtrip(self, tmp_path):
        """Train for one epoch on a tiny synthetic set, then load + predict."""
        pytest.importorskip("torch")
        from PIL import Image
        from models.classifier import train_classifier

        rng = np.random.default_rng(0)
        for cls in ("Destroyer", "frigate"):          # folder names are fuzzy-matched
            (tmp_path / "data" / cls).mkdir(parents=True)
            for i in range(3):
                arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(tmp_path / "data" / cls / f"{i}.png")

        out = train_classifier(tmp_path / "data", epochs=1, batch_size=2,
                               pretrained=False, output=tmp_path / "clf.pt")
        assert out.exists()

        clf = self.WarshipClassifier(weights=out, conf_min=0.0)
        assert clf.method == "cnn"
        d = _ship_tile(tmp_path / "t.png", 155, 20)
        clf.classify_detections([d], gsd_m=1.0)      # sharp imagery → CNN is used
        assert d.type_source == "cnn"
        assert d.warship_type in clf.classes
        assert 0.0 < d.warship_conf <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing — georeferencing
# ══════════════════════════════════════════════════════════════════════════════

class TestGeoreferencing:

    def setup_method(self):
        pytest.importorskip("rasterio")
        from data.preprocessing import Preprocessor
        self.Preprocessor = Preprocessor

    def _sidecars(self, tiles):
        return [json.loads(t.with_suffix(".json").read_text()) for t in tiles]

    def test_geographic_geotiff(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin
        tif = tmp_path / "scene.tif"
        data = np.random.default_rng(0).integers(1, 500, (700, 900)).astype(np.uint16)
        with rasterio.open(tif, "w", driver="GTiff", height=700, width=900, count=1,
                           dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(114.0, 20.0, 1e-4, 1e-4)) as dst:
            dst.write(data, 1)

        tiles = self.Preprocessor(out_dir=tmp_path / "tiles").process_geotiff(tif)
        assert len(tiles) == 4
        meta = self._sidecars(tiles)[0]
        assert meta["lat_tl"] == pytest.approx(20.0)
        assert meta["lon_tl"] == pytest.approx(114.0)
        assert meta["lat_br"] == pytest.approx(20.0 - 0.064)
        assert meta["lon_br"] == pytest.approx(114.0 + 0.064)
        assert meta["gsd_m"] == pytest.approx(10.4, abs=0.3)

    def test_projected_crs_is_converted_to_latlon(self, tmp_path):
        """Sentinel-2 is UTM: sidecars must hold degrees, not metres."""
        import rasterio
        from rasterio.transform import from_origin
        tif = tmp_path / "utm.tif"
        data = np.random.default_rng(0).integers(1, 500, (640, 640)).astype(np.uint16)
        with rasterio.open(tif, "w", driver="GTiff", height=640, width=640, count=1,
                           dtype="uint16", crs="EPSG:32650",       # UTM 50N
                           transform=from_origin(500000, 1700000, 10, 10)) as dst:
            dst.write(data, 1)

        tiles = self.Preprocessor(out_dir=tmp_path / "tiles").process_geotiff(tif)
        meta = self._sidecars(tiles)[0]
        assert meta["lon_tl"] == pytest.approx(117.0, abs=0.01)   # zone 50 meridian
        assert 15.0 < meta["lat_tl"] < 16.0
        assert meta["gsd_m"] == pytest.approx(10.0, abs=0.1)

    def test_gcp_georeferencing_rotated_scene(self, tmp_path):
        """Sentinel-1 GRD has GCPs instead of a transform, on a rotated grid."""
        import rasterio
        from rasterio.control import GroundControlPoint
        tif = tmp_path / "s1-iw-grd-vv-test.tiff"
        H = W = 640

        def truth(col, row):       # rotated ~12° relative to north
            return (114.0 + 1e-4 * col + 2e-5 * row,
                    20.0 - 1e-4 * row + 2e-5 * col)

        gcps = []
        for r in (0, H // 2, H):
            for c in (0, W // 2, W):
                lon, lat = truth(c, r)
                gcps.append(GroundControlPoint(row=r, col=c, x=lon, y=lat))

        with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="uint16") as dst:
            dst.write(np.random.default_rng(0).integers(1, 500, (H, W)).astype(np.uint16), 1)
            dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))

        tiles = self.Preprocessor(out_dir=tmp_path / "tiles").process_geotiff(tif)
        meta = self._sidecars(tiles)[0]
        for key, (c, r) in {"tl": (0, 0), "tr": (W, 0), "bl": (0, H), "br": (W, H)}.items():
            lon, lat = truth(c, r)
            assert meta[f"lon_{key}"] == pytest.approx(lon, abs=1e-5)
            assert meta[f"lat_{key}"] == pytest.approx(lat, abs=1e-5)

        # Bilinear interpolation must follow the rotation at an off-diagonal pixel
        lat, lon = ShipDetector._pixel_to_geo(500, 100, meta)
        t_lon, t_lat = truth(500, 100)
        assert lon == pytest.approx(t_lon, abs=1e-5)
        assert lat == pytest.approx(t_lat, abs=1e-5)

    def test_safe_product_prefers_vh_greyscale(self, tmp_path):
        import rasterio
        from rasterio.transform import from_origin
        safe = tmp_path / "S1A_IW_GRDH_1SDV_20240115T101010_TEST.SAFE"
        (safe / "measurement").mkdir(parents=True)
        rng = np.random.default_rng(1)
        for pol, scale in (("vv", 300), ("vh", 80)):
            with rasterio.open(safe / "measurement" / f"s1a-iw-grd-{pol}-001.tiff", "w",
                               driver="GTiff", height=640, width=640, count=1,
                               dtype="uint16", crs="EPSG:4326",
                               transform=from_origin(50.0, 26.0, 1e-4, 1e-4)) as dst:
                dst.write(rng.integers(1, scale, (640, 640)).astype(np.uint16), 1)

        from PIL import Image
        tiles = self.Preprocessor(out_dir=tmp_path / "tiles").process_product(safe, "S1")
        assert len(tiles) == 1
        img = np.asarray(Image.open(tiles[0]))
        assert img.shape == (640, 640, 3)
        # Greyscale (what the detector was trained on), taken from the VH band
        assert np.array_equal(img[..., 0], img[..., 1])
        assert np.array_equal(img[..., 0], img[..., 2])
        assert img.max() > 0

        import shutil
        vh_only = tmp_path / "S1A_IW_GRDH_1SDV_20240115T101010_VH.SAFE"
        (vh_only / "measurement").mkdir(parents=True)
        shutil.copy2(safe / "measurement" / "s1a-iw-grd-vh-001.tiff",
                     vh_only / "measurement" / "s1a-iw-grd-vh-001.tiff")
        ref = self.Preprocessor(out_dir=tmp_path / "t2").process_product(vh_only, "S1")
        assert np.array_equal(img, np.asarray(Image.open(ref[0])))

    def test_small_image_is_padded(self, tmp_path):
        pre = self.Preprocessor(out_dir=tmp_path)
        tiles = pre._tile_image(np.full((300, 400, 3), 9, np.uint8), None, "small")
        assert len(tiles) == 1

    def test_empty_tiles_skipped(self, tmp_path):
        pre = self.Preprocessor(out_dir=tmp_path)
        image = np.zeros((640, 1280, 3), np.uint8)
        image[:, :640] = 50
        tiles = pre._tile_image(image, None, "p", skip_empty=True)
        assert 0 < len(tiles) < 3


# ══════════════════════════════════════════════════════════════════════════════
# GeoDB — track rebuild, timestamps, demo cleanup
# ══════════════════════════════════════════════════════════════════════════════

class TestGeoDBTracks:

    def setup_method(self):
        from geo.geodb import GeoDB, normalize_ts
        self.GeoDB = GeoDB
        self.normalize_ts = normalize_ts

    def test_normalize_ts(self):
        assert self.normalize_ts("2024-01-15T06:00:00Z") == "2024-01-15T06:00:00Z"
        assert self.normalize_ts("2024-01-15T06:00:00.123456") == "2024-01-15T06:00:00Z"
        assert self.normalize_ts("2024-01-15T08:00:00+02:00") == "2024-01-15T06:00:00Z"
        assert self.normalize_ts("") == ""
        assert self.normalize_ts(None) is None

    def test_rebuild_tracks_accumulates(self, tmp_path):
        db = self.GeoDB(db_path=tmp_path / "t.db")
        for i, (ts, wtype) in enumerate([
            ("2024-01-15T06:00:00Z", "Destroyer"),
            ("2024-01-15T12:00:00Z", "Unknown Warship"),
            ("2024-01-15T18:00:00Z", "Destroyer"),
        ]):
            db.insert_acquisition(f"A{i}", "R", "S1")
            db.insert_detections(
                [_det(15.0 + i * 0.1, 114.0, track_id=7, timestamp=ts,
                      warship_type=wtype)], "R", f"A{i}")
            db.rebuild_tracks([7])

        (track,) = db.query_tracks()
        assert track["n_detections"] == 3
        assert track["first_seen"] == "2024-01-15T06:00:00Z"
        assert track["last_seen"] == "2024-01-15T18:00:00Z"
        assert track["warship_type"] == "Destroyer"
        assert [p["lat"] for p in track["trail"]] == pytest.approx([15.0, 15.1, 15.2])

    def test_since_filter_mixed_formats(self, tmp_path):
        db = self.GeoDB(db_path=tmp_path / "t.db")
        db.insert_acquisition("A", "R", "S1")
        db.insert_detections([
            _det(1, 1, timestamp="2024-01-15T06:00:00Z"),
            _det(2, 2, timestamp="2024-01-15T06:00:00.500000"),
            _det(3, 3, timestamp="2024-01-14T06:00:00+00:00"),
        ], "R", "A")
        assert len(db.query_detections(since="2024-01-15T00:00:00")) == 2

    def test_clear_demo_keeps_real_data(self, tmp_path):
        db = self.GeoDB(db_path=tmp_path / "t.db")
        db.insert_acquisition("REAL", "R", "S1")
        db.insert_acquisition("DEMO_0", "R", "DEMO")
        db.insert_detections([_det(1, 1, track_id=1)], "R", "REAL")
        db.insert_detections([_det(2, 2, track_id=2)], "R", "DEMO_0")
        db.rebuild_tracks([1, 2])

        assert db.clear_demo() == 1
        assert [t["track_id"] for t in db.query_tracks()] == [1]
        assert db.summary()["detections"] == 1

    def test_migrates_old_schema(self, tmp_path):
        import sqlite3
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "acquisition_id TEXT, track_id INTEGER, tile_path TEXT, lat REAL, "
                     "lon REAL, bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL, "
                     "confidence REAL, class_name TEXT, warship_type TEXT, "
                     "warship_conf REAL, vessel_name TEXT, vessel_mmsi TEXT, "
                     "vessel_flag TEXT, timestamp TEXT, region TEXT)")
        conn.commit()
        conn.close()

        db = self.GeoDB(db_path=path)
        db.insert_acquisition("A", "R", "S1")
        db.insert_detections([_det(1, 1, length_m=150.0, beam_m=20.0)], "R", "A")
        assert db.query_detections()[0]["length_m"] == 150.0

    def test_geojson_keeps_zero_coordinates(self, tmp_path):
        db = self.GeoDB(db_path=tmp_path / "t.db")
        db.insert_acquisition("A", "R", "S1")
        db.insert_detections([_det(0.0, 0.0)], "R", "A")
        out = db.export_geojson(tmp_path / "o.geojson")
        assert len(json.loads(out.read_text())["features"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# AIS — civilian relabelling and robustness
# ══════════════════════════════════════════════════════════════════════════════

class TestAISCivilian:

    def test_cargo_ship_relabelled(self, tmp_path):
        from config import CIVILIAN_LABEL
        from geo.ais_fusion import AISFusion
        ais_file = tmp_path / "ais.json"
        ais_file.write_text(json.dumps([
            {"MMSI": "1", "SHIPNAME": "EVER GIVEN", "FLAG": "PA", "SHIPTYPE": 70,
             "LAT": 15.0, "LON": 114.0},
            {"MMSI": "2", "SHIPNAME": "WARSHIP", "FLAG": "US", "SHIPTYPE": 35,
             "LAT": 16.0, "LON": 115.0},
        ]))
        fusion = AISFusion(backend="file", ais_file=ais_file)
        cargo = _det(15.001, 114.001, warship_type="Aircraft Carrier", warship_conf=0.6)
        navy  = _det(16.001, 115.001, warship_type="Destroyer", warship_conf=0.6)
        dark  = _det(20.0, 120.0, warship_type="Frigate", warship_conf=0.6)
        fusion.enrich([cargo, navy, dark])

        assert cargo.warship_type == CIVILIAN_LABEL
        assert cargo.vessel_name == "EVER GIVEN"
        assert navy.warship_type == "Destroyer"
        assert dark.warship_type == "Frigate" and dark.vessel_mmsi is None

    def test_military_ais_plus_size_gives_type(self, tmp_path):
        from geo.ais_fusion import AISFusion
        ais_file = tmp_path / "ais.json"
        ais_file.write_text(json.dumps([
            {"MMSI": "1", "SHIPNAME": "NAVY ONE", "FLAG": "US", "SHIPTYPE": 35,
             "LAT": 15.0, "LON": 114.0},
            {"MMSI": "2", "SHIPNAME": "NAVY TWO", "FLAG": "US", "SHIPTYPE": 35,
             "LAT": 16.0, "LON": 115.0},
        ]))
        sized = _det(15.0, 114.0, warship_type="Unidentified Vessel",
                     size_class="Destroyer", size_conf=0.7, type_source="size")
        unsized = _det(16.0, 115.0, warship_type="Unidentified Vessel")
        dark = _det(20.0, 120.0, warship_type="Unidentified Vessel",
                    size_class="Aircraft Carrier", size_conf=0.6, type_source="size")
        AISFusion(backend="file", ais_file=ais_file).enrich([sized, unsized, dark])

        assert (sized.warship_type, sized.warship_conf, sized.type_source) == \
            ("Destroyer", 0.7, "ais+size")
        assert unsized.warship_type == "Unknown Warship"
        # No AIS → a carrier-sized hull is still not called a carrier
        assert dark.warship_type == "Unidentified Vessel"

    def test_crowded_waters_marked_civilian(self):
        from geo.ais_fusion import label_crowded_waters
        # Ten ships anchored ~550 m apart, one lone ship 100 km away, and one
        # ship in the crowd that AIS has already identified
        crowd = [_det(1.20 + 0.005 * i, 103.90, warship_type="Unidentified Vessel",
                      size_class="Aircraft Carrier") for i in range(10)]
        lone = _det(2.20, 104.90, warship_type="Unidentified Vessel",
                    size_class="Destroyer")
        known = _det(1.22, 103.901, warship_type="Destroyer", type_source="ais+size")
        label_crowded_waters(crowd + [lone, known])

        assert {d.warship_type for d in crowd} == {"Civilian Vessel"}
        assert {d.type_source for d in crowd} == {"context"}
        assert crowd[0].size_class == "Aircraft Carrier"      # size class is kept
        assert lone.warship_type == "Unidentified Vessel"
        assert known.warship_type == "Destroyer"

    def test_crowd_rule_can_be_disabled(self):
        from geo.ais_fusion import label_crowded_waters
        crowd = [_det(1.20 + 0.005 * i, 103.90, warship_type="Unidentified Vessel")
                 for i in range(10)]
        label_crowded_waters(crowd, min_neighbours=0)
        assert {d.warship_type for d in crowd} == {"Unidentified Vessel"}

    def test_csv_with_blank_fields(self, tmp_path):
        from geo.ais_fusion import AISFusion
        csv_file = tmp_path / "ais.csv"
        csv_file.write_text("MMSI,SHIPNAME,SHIPTYPE,LAT,LON,SPEED\n"
                            "1,ALPHA,,15.0,114.0,\n"
                            "2,NOPOS,70,,,3\n")
        fusion = AISFusion(backend="file", ais_file=csv_file)
        d = _det(15.0, 114.0)
        fusion.enrich([d])
        assert d.vessel_name == "ALPHA"

    def test_cache_not_reused_for_other_area(self):
        from geo.ais_fusion import AISFusion
        fusion = AISFusion(backend="file", ais_file=None)
        calls = []
        fusion._read_ais_file = lambda p: calls.append(1) or []
        fusion._fetch_vessels((0, 0, 1, 1))
        fusion._fetch_vessels((50, 50, 51, 51))
        assert len(calls) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Dataset preparation
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetPrep:

    def test_prepare_flat_yolo_layout(self, tmp_path):
        """The SAR-Ship download is flat: <name>.jpg + <name>.txt (YOLO)."""
        from PIL import Image
        from train.finetune import prepare_sar_ship_dataset
        from data.dataset import DatasetManager

        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(10):
            Image.new("L", (256, 256)).save(raw / f"chip{i}.jpg")
            (raw / f"chip{i}.txt").write_text("0 0.5 0.5 0.1 0.2\n")
        Image.new("L", (256, 256)).save(raw / "orphan.jpg")      # no label

        out = tmp_path / "out"
        prepare_sar_ship_dataset(raw, out)

        stats = DatasetManager(out).stats(out)
        assert stats["train"]["images"] == 8
        assert stats["val"]["images"] == 2
        assert stats["total_instances"] == 10
        assert DatasetManager.verify_labels(out) == []
        yaml = (out / "data.yaml").read_text()
        assert "nc: 1" in yaml and "\\" not in yaml

    def test_prepare_voc_layout(self, tmp_path):
        from PIL import Image
        from data.dataset import DatasetManager
        raw = tmp_path / "raw"
        (raw / "Annotations").mkdir(parents=True)
        (raw / "JPEGImages").mkdir()
        for i in range(5):
            Image.new("L", (100, 100)).save(raw / "JPEGImages" / f"im{i}.jpg")
            (raw / "Annotations" / f"im{i}.xml").write_text(
                "<annotation><size><width>100</width><height>100</height></size>"
                "<object><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax>"
                "<ymax>60</ymax></bndbox></object></annotation>")
        out = DatasetManager(tmp_path / "ds").prepare_sar_ship(raw, tmp_path / "out")
        label = next((out / "labels" / "train").glob("*.txt")).read_text()
        assert label == "0 0.200000 0.400000 0.200000 0.400000"

    def test_resolve_device_never_requests_missing_gpu(self):
        torch = pytest.importorskip("torch")
        from train.finetune import resolve_device
        if not torch.cuda.is_available():
            assert resolve_device("auto") == "cpu"
            assert resolve_device("0") == "cpu"
        assert resolve_device("cpu") == "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline (stub detector — no YOLO weights needed)
# ══════════════════════════════════════════════════════════════════════════════

def _make_pipeline(tmp_path, **kw):
    from pipeline import WarshipPipeline
    from geo.geodb import GeoDB
    from data.preprocessing import Preprocessor
    from models.classifier import WarshipClassifier

    pipe = WarshipPipeline(enable_ais=kw.pop("enable_ais", False),
                           export_geojson=False, **kw)
    pipe.db = GeoDB(db_path=tmp_path / "pipe.db")
    pipe.preprocessor = Preprocessor(out_dir=tmp_path / "tiles")
    if pipe.classifier is not None:
        pipe.classifier = WarshipClassifier(weights=tmp_path / "none.pt")
    return pipe


class TestDemoWithClassifier:

    def test_demo_classifies_every_ship_correctly(self, tmp_path):
        pipe = _make_pipeline(tmp_path)
        dets = pipe.run_demo()
        assert len(dets) == 21

        tracks = pipe.db.query_tracks()
        assert len(tracks) == 7
        # Warship types only where AIS backs them; dark ships stay unidentified
        by_name = {t["vessel_name"]: t["warship_type"] for t in tracks if t["vessel_name"]}
        assert by_name == {
            "DEMO CARRIER": "Aircraft Carrier", "DEMO DESTROYER": "Destroyer",
            "DEMO AMPHIB": "Amphibious Assault",
            "DEMO TANKER": "Civilian Vessel", "DEMO CARGO": "Civilian Vessel",
        }
        dark = [t for t in tracks if not t["vessel_name"]]
        assert [t["warship_type"] for t in dark] == ["Unidentified Vessel"] * 2
        assert {d.size_class for d in dets if d.type_source == "size"} == {
            "Frigate", "Patrol / Corvette"}
        for t in tracks:
            assert t["n_detections"] == 3
            assert len(t["trail"]) == 3          # full trail → drawn on the map
        # Each track is one ship throughout (the carrier and frigate cross paths)
        for t in tracks:
            sizes = {d.size_class for d in dets if d.track_id == t["track_id"]}
            assert len(sizes) == 1
        for d in dets:
            assert d.length_m and d.beam_m

    def test_demo_is_idempotent(self, tmp_path):
        pipe = _make_pipeline(tmp_path)
        pipe.run_demo()
        pipe.run_demo()
        s = pipe.db.summary()
        assert (s["detections"], s["tracks"], s["acquisitions"]) == (21, 7, 3)

    def test_demo_data_is_recent(self, tmp_path):
        """The dashboard defaults to the last 7 days — demo data must show up."""
        from datetime import datetime, timedelta, timezone
        pipe = _make_pipeline(tmp_path)
        pipe.run_demo()
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        assert len(pipe.db.query_detections(since=since)) == 21


class TestPipelineLocal:
    """Full run on synthetic Sentinel-1 scenes with the YOLO model stubbed."""

    SHIPS = [   # (row, col, length_m, beam_m, heading, type)
        (300,  400, 333, 76,  20, "Aircraft Carrier"),
        (900,  700, 155, 20, 110, "Destroyer"),
        (500, 1000, 125, 16,  60, "Frigate"),
    ]

    def _scene(self, path: Path, shift_px: int = 0):
        """1280×1280 SAR-like GeoTIFF at ~10 m/px with three bright hulls."""
        import rasterio
        from rasterio.transform import from_origin
        rng = np.random.default_rng(0)
        img = rng.gamma(4.0, 20.0, (1280, 1280))
        yy, xx = np.mgrid[0:1280, 0:1280]
        for r, c, L, B, hdg, _ in self.SHIPS:
            th = math.radians(hdg)
            dx, dy = xx - (c + shift_px), yy - r
            along  = dx * math.sin(th) - dy * math.cos(th)
            across = dx * math.cos(th) + dy * math.sin(th)
            img[(np.abs(along) <= L / 20) & (np.abs(across) <= max(B / 20, 0.5))] = 6000
        deg = 10 / 111_320
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=1280, width=1280,
                           count=1, dtype="uint16", crs="EPSG:4326",
                           transform=from_origin(114.0, 15.0, deg, deg)) as dst:
            dst.write(img.astype(np.uint16), 1)

    @staticmethod
    def _stub_detector(pipe):
        """Replace YOLO with a bright-blob detector producing the same output."""
        from PIL import Image
        from scipy import ndimage

        def run_batch(tile_paths, batch_size=8):
            out = []
            for tp in tile_paths:
                gray = np.asarray(Image.open(tp))[..., 0]
                labels, _ = ndimage.label(gray > 240, structure=np.ones((3, 3)))
                geo = ShipDetector._load_geo(tp)
                for sl in ndimage.find_objects(labels):
                    if (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start) < 20:   # speckle
                        continue
                    box = (float(sl[1].start), float(sl[0].start),
                           float(sl[1].stop), float(sl[0].stop))
                    lat, lon = ShipDetector._pixel_to_geo(
                        (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, geo)
                    out.append(Detection(tp, box, 0.9, 0, lat=lat, lon=lon))
            return out

        pipe.detector.run_batch = run_batch

    def test_local_product_end_to_end(self, tmp_path):
        pytest.importorskip("rasterio")
        pipe = _make_pipeline(tmp_path)
        self._stub_detector(pipe)

        safe1 = tmp_path / "S1A_IW_GRDH_1SDV_20240115T060000_AAAA.SAFE"
        self._scene(safe1 / "measurement" / "s1a-iw-grd-vv-001.tiff")
        dets = pipe.run_local(safe1, region="custom", sensor="S1")

        # Overlapping tiles see ships twice; NMS must leave exactly three
        assert len(dets) == 3
        assert sorted(d.size_class for d in dets) == sorted(s[5] for s in self.SHIPS)
        assert {d.warship_type for d in dets} == {"Unidentified Vessel"}
        assert all(d.track_id is not None for d in dets)
        assert all(d.timestamp == "2024-01-15T06:00:00Z" for d in dets)

        # Positions must match where the ships were drawn
        deg = 10 / 111_320
        for r, c, *_rest, wtype in self.SHIPS:
            d = next(x for x in dets if x.size_class == wtype)
            assert d.lat == pytest.approx(15.0 - r * deg, abs=3 * deg)
            assert d.lon == pytest.approx(114.0 + c * deg, abs=3 * deg)

        # Second pass 6 h later, ships moved 40 px (400 m) east → same tracks
        safe2 = tmp_path / "S1A_IW_GRDH_1SDV_20240115T120000_BBBB.SAFE"
        self._scene(safe2 / "measurement" / "s1a-iw-grd-vv-001.tiff", shift_px=40)
        dets2 = pipe.run_local(safe2, region="custom", sensor="S1")

        by_type1 = {d.size_class: d.track_id for d in dets}
        by_type2 = {d.size_class: d.track_id for d in dets2}
        assert by_type1 == by_type2

        tracks = pipe.db.query_tracks(region="custom")
        assert len(tracks) == 3
        assert all(t["n_detections"] == 2 and len(t["trail"]) == 2 for t in tracks)
        assert pipe.db.summary()["acquisitions"] == 2

    def test_missing_local_dir_raises(self, tmp_path):
        pipe = _make_pipeline(tmp_path, enable_classifier=False)
        with pytest.raises(FileNotFoundError):
            pipe.run_local(tmp_path / "nope.SAFE", region="custom")


class TestSizeFilter:

    def test_island_sized_boxes_are_dropped(self):
        ship   = Detection(Path("t.png"), (10, 10, 40, 20), 0.9, 0, lat=1.0, lon=1.0)
        island = Detection(Path("t.png"), (100, 100, 190, 160), 0.9, 0, lat=1.1, lon=1.1)
        # No sidecar → 10 m/px: the ship is 300 m long, the "island" 900 m
        assert ShipDetector.filter_by_size([ship, island]) == [ship]


class TestPipelineHelpers:

    def test_timestamp_from_product_name(self):
        from pipeline import _timestamp_from_name, _extract_timestamp
        name = "S1A_IW_GRDH_1SDV_20240115T101010_20240115T101035_052_0AB_1C2D.SAFE"
        assert _timestamp_from_name(name) == "2024-01-15T10:10:10Z"
        assert _timestamp_from_name("no_date_here") is None
        assert _extract_timestamp(
            {"ContentDate": {"Start": "2024-02-01T00:00:00.000Z"}}
        ) == "2024-02-01T00:00:00.000Z"
        assert _extract_timestamp({"Name": name}) == "2024-01-15T10:10:10Z"

    def test_missing_credentials_give_clear_error(self, monkeypatch):
        import data.ingestion as ing
        monkeypatch.setattr(ing, "COPERNICUS_USER", "your_username")
        with pytest.raises(RuntimeError, match="COPERNICUS_USER"):
            ing.SentinelIngester()._get_token()


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard (headless, via Streamlit's AppTest)
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboard:

    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        pytest.importorskip("streamlit")
        pytest.importorskip("streamlit_folium")
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        import config, geo.geodb

        # Point everything at a throw-away DB / tile directory
        monkeypatch.setattr(config, "DB_PATH", tmp_path / "dash.db")
        monkeypatch.setattr(geo.geodb.GeoDB.__init__, "__defaults__",
                            (tmp_path / "dash.db",))
        monkeypatch.setattr(config, "TILES_DIR", tmp_path / "tiles")
        import data.preprocessing as pp
        monkeypatch.setattr(pp.Preprocessor.__init__, "__defaults__",
                            (pp.TILE_SIZE_PX, pp.TILE_STRIDE, tmp_path / "tiles"))
        st.cache_data.clear()
        st.cache_resource.clear()

        script = str(Path(__file__).parent.parent / "dashboard.py")
        yield AppTest.from_file(script, default_timeout=120)
        st.cache_data.clear()
        st.cache_resource.clear()

    def test_renders_empty(self, app):
        app.run()
        assert not app.exception
        assert any("No detections" in i.value for i in app.info)

    def test_demo_button_populates_everything(self, app):
        app.run()
        app.sidebar.button[0].click().run()
        assert not app.exception

        # Detections table shows all 15 classified rows
        det_table = app.dataframe[0].value
        assert len(det_table) == 21
        assert set(det_table["warship_type"]) == {
            "Aircraft Carrier", "Destroyer", "Amphibious Assault",
            "Civilian Vessel", "Unidentified Vessel",
        }
        assert {"type_source", "size_class"} <= set(det_table.columns)
        # Tracks table: 7 tracks with 3 trail points each
        track_table = app.dataframe[1].value
        assert len(track_table) == 7
        assert set(track_table["Trail Points"]) == {3}

    def test_type_filter(self, app):
        app.run()
        app.sidebar.button[0].click().run()
        app.sidebar.multiselect[0].set_value(["Destroyer"]).run()
        assert not app.exception
        assert set(app.dataframe[0].value["warship_type"]) == {"Destroyer"}
        assert len(app.dataframe[1].value) == 1

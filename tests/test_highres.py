"""
tests/test_highres.py — Tests for the high-resolution optical mode
(sensor "NAIP"): imagery windowing, multi-scale tiling, oriented-box
detections, hull-outline duplicate removal and CNN classification.

Network calls and the detector model are replaced by local stand-ins.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import Detection, ShipDetector


def _rgb_tif(path: Path, h=1500, w=2000, res=0.6, crs="EPSG:26918",
             x0=380000.0, y0=4092000.0):
    """A fake NAIP quad: uint8 RGB(+NIR), UTM, 0.6 m pixels."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    rng = np.random.default_rng(0)
    data = rng.integers(10, 250, (4, h, w)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=4,
                       dtype="uint8", crs=crs,
                       transform=from_origin(x0, y0, res, res)) as dst:
        dst.write(data)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Imagery download
# ══════════════════════════════════════════════════════════════════════════════

class TestNaipIngestion:

    def _ingester(self, monkeypatch, **kw):
        from data.ingestion import PlanetaryComputerIngester
        ing = PlanetaryComputerIngester(**kw)
        monkeypatch.setattr(ing, "_sign", lambda href: href)
        return ing

    def _aoi(self, tif, col0, row0, col1, row1):
        """lat/lon bbox of a pixel rectangle of the fake quad."""
        import rasterio
        from rasterio.warp import transform
        with rasterio.open(tif) as src:
            xs, ys = zip(src.transform * (col0, row0), src.transform * (col1, row1))
            lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
        return (min(lats), min(lons), max(lats), max(lons))

    def test_window_is_resampled_and_georeferenced(self, tmp_path, monkeypatch):
        import rasterio
        from rasterio.warp import transform
        tif = _rgb_tif(tmp_path / "remote.tif")
        aoi = self._aoi(tif, 400, 300, 1600, 1100)
        ing = self._ingester(monkeypatch)
        out = ing._download_naip(
            {"Name": "va_m_test", "assets": {"image": str(tif)}, "aoi": aoi},
            tmp_path / "raw")

        with rasterio.open(out / "image.tif") as src:
            assert src.count == 3                                # RGB, NIR dropped
            assert abs(src.res[0]) == pytest.approx(1.2, abs=0.01)   # 0.6 m → 1.2 m
            # (the lat/lon box of a UTM rectangle is a few pixels larger)
            assert src.width == pytest.approx(600, abs=15)
            assert src.height == pytest.approx(400, abs=15)
            # top-left corner still maps to the AOI's north-west corner
            x, y = src.transform * (0, 0)
            lon, lat = transform(src.crs, "EPSG:4326", [x], [y])
        assert lat[0] == pytest.approx(aoi[2], abs=1.5e-4)
        assert lon[0] == pytest.approx(aoi[1], abs=1.5e-4)

    def test_image_outside_aoi_is_skipped(self, tmp_path, monkeypatch):
        tif = _rgb_tif(tmp_path / "remote.tif")
        ing = self._ingester(monkeypatch)
        out = ing._download_naip(
            {"Name": "va_m_test", "assets": {"image": str(tif)},
             "aoi": (10.0, 10.0, 10.1, 10.1)}, tmp_path / "raw")
        assert out is None

    def test_query_keeps_only_newest_flight(self, monkeypatch):
        import data.ingestion as ing_mod

        def feature(fid, date):
            return {"id": fid, "properties": {"datetime": f"{date}T00:00:00Z"},
                    "assets": {"image": {"href": f"https://x/{fid}.tif"}}}

        class Resp:
            def raise_for_status(self): pass
            def json(self):
                return {"features": [feature("a_2023", "2023-10-09"),
                                     feature("b_2023", "2023-10-09"),
                                     feature("a_2021", "2021-09-13")]}

        sent = {}
        monkeypatch.setattr(ing_mod.requests, "post",
                            lambda url, json, timeout: sent.update(json) or Resp())
        products = ing_mod.PlanetaryComputerIngester().query_bbox(
            (36.9, -76.4, 37.0, -76.3), sensor="NAIP")

        assert sent["collections"] == ["naip"] and "datetime" not in sent
        assert [p["Name"] for p in products] == ["a_2023", "b_2023"]
        assert all(p["kind"] == "naip" for p in products)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-scale tiling
# ══════════════════════════════════════════════════════════════════════════════

class TestHighResTiling:

    def test_scales_have_matching_georeferencing(self, tmp_path):
        from data.preprocessing import Preprocessor
        from PIL import Image
        product = tmp_path / "va_m_test.NAIP"
        _rgb_tif(product / "image.tif", h=1024, w=1024, res=1.2)

        pre = Preprocessor(tile_size=512, stride=256, out_dir=tmp_path / "tiles")
        tiles = pre.process_product(product, sensor="NAIP")

        native = [t for t in tiles if not t.parent.name.endswith("_x2")
                  and not t.parent.name.endswith("_x3")]
        half = [t for t in tiles if t.parent.name.endswith("_x2")]
        assert len(native) == 9 and len(half) == 1

        m1 = json.loads(native[0].with_suffix(".json").read_text())
        m2 = json.loads(half[0].with_suffix(".json").read_text())
        assert m1["gsd_m"] == pytest.approx(1.2, abs=0.02)
        assert m2["gsd_m"] == pytest.approx(2.4, abs=0.04)
        # Both sets start at the same place; the coarse tile covers 2× the ground
        assert (m2["lat_tl"], m2["lon_tl"]) == pytest.approx(
            (m1["lat_tl"], m1["lon_tl"]), abs=1e-6)
        assert m2["lat_tl"] - m2["lat_br"] == pytest.approx(
            2 * (m1["lat_tl"] - m1["lat_br"]), rel=0.01)
        # Aerial imagery is not contrast-stretched
        assert np.asarray(Image.open(native[0])).shape == (512, 512, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Oriented-box detections
# ══════════════════════════════════════════════════════════════════════════════

def _obb_result(boxes):
    """Stand-in for an ultralytics OBB result. boxes: (cx, cy, w, h, conf, cls)."""
    corners, xyxy, xywhr, conf, cls = [], [], [], [], []
    for cx, cy, w, h, c, k in boxes:
        corners.append([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        xywhr.append([cx, cy, w, h, 0.0])
        conf.append(c)
        cls.append(k)

    def col(v):
        return SimpleNamespace(tolist=lambda: v)

    return SimpleNamespace(obb=SimpleNamespace(
        xyxyxyxy=col(corners), xyxy=col(xyxy), xywhr=col(xywhr),
        conf=col(conf), cls=col(cls)), boxes=None)


def _tile(tmp_path, name="t", lat=36.95, lon=-76.33, size=1024, gsd=1.2):
    """Blank tile with a north-up sidecar; returns its path."""
    from PIL import Image
    import math
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (size, size)).save(path)
    dlat = size * gsd / 111_320.0
    dlon = dlat / math.cos(math.radians(lat))
    path.with_suffix(".json").write_text(json.dumps({
        "tile_size": size, "gsd_m": gsd,
        "lat_tl": lat, "lon_tl": lon, "lat_tr": lat, "lon_tr": lon + dlon,
        "lat_bl": lat - dlat, "lon_bl": lon, "lat_br": lat - dlat, "lon_br": lon + dlon,
    }))
    return path


class TestOrientedDetections:

    def _detector(self):
        det = ShipDetector(weights=Path("unused.pt"), keep_classes={"ship"})
        det._model = SimpleNamespace(names={0: "plane", 1: "ship", 7: "harbor"})
        return det

    def test_parse_gives_true_hull_size_and_outline(self, tmp_path):
        tile = _tile(tmp_path)
        r = _obb_result([(500, 400, 130, 16, 0.9, 1),      # 156 × 19 m ship
                         (200, 200, 300, 40, 0.8, 7),      # a harbour → dropped
                         (700, 700, 50, 50, 0.7, 0)])      # a plane → dropped
        (d,) = self._detector()._parse_result(tile, r)

        assert d.class_name == "ship"
        assert (d.length_m, d.beam_m) == (156.0, 19.2)
        assert len(d.footprint) == 4
        assert d.lat == pytest.approx(36.95 - 400 * 1.2 / 111_320, abs=1e-5)

    def test_ships_moored_side_by_side_are_both_kept(self, tmp_path):
        """Two destroyers 30 m apart must not be merged as duplicates."""
        det = self._detector()
        a = det._parse_result(_tile(tmp_path, "a"),
                              _obb_result([(500, 400, 130, 16, 0.9, 1)]))
        b = det._parse_result(_tile(tmp_path, "b"),
                              _obb_result([(500, 425, 130, 16, 0.8, 1)]))
        assert len(ShipDetector.nms_across_tiles(a + b)) == 2

    def test_same_ship_seen_at_two_scales_is_merged(self, tmp_path):
        det = self._detector()
        fine = det._parse_result(_tile(tmp_path, "fine"),
                                 _obb_result([(500, 400, 130, 16, 0.6, 1)]))
        # the same hull in a 2.4 m/px tile: half the pixels, same ground footprint
        coarse = det._parse_result(_tile(tmp_path, "coarse", gsd=2.4),
                                   _obb_result([(250, 200, 66, 9, 0.9, 1)]))
        kept = ShipDetector.nms_across_tiles(fine + coarse)
        assert len(kept) == 1 and kept[0].confidence == 0.9

    def test_min_length_filter(self):
        boat = Detection(Path("t.png"), (0, 0, 10, 10), 0.9, 1, length_m=18.0)
        ship = Detection(Path("t.png"), (0, 0, 10, 10), 0.9, 1, length_m=150.0)
        assert ShipDetector.filter_by_size([boat, ship], min_length_m=30) == [ship]


# ══════════════════════════════════════════════════════════════════════════════
# Classification on sharp imagery
# ══════════════════════════════════════════════════════════════════════════════

class TestHighResClassification:

    def test_detector_dimensions_are_kept(self, tmp_path):
        """Hull size from the oriented box is not overwritten by crop segmentation."""
        from models.classifier import WarshipClassifier
        tile = _tile(tmp_path)
        d = Detection(tile, (400, 380, 600, 420), 0.9, 1, lat=36.95, lon=-76.33,
                      length_m=155.0, beam_m=20.0)
        WarshipClassifier(weights=tmp_path / "none.pt").classify_detections([d])
        assert (d.length_m, d.beam_m) == (155.0, 20.0)
        assert d.size_class == "Destroyer"

    def test_crowd_rule_leaves_cnn_undecided_ships_alone(self):
        """Warships moored together at a base must not become 'civilian'."""
        from geo.ais_fusion import label_crowded_waters
        pier = [Detection(Path("t.png"), (0, 0, 9, 9), 0.9, 1,
                          lat=36.95 + 0.0005 * i, lon=-76.33,
                          warship_type="Unidentified Vessel", type_source="cnn")
                for i in range(10)]
        label_crowded_waters(pier)
        assert {d.warship_type for d in pier} == {"Unidentified Vessel"}

    def test_pipeline_routes_naip_to_highres_components(self, tmp_path):
        """A local high-res product runs through the HR tiler, detector and CNN path."""
        from tests.test_components import _make_pipeline
        from data.preprocessing import Preprocessor
        pipe = _make_pipeline(tmp_path)
        pipe.preprocessor_hr = Preprocessor(tile_size=512, stride=256,
                                            out_dir=tmp_path / "hr_tiles")
        product = tmp_path / "va_m_test.NAIP"
        _rgb_tif(product / "image.tif", h=512, w=512, res=1.2)

        seen = {}

        def fake_run_batch(tile_paths, batch_size=8, quiet=False):
            seen["tiles"] = list(tile_paths)
            tile = next(t for t in tile_paths if "_x" not in t.parent.name)
            geo = ShipDetector._load_geo(tile)
            lat, lon = ShipDetector._pixel_to_geo(256, 256, geo)
            return [
                Detection(tile, (190, 248, 322, 264), 0.9, 1, lat=lat, lon=lon,
                          length_m=158.0, beam_m=19.0),
                Detection(tile, (10, 10, 20, 14), 0.9, 1, lat=lat + 0.002, lon=lon,
                          length_m=12.0, beam_m=4.0),          # a boat → dropped
            ]

        pipe.detector_hr.run_batch = fake_run_batch
        pipe.detector.run_batch = lambda *a, **k: pytest.fail("SAR detector used")

        (d,) = pipe.run_local(product, region="Test_Base", sensor="NAIP")
        assert any("_x2" in t.parent.name for t in seen["tiles"])   # multi-scale
        assert d.size_class == "Destroyer"
        assert d.track_id is not None
        assert pipe.db.summary()["detections"] == 1

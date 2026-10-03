"""
tests/test_real_data.py — Tests for the pieces that handle real data:
the credential-free Planetary Computer source, FGSC-23 preparation, and the
CNN classifier's training / evaluation / resolution gating.

Network calls are mocked; nothing here downloads anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import Detection


# ══════════════════════════════════════════════════════════════════════════════
# Planetary Computer ingestion
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanetaryComputer:

    @pytest.fixture
    def scene(self, tmp_path):
        """A fake remote Sentinel-1 scene: 2000×3000 px with GCPs, north-up."""
        rasterio = pytest.importorskip("rasterio")
        from rasterio.control import GroundControlPoint

        H, W, deg = 2000, 3000, 1e-4
        self.truth = lambda col, row: (104.0 + deg * col, 2.0 - deg * row)
        gcps = []
        for r in np.linspace(0, H, 5):
            for c in np.linspace(0, W, 5):
                lon, lat = self.truth(c, r)
                gcps.append(GroundControlPoint(row=float(r), col=float(c), x=lon, y=lat))

        paths = {}
        rng = np.random.default_rng(0)
        for pol in ("vv", "vh"):
            path = tmp_path / f"remote-{pol}.tiff"
            data = rng.integers(1, 400, (H, W)).astype(np.uint16)
            data[1000, 1500] = 9000 if pol == "vv" else 7000      # a marker pixel
            with rasterio.open(path, "w", driver="GTiff", height=H, width=W,
                               count=1, dtype="uint16") as dst:
                dst.write(data, 1)
                dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))
            paths[pol] = str(path)
        return paths

    def _ingester(self, monkeypatch, **kw):
        from data.ingestion import PlanetaryComputerIngester
        ing = PlanetaryComputerIngester(**kw)
        monkeypatch.setattr(ing, "_sign", lambda href: href)     # no network
        return ing

    def _product(self, scene, aoi):
        return {"Id": "S1X", "Name": "S1X_IW_GRDH_1SDV_20240115T060000_TEST",
                "ContentDate": {"Start": "2024-01-15T06:00:00Z"},
                "assets": scene, "aoi": aoi}

    def test_downloads_only_the_aoi_window(self, scene, tmp_path, monkeypatch):
        import rasterio
        ing = self._ingester(monkeypatch)
        # AOI = cols 1000–2000, rows 500–1500 of the scene
        aoi = (2.0 - 0.15, 104.10, 2.0 - 0.05, 104.20)
        (out,) = ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw")

        files = sorted((out / "measurement").glob("*.tiff"))
        assert [f.name.rsplit("-", 1)[1] for f in files] == ["vh.tiff", "vv.tiff"]
        with rasterio.open(files[1]) as src:
            assert abs(src.width - 1000) <= 2 and abs(src.height - 1000) <= 2
            data = src.read(1)
            gcps = src.gcps[0]
        # The marker pixel (scene col 1500, row 1000) lands mid-window
        r, c = np.unravel_index(np.argmax(data), data.shape)
        assert abs(r - 500) <= 1 and abs(c - 500) <= 1
        assert len(gcps) >= 9

    def test_window_keeps_correct_georeferencing(self, scene, tmp_path, monkeypatch):
        """Tiles cut from the window must carry the scene's true lat/lon."""
        from data.preprocessing import Preprocessor
        ing = self._ingester(monkeypatch)
        aoi = (2.0 - 0.15, 104.10, 2.0 - 0.05, 104.20)
        (out,) = ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw")

        tiles = Preprocessor(out_dir=tmp_path / "tiles").process_product(out, "S1")
        meta = json.loads(tiles[0].with_suffix(".json").read_text())
        assert meta["lon_tl"] == pytest.approx(104.10, abs=3e-4)
        assert meta["lat_tl"] == pytest.approx(1.95, abs=3e-4)
        assert meta["gsd_m"] == pytest.approx(11.1, abs=0.3)

    def test_window_is_capped(self, scene, tmp_path, monkeypatch):
        import rasterio
        ing = self._ingester(monkeypatch, max_window_px=512)
        aoi = (1.0, 103.0, 3.0, 105.0)             # bigger than the whole scene
        (out,) = ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw")
        with rasterio.open(next((out / "measurement").glob("*-vv.tiff"))) as src:
            assert (src.width, src.height) == (512, 512)

    def test_scene_outside_aoi_is_skipped(self, scene, tmp_path, monkeypatch):
        ing = self._ingester(monkeypatch)
        aoi = (50.0, 10.0, 50.2, 10.2)
        assert ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw") == []
        assert not list((tmp_path / "raw").glob("*.part"))

    def test_second_download_is_cached(self, scene, tmp_path, monkeypatch):
        ing = self._ingester(monkeypatch)
        aoi = (1.85, 104.10, 1.95, 104.20)
        first = ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw")
        monkeypatch.setattr(ing, "_sign", lambda href: 1 / 0)     # must not be called
        assert ing.download([self._product(scene, aoi)], output_dir=tmp_path / "raw") == first

    def test_query_parses_stac_response(self, monkeypatch):
        import data.ingestion as ing_mod

        class Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"features": [{
                    "id": "S1A_IW_GRDH_1SDV_20240115T060000",
                    "properties": {"datetime": "2024-01-15T06:00:12Z"},
                    "assets": {"vv": {"href": "https://x/vv.tiff"},
                               "vh": {"href": "https://x/vh.tiff"},
                               "thumbnail": {"href": "https://x/t.png"}},
                }]}

        sent = {}
        monkeypatch.setattr(ing_mod.requests, "post",
                            lambda url, json, timeout: sent.update(json) or Resp())
        (p,) = ing_mod.PlanetaryComputerIngester().query_region("Red_Sea", days_back=3)

        assert sent["collections"] == ["sentinel-1-grd"]
        assert sent["bbox"] == [32.0, 12.0, 43.5, 30.0]           # lon/lat order
        assert set(p["assets"]) == {"vv", "vh"}
        assert p["ContentDate"]["Start"] == "2024-01-15T06:00:12Z"
        assert p["aoi"] == (12.0, 32.0, 30.0, 43.5)

    def test_sentinel2_not_supported(self):
        from data.ingestion import PlanetaryComputerIngester
        with pytest.raises(ValueError, match="Sentinel-1 only"):
            PlanetaryComputerIngester().query_bbox((0, 0, 1, 1), sensor="S2")

    def test_auto_source_without_credentials(self, monkeypatch):
        import data.ingestion as ing_mod
        monkeypatch.setattr(ing_mod, "COPERNICUS_USER", "your_username")
        assert isinstance(ing_mod.make_ingester("auto"),
                          ing_mod.PlanetaryComputerIngester)
        monkeypatch.setattr(ing_mod, "COPERNICUS_USER", "someone")
        monkeypatch.setattr(ing_mod, "COPERNICUS_PASSWORD", "secret")
        assert isinstance(ing_mod.make_ingester("auto"), ing_mod.SentinelIngester)


# ══════════════════════════════════════════════════════════════════════════════
# FGSC-23 preparation and CNN classifier
# ══════════════════════════════════════════════════════════════════════════════

def _fake_fgsc(root: Path, ids=(0, 1, 2, 12, 17), n=4):
    from PIL import Image
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for ship_id in ids:
            d = root / "FGSC-23" / split / str(ship_id)
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{ship_id}_{i}.jpg")
    return root


class TestFGSC:

    def test_prepare_maps_ids_to_classes(self, tmp_path):
        from data.dataset import DatasetManager
        out = DatasetManager(tmp_path / "ds").prepare_fgsc23(
            _fake_fgsc(tmp_path / "raw"), tmp_path / "clf")

        for split in ("train", "test"):
            folders = {p.name for p in (out / split).iterdir()}
            # 0 = non-ship is dropped; "Patrol / Corvette" becomes a legal folder name
            assert folders == {"Aircraft_Carrier", "Destroyer", "Patrol_Corvette",
                               "Civilian_Vessel"}
            assert len(list((out / split / "Destroyer").iterdir())) == 4

    def test_folders_load_with_right_labels(self, tmp_path):
        pytest.importorskip("torch")
        from config import CLASSIFIER_CLASSES
        from data.dataset import DatasetManager
        from models.classifier import WarshipDataset
        out = DatasetManager(tmp_path / "ds").prepare_fgsc23(
            _fake_fgsc(tmp_path / "raw"), tmp_path / "clf")

        ds = WarshipDataset(out / "train")
        names = {CLASSIFIER_CLASSES[label] for _, label in ds.samples}
        assert names == {"Aircraft Carrier", "Destroyer", "Patrol / Corvette",
                         "Civilian Vessel"}


class TestCNNClassifier:

    @pytest.fixture
    def trained(self, tmp_path):
        pytest.importorskip("torch")
        from data.dataset import DatasetManager
        from models.classifier import train_classifier
        data = DatasetManager(tmp_path / "ds").prepare_fgsc23(
            _fake_fgsc(tmp_path / "raw"), tmp_path / "clf")
        weights = train_classifier(data / "train", val_dir=data / "test", epochs=1,
                                   batch_size=4, pretrained=False,
                                   output=tmp_path / "clf.pt")
        return data, weights

    def test_checkpoint_includes_civilian_class(self, trained):
        from config import CIVILIAN_LABEL
        from models.classifier import WarshipClassifier
        _, weights = trained
        clf = WarshipClassifier(weights=weights)
        assert clf.method == "cnn"
        assert len(clf.classes) == 9 and CIVILIAN_LABEL in clf.classes

    def test_evaluate_reports_per_class(self, trained):
        from models.classifier import evaluate_classifier
        data, weights = trained
        r = evaluate_classifier(data / "test", weights=weights)
        assert 0.0 <= r["accuracy"] <= 1.0
        assert set(r["per_class"]) == {"Aircraft Carrier", "Destroyer",
                                       "Patrol / Corvette", "Civilian Vessel"}
        assert sum(sum(row) for row in r["confusion"]) == 16

    def test_cnn_only_used_on_sharp_imagery(self, trained, tmp_path):
        """At Sentinel's 10 m pixels the CNN must be bypassed for the size method."""
        from tests.test_components import _ship_tile
        from models.classifier import WarshipClassifier
        _, weights = trained
        clf = WarshipClassifier(weights=weights, conf_min=0.0)
        assert clf.method == "cnn"          # loads the (lazy) model
        calls = []
        real = clf._model.forward
        clf._model.forward = lambda x: calls.append(1) or real(x)

        coarse = _ship_tile(tmp_path / "t.png", 155, 20)        # sidecar says 10 m/px
        clf.classify_detections([coarse])
        assert calls == []
        assert coarse.size_class == "Destroyer"                 # from hull size
        assert coarse.warship_type == "Unidentified Vessel"

        sharp = _ship_tile(tmp_path / "t.png", 155, 20)
        clf.classify_detections([sharp], gsd_m=1.0)
        assert calls == [1]
        assert sharp.warship_type in clf.classes

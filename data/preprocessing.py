"""
data/preprocessing.py — Satellite image preprocessing pipeline.

Steps:
  1. Read GeoTIFF bands from a Sentinel product folder.
  2. Apply cloud masking (Sentinel-2 SCL band).
  3. Log-stretch SAR amplitude to 8-bit.
  4. Tile large images into TILE_SIZE_PX × TILE_SIZE_PX patches.
  5. Save tiles as PNG with geo-metadata sidecar JSON.

Usage:
    from data.preprocessing import Preprocessor
    pre = Preprocessor()
    tiles = pre.process_product(Path("data/raw/S1_product.SAFE"))
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

# rasterio / GDAL for geospatial raster handling
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
    import rasterio.mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARNING] rasterio not found — install it with: pip install rasterio")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] Pillow not found — install it with: pip install Pillow")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TILE_SIZE_PX, TILE_STRIDE, TILES_DIR
from utils.geo_utils import haversine_km
from utils.logger import get_logger

log = get_logger("preprocessing")

# (cols, rows) pixel arrays → (lons, lats) in EPSG:4326
GeoFn = Callable[[List[float], List[float]], Tuple[List[float], List[float]]]

# Percentile stretches are estimated on at most this many pixels
_MAX_STAT_PIXELS = 4_000_000


def _subsample(arr: np.ndarray) -> np.ndarray:
    """Thin out a large raster so percentile estimation stays fast."""
    step = int(np.ceil(np.sqrt(arr.size / _MAX_STAT_PIXELS))) if arr.size else 1
    return arr[::step, ::step] if step > 1 else arr


def make_geo_fn(src) -> Optional[GeoFn]:
    """
    Build a pixel → lon/lat function for an open rasterio dataset.

    Handles the three cases we meet in practice:
      • affine transform in a geographic CRS (plain GeoTIFF),
      • affine transform in a projected CRS (Sentinel-2 is UTM),
      • no transform but ground control points (Sentinel-1 GRD).
    Returns None for un-georeferenced rasters.
    """
    crs, transform = src.crs, src.transform

    if crs is not None and not transform.is_identity:
        geographic = crs.is_geographic

        def affine_fn(cols, rows):
            xs, ys = [], []
            for c, r in zip(cols, rows):
                x, y = transform * (c, r)
                xs.append(x)
                ys.append(y)
            if not geographic:
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform(crs, "EPSG:4326", xs, ys)
            return list(xs), list(ys)
        return affine_fn

    gcps, gcp_crs = src.gcps
    if gcps:
        from rasterio.transform import GCPTransformer

        def gcp_fn(cols, rows):
            with GCPTransformer(gcps) as tf:
                xs, ys = tf.xy(list(rows), list(cols), offset="ul")
            xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
            if gcp_crs is not None and not gcp_crs.is_geographic:
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform(gcp_crs, "EPSG:4326", list(xs), list(ys))
            return [float(x) for x in xs], [float(y) for y in ys]
        return gcp_fn

    return None


# ── Band Definitions ───────────────────────────────────────────────────────────

# Sentinel-2 bands used for RGB composite (B04=Red, B03=Green, B02=Blue)
S2_RGB_BANDS = ["B04", "B03", "B02"]

# Sentinel-1 polarisations
S1_POLS = ["VV", "VH"]

# Sentinel-2 SCL cloud classes to mask
S2_CLOUD_SCL_VALUES = {8, 9, 10}  # medium/high probability cloud + cirrus


class Preprocessor:
    """
    Converts raw Sentinel .SAFE products into 640×640 PNG tiles
    ready for YOLOv8 inference.
    """

    def __init__(self, tile_size: int = TILE_SIZE_PX, stride: int = TILE_STRIDE,
                 out_dir: Path = TILES_DIR):
        self.tile_size = tile_size
        self.stride    = stride
        self.out_dir   = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_product(self, product_dir: Path, sensor: str = "auto") -> List[Path]:
        """
        Detect sensor type, read bands, tile, and return tile paths.

        Args:
            product_dir: Path to the unpacked .SAFE directory.
            sensor:      "S1", "S2", or "auto" (detected from dir name).

        Returns:
            List of paths to tile PNG files.
        """
        product_dir = Path(product_dir)
        if sensor == "auto":
            sensor = "S1" if "S1" in product_dir.name else "S2"

        log.info("Processing %s product: %s", sensor, product_dir.name)

        if product_dir.is_file():
            return self.process_geotiff(product_dir, sensor=sensor)

        if sensor == "S1":
            image_array, geo_fn = self._read_sar(product_dir)
        else:
            image_array, geo_fn = self._read_optical(product_dir)

        if image_array is None:
            log.error("Could not read image data from %s", product_dir)
            return []
        if geo_fn is None:
            log.warning("%s has no georeferencing — detections will have no "
                        "lat/lon.", product_dir.name)

        tiles = self._tile_image(image_array, None, product_dir.name,
                                 geo_fn=geo_fn, skip_empty=True)
        log.info("Generated %d tiles from %s", len(tiles), product_dir.name)
        return tiles

    def process_geotiff(self, tif_path: Path, sensor: str = "S1") -> List[Path]:
        """Process a single GeoTIFF directly (useful for testing)."""
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for GeoTIFF processing")

        tif_path = Path(tif_path)
        with rasterio.open(tif_path) as src:
            if sensor == "S1" or src.count < 3:
                arr = self._sar_to_uint8(src.read(1).astype(np.float32))
                arr = np.stack([arr, arr, arr], axis=-1)
            else:
                r = self._normalize_band(src.read(1).astype(np.float32))
                g = self._normalize_band(src.read(2).astype(np.float32))
                b = self._normalize_band(src.read(3).astype(np.float32))
                arr = np.stack([r, g, b], axis=-1)
            geo_fn = make_geo_fn(src)

        return self._tile_image(arr, None, tif_path.stem, geo_fn=geo_fn,
                                skip_empty=True)

    # ── Sentinel-1 SAR Reader ─────────────────────────────────────────────────

    def _read_sar(
        self, product_dir: Path
    ) -> Tuple[Optional[np.ndarray], Optional[GeoFn]]:
        """
        Read a Sentinel-1 GRD product and log-stretch it to an 8-bit greyscale
        image (replicated to 3 channels).

        The cross-polarised VH band is used when present — the sea surface is
        much darker in VH, so ships stand out better — otherwise VV. Tiles are
        deliberately single-band greyscale: that is what the SAR-Ship training
        chips look like, and a detector fine-tuned on them finds almost
        nothing in a false-colour VV/VH composite.
        """
        if not HAS_RASTERIO:
            log.error("rasterio not installed — cannot read SAR data.")
            return None, None

        def find(pol: str) -> Optional[Path]:
            # Locate measurement TIFFs inside the .SAFE directory
            meas_dir = product_dir / "measurement"
            root = meas_dir if meas_dir.exists() else product_dir
            for pattern in (f"*-{pol}-*.tif*", f"*-{pol}.tif*", f"*{pol}*.tif*"):
                hits = sorted(root.rglob(pattern))
                if hits:
                    return hits[0]
            return None

        path, name = find("vh"), "VH"
        if path is None:
            path, name = find("vv"), "VV"
        if path is None:
            log.error("No VV/VH band TIFF found in %s", product_dir)
            return None, None

        with rasterio.open(path) as src:
            # GRD measurements are digital-number amplitudes, not dB
            raw = src.read(1).astype(np.float32)
            geo_fn = make_geo_fn(src)
        gray = self._sar_to_uint8(raw)
        del raw
        log.info("Read %s band: %s (%d×%d)", name, path.name,
                 gray.shape[0], gray.shape[1])

        return np.stack([gray, gray, gray], axis=-1), geo_fn

    # ── Sentinel-2 Optical Reader ──────────────────────────────────────────────

    def _read_optical(
        self, product_dir: Path
    ) -> Tuple[Optional[np.ndarray], Optional[GeoFn]]:
        """
        Read Sentinel-2 L2A RGB bands (10m resolution) and apply SCL cloud mask.
        Returns H×W×3 uint8 array.
        """
        if not HAS_RASTERIO:
            log.error("rasterio not installed — cannot read optical data.")
            return None, None

        granule_dirs = [d for d in (product_dir / "GRANULE").glob("*") if d.is_dir()]
        if not granule_dirs:
            log.error("No GRANULE subdir found in %s", product_dir)
            return None, None

        granule = granule_dirs[0]
        r10_dir = granule / "IMG_DATA" / "R10m"

        channels = []
        geo_fn = None

        for band_name in S2_RGB_BANDS:
            band_files = list(r10_dir.glob(f"*_{band_name}_10m.jp2"))
            if not band_files:
                band_files = list(r10_dir.glob(f"*{band_name}*.jp2"))
            if not band_files:
                log.warning("Band %s not found in %s", band_name, r10_dir)
                continue
            with rasterio.open(band_files[0]) as src:
                arr = src.read(1).astype(np.float32)
                if geo_fn is None:
                    geo_fn = make_geo_fn(src)
                channels.append(self._normalize_band(arr))

        if len(channels) != 3:
            log.error("Could not read all 3 RGB bands. Got %d.", len(channels))
            return None, None

        rgb = np.stack(channels, axis=-1)

        # Optionally apply cloud mask from SCL band
        scl_files = list((granule / "IMG_DATA" / "R20m").glob("*SCL*20m.jp2"))
        if scl_files:
            rgb = self._apply_cloud_mask(rgb, scl_files[0], target_shape=rgb.shape[:2])

        return rgb, geo_fn

    # ── Tiling ─────────────────────────────────────────────────────────────────

    def _tile_image(
        self,
        image: np.ndarray,
        transform: Optional[object],
        product_name: str,
        geo_fn: Optional[GeoFn] = None,
        skip_empty: bool = False,
    ) -> List[Path]:
        """
        Slice a large H×W×3 array into TILE_SIZE×TILE_SIZE patches with overlap.
        Each tile is saved as PNG. A sidecar JSON stores the lat/lon of all
        four tile corners and the ground sample distance.

        Args:
            transform:  Affine transform whose units are already lon/lat
                        degrees (ignored when geo_fn is given).
            geo_fn:     Pixel → lon/lat function from make_geo_fn().
            skip_empty: Drop tiles that contain no data at all (scene borders).
        """
        if not HAS_PIL:
            log.error("Pillow not installed — cannot save tiles.")
            return []

        if geo_fn is None and transform is not None:
            def geo_fn(cols, rows, _t=transform):
                pts = [_t * (c, r) for c, r in zip(cols, rows)]
                return [p[0] for p in pts], [p[1] for p in pts]

        ts = self.tile_size
        H, W, C = image.shape

        # Pad scenes smaller than one tile so they still produce output
        if H < ts or W < ts:
            padded = np.zeros((max(H, ts), max(W, ts), C), dtype=image.dtype)
            padded[:H, :W] = image
            image = padded
            H, W = image.shape[:2]

        tile_dir = self.out_dir / product_name
        tile_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        row_starts = list(range(0, H - ts + 1, self.stride))
        col_starts = list(range(0, W - ts + 1, self.stride))

        # Ensure we always include the last row/col
        if row_starts[-1] + ts < H:
            row_starts.append(H - ts)
        if col_starts[-1] + ts < W:
            col_starts.append(W - ts)

        n_skipped = 0
        for r in row_starts:
            for c in col_starts:
                patch = image[r:r + ts, c:c + ts]
                if skip_empty and not patch.any():
                    n_skipped += 1
                    continue

                tile_name = f"tile_{r:05d}_{c:05d}"
                png_path  = tile_dir / f"{tile_name}.png"

                img = Image.fromarray(np.ascontiguousarray(patch, dtype=np.uint8))
                img.save(png_path)

                # Geo-metadata sidecar
                meta: Dict = {"row": r, "col": c, "product": product_name,
                              "tile_size": ts}
                if geo_fn is not None:
                    # Corner pixels → geographic coordinates (TL, TR, BL, BR)
                    lons, lats = geo_fn([c, c + ts, c, c + ts],
                                        [r, r, r + ts, r + ts])
                    for key, lon, lat in zip(("tl", "tr", "bl", "br"), lons, lats):
                        meta[f"lat_{key}"] = round(float(lat), 6)
                        meta[f"lon_{key}"] = round(float(lon), 6)
                    width_km = haversine_km((lats[0], lons[0]), (lats[1], lons[1]))
                    meta["gsd_m"] = round(width_km * 1000.0 / ts, 3)

                with open(png_path.with_suffix(".json"), "w") as f:
                    json.dump(meta, f)

                paths.append(png_path)

        if n_skipped:
            log.info("Skipped %d empty (no-data) tile(s).", n_skipped)
        return paths

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_band(arr: np.ndarray, low_pct: float = 2.0,
                        high_pct: float = 98.0) -> np.ndarray:
        """Stretch percentile range to 0–255 uint8."""
        lo, hi = np.percentile(_subsample(arr), [low_pct, high_pct])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        return (arr * 255).astype(np.uint8)

    @staticmethod
    def _sar_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Logarithmic stretch of SAR amplitude / power values to uint8."""
        arr = np.log1p(np.maximum(arr, 0))
        sample = _subsample(arr)
        valid = sample[sample > 0]
        if valid.size:
            lo, med, p999 = np.percentile(valid, [1, 50, 99.9])
            # Ships are a tiny, very bright fraction of the scene. Stretching
            # only to a high percentile would spend the whole 8-bit range on
            # sea speckle and saturate ships together with its brightest
            # pixels, so keep ~26 dB of headroom above the median.
            hi = max(p999, med + 3.0)
        else:
            lo, hi = 0, 1
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        return (arr * 255).astype(np.uint8)

    def _apply_cloud_mask(
        self,
        rgb: np.ndarray,
        scl_path: Path,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Zero-out cloud pixels using Sentinel-2 SCL classification band."""
        if not HAS_RASTERIO:
            return rgb
        with rasterio.open(scl_path) as src:
            scl = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.nearest,
            )
        cloud_mask = np.isin(scl, list(S2_CLOUD_SCL_VALUES))
        rgb[cloud_mask] = 0
        cloud_pct = 100.0 * cloud_mask.mean()
        log.info("Cloud mask applied — %.1f%% of pixels masked.", cloud_pct)
        return rgb


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a Sentinel .SAFE product")
    parser.add_argument("product_dir", type=Path, help="Path to unpacked .SAFE directory")
    parser.add_argument("--sensor", default="auto", choices=["auto", "S1", "S2"])
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_PX)
    args = parser.parse_args()

    pre = Preprocessor(tile_size=args.tile_size)
    tiles = pre.process_product(args.product_dir, sensor=args.sensor)
    print(f"\n✔ Generated {len(tiles)} tiles → {TILES_DIR}")

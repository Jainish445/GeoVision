"""
data/preprocessing.py — Satellite image preprocessing pipeline.

Steps:
  1. Read GeoTIFF bands from a Sentinel product folder.
  2. Apply cloud masking (Sentinel-2 SCL band).
  3. Convert SAR backscatter from dB to linear scale.
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
from typing import List, Optional, Tuple, Dict

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
from utils.logger import get_logger

log = get_logger("preprocessing")


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

        if sensor == "S1":
            image_array, transform, crs = self._read_sar(product_dir)
        else:
            image_array, transform, crs = self._read_optical(product_dir)

        if image_array is None:
            log.error("Could not read image data from %s", product_dir)
            return []

        tiles = self._tile_image(image_array, transform, product_dir.name)
        log.info("Generated %d tiles from %s", len(tiles), product_dir.name)
        return tiles

    def process_geotiff(self, tif_path: Path, sensor: str = "S1") -> List[Path]:
        """Process a single GeoTIFF directly (useful for testing)."""
        if not HAS_RASTERIO:
            raise ImportError("rasterio required for GeoTIFF processing")

        with rasterio.open(tif_path) as src:
            if sensor == "S1":
                arr = self._sar_to_uint8(src.read(1))
                arr = np.stack([arr, arr, arr], axis=-1)
            else:
                r = self._normalize_band(src.read(1))
                g = self._normalize_band(src.read(2))
                b = self._normalize_band(src.read(3))
                arr = np.stack([r, g, b], axis=-1)
            transform = src.transform

        return self._tile_image(arr, transform, tif_path.stem)

    # ── Sentinel-1 SAR Reader ─────────────────────────────────────────────────

    def _read_sar(
        self, product_dir: Path
    ) -> Tuple[Optional[np.ndarray], Optional[object], Optional[object]]:
        """
        Read Sentinel-1 GRD VV+VH bands, apply dB→linear conversion,
        and produce an RGB-like array (VV, VH, VV/VH ratio).
        """
        if not HAS_RASTERIO:
            log.error("rasterio not installed — cannot read SAR data.")
            return None, None, None

        # Locate measurement TIFFs inside the .SAFE directory
        meas_dir = product_dir / "measurement"
        if not meas_dir.exists():
            # Try GRD folder structure
            candidates = list(product_dir.rglob("*vv*.tiff")) + \
                         list(product_dir.rglob("*vv*.tif"))
            if not candidates:
                log.error("No VV band TIFF found in %s", product_dir)
                return None, None, None
            vv_path = candidates[0]
        else:
            vv_paths = list(meas_dir.glob("*vv*"))
            vh_paths = list(meas_dir.glob("*vh*"))
            vv_path  = vv_paths[0] if vv_paths else None
            vh_path  = vh_paths[0] if vh_paths else None

        channels = []
        transform, crs = None, None

        for path, name in [(vv_path, "VV"), (getattr(locals(), "vh_path", None), "VH")]:
            if path and Path(path).exists():
                with rasterio.open(path) as src:
                    raw = src.read(1).astype(np.float32)
                    transform = src.transform
                    crs = src.crs
                    linear = np.power(10.0, raw / 10.0)   # dB → linear power
                    uint8  = self._sar_to_uint8(linear)
                    channels.append(uint8)
                    log.info("Read %s band: %s (%d×%d)", name, path.name,
                             uint8.shape[0], uint8.shape[1])

        if len(channels) == 0:
            return None, None, None

        if len(channels) == 1:
            channels = channels * 3   # replicate single polarisation

        if len(channels) == 2:
            # Third channel: VV/VH ratio (useful for distinguishing ship from sea)
            ratio = np.clip(
                channels[0].astype(np.float32) / (channels[1].astype(np.float32) + 1e-6),
                0, 255
            ).astype(np.uint8)
            channels.append(ratio)

        return np.stack(channels, axis=-1), transform, crs

    # ── Sentinel-2 Optical Reader ──────────────────────────────────────────────

    def _read_optical(
        self, product_dir: Path
    ) -> Tuple[Optional[np.ndarray], Optional[object], Optional[object]]:
        """
        Read Sentinel-2 L2A RGB bands (10m resolution) and apply SCL cloud mask.
        Returns H×W×3 uint8 array.
        """
        if not HAS_RASTERIO:
            log.error("rasterio not installed — cannot read optical data.")
            return None, None, None

        granule_dirs = list((product_dir / "GRANULE").glob("*"))
        if not granule_dirs:
            log.error("No GRANULE subdir found in %s", product_dir)
            return None, None, None

        granule = granule_dirs[0]
        r10_dir = granule / "IMG_DATA" / "R10m"

        channels = []
        transform, crs = None, None

        for band_name in S2_RGB_BANDS:
            band_files = list(r10_dir.glob(f"*_{band_name}_10m.jp2"))
            if not band_files:
                band_files = list(r10_dir.glob(f"*{band_name}*.jp2"))
            if not band_files:
                log.warning("Band %s not found in %s", band_name, r10_dir)
                continue
            with rasterio.open(band_files[0]) as src:
                arr = src.read(1).astype(np.float32)
                if transform is None:
                    transform = src.transform
                    crs = src.crs
                channels.append(self._normalize_band(arr))

        if len(channels) != 3:
            log.error("Could not read all 3 RGB bands. Got %d.", len(channels))
            return None, None, None

        rgb = np.stack(channels, axis=-1)

        # Optionally apply cloud mask from SCL band
        scl_files = list((granule / "IMG_DATA" / "R20m").glob("*SCL*20m.jp2"))
        if scl_files:
            rgb = self._apply_cloud_mask(rgb, scl_files[0], target_shape=rgb.shape[:2])

        return rgb, transform, crs

    # ── Tiling ─────────────────────────────────────────────────────────────────

    def _tile_image(
        self,
        image: np.ndarray,
        transform: Optional[object],
        product_name: str,
    ) -> List[Path]:
        """
        Slice a large H×W×3 array into TILE_SIZE×TILE_SIZE patches with overlap.
        Each tile is saved as PNG. A sidecar JSON stores geographic coordinates.
        """
        if not HAS_PIL:
            log.error("Pillow not installed — cannot save tiles.")
            return []

        H, W, C = image.shape
        tile_dir = self.out_dir / product_name
        tile_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        row_starts = list(range(0, H - self.tile_size + 1, self.stride))
        col_starts = list(range(0, W - self.tile_size + 1, self.stride))

        # Ensure we always include the last row/col
        if H > self.tile_size and row_starts[-1] + self.tile_size < H:
            row_starts.append(H - self.tile_size)
        if W > self.tile_size and col_starts[-1] + self.tile_size < W:
            col_starts.append(W - self.tile_size)

        for r in row_starts:
            for c in col_starts:
                patch = image[r:r + self.tile_size, c:c + self.tile_size]
                tile_name = f"tile_{r:05d}_{c:05d}"
                png_path  = tile_dir / f"{tile_name}.png"

                img = Image.fromarray(patch.astype(np.uint8), mode="RGB")
                img.save(png_path)

                # Geo-metadata sidecar
                meta: Dict = {"row": r, "col": c, "product": product_name}
                if transform is not None:
                    # Top-left corner pixel → geographic coordinate
                    lon, lat = transform * (c, r)
                    meta["lat_tl"] = round(float(lat), 6)
                    meta["lon_tl"] = round(float(lon), 6)
                    # Bottom-right corner
                    lon2, lat2 = transform * (c + self.tile_size, r + self.tile_size)
                    meta["lat_br"] = round(float(lat2), 6)
                    meta["lon_br"] = round(float(lon2), 6)

                with open(png_path.with_suffix(".json"), "w") as f:
                    json.dump(meta, f)

                paths.append(png_path)

        return paths

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_band(arr: np.ndarray, low_pct: float = 2.0,
                        high_pct: float = 98.0) -> np.ndarray:
        """Stretch percentile range to 0–255 uint8."""
        lo = np.percentile(arr, low_pct)
        hi = np.percentile(arr, high_pct)
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        return (arr * 255).astype(np.uint8)

    @staticmethod
    def _sar_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Logarithmic stretch of SAR linear power values to uint8."""
        arr = np.log1p(arr)
        lo, hi = np.percentile(arr[arr > 0], [1, 99]) if np.any(arr > 0) else (0, 1)
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

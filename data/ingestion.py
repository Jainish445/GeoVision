"""
data/ingestion.py — Satellite imagery ingestion.

Two sources of the same Sentinel imagery are supported:

  • SentinelIngester            — Copernicus Data Space (CDSE). Downloads whole
                                  products; needs a free Copernicus account.
      – Sentinel-1 GRD (SAR, cloud-independent, best for open ocean)
      – Sentinel-2 L2A (optical, cloud-screened)
  • PlanetaryComputerIngester   — Microsoft Planetary Computer. No account
                                  needed; fetches only the window of the
                                  Sentinel-1 GRD scene that covers the area
                                  of interest.

make_ingester() picks one based on config.DATA_SOURCE.

Usage:
    from data.ingestion import SentinelIngester
    ingester = SentinelIngester()
    products = ingester.query_region("South_China_Sea", days_back=5, sensor="S1")
    ingester.download(products, max_items=2)
"""

import os
import sys
import json
import time
import requests
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    COPERNICUS_USER, COPERNICUS_PASSWORD,
    CDSE_BASE_URL, CDSE_DOWNLOAD_URL,
    DATA_SOURCE, PC_STAC_URL, PC_SIGN_URL, PC_MAX_WINDOW_PX,
    DATA_DIR, REGIONS_OF_INTEREST,
    SAR_PRODUCT_TYPE, SAR_MAX_RESULTS,
    OPT_MAX_CLOUD_COVER, OPT_MAX_RESULTS,
)
from utils.logger import get_logger

log = get_logger("ingestion")


class SentinelIngester:
    """
    Handles authentication, querying, and downloading of Sentinel products
    from the Copernicus Data Space Ecosystem (CDSE).
    """

    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ── Authentication ─────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Fetch or refresh an OAuth2 bearer token."""
        if self._access_token and time.time() < self._token_expiry - 30:
            return self._access_token

        if (not COPERNICUS_USER or not COPERNICUS_PASSWORD
                or COPERNICUS_USER == "your_username"):
            raise RuntimeError(
                "Copernicus credentials are not set. Register for free at "
                "https://dataspace.copernicus.eu and set the COPERNICUS_USER "
                "and COPERNICUS_PASSWORD environment variables."
            )

        log.info("Fetching new Copernicus access token …")
        resp = requests.post(
            self.TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": COPERNICUS_USER,
                "password": COPERNICUS_PASSWORD,
            },
            timeout=30,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                "Copernicus rejected the credentials in COPERNICUS_USER / "
                "COPERNICUS_PASSWORD (HTTP 401)."
            )
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 600)
        log.info("Token obtained, expires in %ds", payload.get("expires_in", 0))
        return self._access_token

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}

    # ── Query ──────────────────────────────────────────────────────────────────

    def query_region(
        self,
        region_name: str,
        days_back: int = 7,
        sensor: str = "S1",          # "S1" = SAR, "S2" = optical
        date_start: Optional[str] = None,
        date_end:   Optional[str]  = None,
    ) -> List[Dict]:
        """
        Query CDSE for products over a named ROI.

        Args:
            region_name: Key from REGIONS_OF_INTEREST dict.
            days_back:   How many days of history to search.
            sensor:      "S1" for Sentinel-1 SAR, "S2" for Sentinel-2 optical.
            date_start:  ISO format override, e.g. "2024-01-01".
            date_end:    ISO format override.

        Returns:
            List of product metadata dicts.
        """
        if region_name not in REGIONS_OF_INTEREST:
            raise ValueError(f"Unknown region '{region_name}'. "
                             f"Choose from: {list(REGIONS_OF_INTEREST.keys())}")

        lat_min, lon_min, lat_max, lon_max = REGIONS_OF_INTEREST[region_name]
        bbox_wkt = (
            f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
        )

        now = datetime.now(timezone.utc)
        t_end   = date_end   or now.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_start = date_start or (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        if sensor == "S1":
            collection = "SENTINEL-1"
            extra_filters = (
                f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
                f" and att/OData.CSC.StringAttribute/Value eq '{SAR_PRODUCT_TYPE}')"
            )
            max_results = SAR_MAX_RESULTS
        else:
            collection = "SENTINEL-2"
            extra_filters = (
                f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'"
                f" and att/OData.CSC.DoubleAttribute/Value le {OPT_MAX_CLOUD_COVER})"
                f" and contains(Name,'L2A')"
            )
            max_results = OPT_MAX_RESULTS

        params = {
            "$filter": (
                f"Collection/Name eq '{collection}'"
                f" and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
                f" and ContentDate/Start gt {t_start}"
                f" and ContentDate/Start lt {t_end}"
                f"{extra_filters}"
            ),
            "$orderby": "ContentDate/Start desc",
            "$top": max_results,
        }

        log.info("[%s] Querying %s products for region '%s' (%s → %s) …",
                 sensor, collection, region_name, t_start, t_end)
        # The catalogue is public — only downloads need a token
        resp = requests.get(f"{CDSE_BASE_URL}/Products", params=params, timeout=60)
        resp.raise_for_status()
        products = resp.json().get("value", [])
        if sensor == "S1":
            # Cloud-optimised duplicates carry the same scene; keep one copy
            products = [p for p in products if "_COG" not in p.get("Name", "")] or products
        log.info("Found %d product(s).", len(products))
        return products

    def query_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        days_back: int = 5,
        sensor: str = "S1",
    ) -> List[Dict]:
        """Query using a raw (lat_min, lon_min, lat_max, lon_max) bounding box."""
        roi_key = "_custom_bbox_"
        REGIONS_OF_INTEREST[roi_key] = bbox
        try:
            return self.query_region(roi_key, days_back=days_back, sensor=sensor)
        finally:
            del REGIONS_OF_INTEREST[roi_key]

    # ── Download ───────────────────────────────────────────────────────────────

    def download(
        self,
        products: List[Dict],
        output_dir: Path = DATA_DIR,
        max_items: int = 2,
        unzip: bool = True,
    ) -> List[Path]:
        """
        Download product ZIPs from CDSE and optionally unzip them.

        Returns:
            List of paths to downloaded (or extracted) directories.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []

        for product in products[:max_items]:
            pid   = product["Id"]
            pname = product["Name"]
            dest_dir = output_dir / pname

            if dest_dir.exists():
                log.info("Already downloaded: %s — skipping.", pname)
                downloaded.append(dest_dir)
                continue

            zip_path  = output_dir / f"{pname}.zip"
            part_path = output_dir / f"{pname}.zip.part"
            dl_url    = f"{CDSE_DOWNLOAD_URL}/Products({pid})/$value"
            log.info("Downloading %s …", pname)

            with requests.get(dl_url, headers=self._auth_headers(),
                              stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded_bytes = 0
                with open(part_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total:
                            pct = 100 * downloaded_bytes / total
                            print(f"\r  {pname} — {pct:.1f}%", end="", flush=True)
            print()
            # Only a complete download gets the final name
            part_path.replace(zip_path)

            if unzip:
                log.info("Extracting %s …", zip_path.name)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(output_dir)
                zip_path.unlink()
                downloaded.append(dest_dir)
            else:
                downloaded.append(zip_path)

        return downloaded

    # ── Convenience ───────────────────────────────────────────────────────────

    def fetch_latest(
        self,
        region_name: str,
        sensor: str = "S1",
        days_back: int = 5,
        max_items: int = 1,
    ) -> List[Path]:
        """One-shot helper: query → download latest product(s) for a region."""
        products = self.query_region(region_name, days_back=days_back, sensor=sensor)
        if not products:
            log.warning("No products found for region '%s'.", region_name)
            return []
        return self.download(products, max_items=max_items)


# ── Planetary Computer (no credentials) ────────────────────────────────────────

class PlanetaryComputerIngester:
    """
    Sentinel-1 GRD from Microsoft Planetary Computer's public STAC catalogue.

    Scenes are stored as cloud-optimised GeoTIFFs, so instead of downloading a
    ~1.7 GB product only the pixels covering the area of interest are read.
    The window is saved in the same layout as a .SAFE product
    (<name>/measurement/*-vv.tiff, *-vh.tiff, with ground control points), so
    the rest of the pipeline treats it exactly like a Copernicus download.
    """

    def __init__(self, max_window_px: int = PC_MAX_WINDOW_PX):
        self.max_window_px = max_window_px

    # ── Query ──────────────────────────────────────────────────────────────────

    def query_region(
        self,
        region_name: str,
        days_back: int = 7,
        sensor: str = "S1",
        date_start: Optional[str] = None,
        date_end:   Optional[str] = None,
    ) -> List[Dict]:
        if region_name not in REGIONS_OF_INTEREST:
            raise ValueError(f"Unknown region '{region_name}'. "
                             f"Choose from: {list(REGIONS_OF_INTEREST.keys())}")
        return self.query_bbox(REGIONS_OF_INTEREST[region_name], days_back=days_back,
                               sensor=sensor, date_start=date_start, date_end=date_end)

    def query_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        days_back: int = 7,
        sensor: str = "S1",
        date_start: Optional[str] = None,
        date_end:   Optional[str] = None,
    ) -> List[Dict]:
        """Query using a (lat_min, lon_min, lat_max, lon_max) bounding box."""
        if sensor != "S1":
            raise ValueError(
                "The Planetary Computer source supports Sentinel-1 only. For "
                "Sentinel-2 set Copernicus credentials and use --source copernicus."
            )
        lat_min, lon_min, lat_max, lon_max = bbox
        now = datetime.now(timezone.utc)
        t_end   = date_end   or now.strftime("%Y-%m-%dT%H:%M:%SZ")
        t_start = date_start or (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        body = {
            "collections": ["sentinel-1-grd"],
            "bbox": [lon_min, lat_min, lon_max, lat_max],
            "datetime": f"{t_start}/{t_end}",
            "query": {"sar:instrument_mode": {"eq": "IW"}},
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
            "limit": SAR_MAX_RESULTS,
        }
        log.info("[S1] Querying Planetary Computer for bbox %s (%s → %s) …",
                 bbox, t_start, t_end)
        resp = requests.post(PC_STAC_URL, json=body, timeout=60)
        resp.raise_for_status()

        products = []
        for f in resp.json().get("features", []):
            assets = {k: v["href"] for k, v in f.get("assets", {}).items()
                      if k in ("vv", "vh")}
            if not assets:
                continue
            products.append({
                "Id": f["id"],
                "Name": f["id"],
                "ContentDate": {"Start": f["properties"].get("datetime", "")},
                "assets": assets,
                "aoi": tuple(bbox),
            })
        log.info("Found %d product(s).", len(products))
        return products

    # ── Download ───────────────────────────────────────────────────────────────

    def download(
        self,
        products: List[Dict],
        output_dir: Path = DATA_DIR,
        max_items: int = 2,
        unzip: bool = True,          # accepted for interface compatibility
    ) -> List[Path]:
        """
        Fetch the area-of-interest window of each product.

        Returns:
            List of local product directories, in the same order as `products`.
        """
        import numpy as np
        import rasterio
        from rasterio.control import GroundControlPoint
        from rasterio.transform import GCPTransformer
        from rasterio.windows import Window

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []

        for product in products[:max_items]:
            lat_min, lon_min, lat_max, lon_max = product["aoi"]
            tag = "%+08.3f%+08.3f" % ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
            dest_dir = output_dir / f"{product['Name']}_{tag}.SAFE"
            if dest_dir.exists() and any((dest_dir / "measurement").glob("*.tiff")):
                log.info("Already downloaded: %s — skipping.", dest_dir.name)
                downloaded.append(dest_dir)
                continue

            window, gcps = None, None
            tmp_dir = output_dir / (dest_dir.name + ".part")
            (tmp_dir / "measurement").mkdir(parents=True, exist_ok=True)

            for pol, href in product["assets"].items():
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  GDAL_HTTP_MAX_RETRY="3"):
                    with rasterio.open(self._sign(href)) as src:
                        if window is None:
                            window, gcps = self._aoi_window(src, product["aoi"])
                            if window is None:
                                break
                            log.info("Downloading %s — %d×%d px window of a "
                                     "%d×%d scene …", product["Name"],
                                     window.width, window.height,
                                     src.width, src.height)
                        data = src.read(1, window=window)

                out = tmp_dir / "measurement" / f"{product['Name'].lower()}-{pol}.tiff"
                with rasterio.open(out, "w", driver="GTiff", height=data.shape[0],
                                   width=data.shape[1], count=1, dtype=data.dtype,
                                   compress="deflate") as dst:
                    dst.write(data, 1)
                    dst.gcps = (gcps, rasterio.crs.CRS.from_epsg(4326))
                log.info("  %s: %.0f MB of pixels saved.", pol.upper(), data.nbytes / 1e6)

            if window is None:
                log.warning("%s does not cover the area of interest — skipped.",
                            product["Name"])
                for f in (tmp_dir / "measurement").glob("*"):
                    f.unlink()
                (tmp_dir / "measurement").rmdir()
                tmp_dir.rmdir()
                continue

            tmp_dir.replace(dest_dir)       # only a complete product gets the final name
            downloaded.append(dest_dir)

        return downloaded

    def fetch_latest(self, region_name: str, sensor: str = "S1",
                     days_back: int = 5, max_items: int = 1) -> List[Path]:
        """One-shot helper: query → download latest product(s) for a region."""
        products = self.query_region(region_name, days_back=days_back, sensor=sensor)
        if not products:
            log.warning("No products found for region '%s'.", region_name)
            return []
        return self.download(products, max_items=max_items)

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sign(href: str) -> str:
        """Exchange a blob URL for a short-lived signed one (anonymous)."""
        for attempt in range(4):          # the signing service times out now and then
            try:
                resp = requests.get(PC_SIGN_URL, params={"href": href}, timeout=60)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                if attempt == 3:
                    raise
            log.warning("Planetary Computer signing failed — retrying …")
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        return resp.json()["href"]

    def _aoi_window(self, src, aoi):
        """
        Pixel window of `src` covering the AOI, capped at max_window_px per side
        (centred on the AOI), plus ground control points for that window.
        Returns (None, None) if the scene does not overlap the AOI.
        """
        import numpy as np
        from rasterio.control import GroundControlPoint
        from rasterio.transform import GCPTransformer
        from rasterio.windows import Window

        lat_min, lon_min, lat_max, lon_max = aoi
        scene_gcps = src.gcps[0]
        if not scene_gcps:
            raise RuntimeError("Scene has no ground control points.")

        with GCPTransformer(scene_gcps) as tf:
            rows, cols = tf.rowcol([lon_min, lon_max, lon_min, lon_max],
                                   [lat_min, lat_min, lat_max, lat_max])
            r0, r1 = max(0, int(min(rows))), min(src.height, int(max(rows)) + 1)
            c0, c1 = max(0, int(min(cols))), min(src.width, int(max(cols)) + 1)
            if r1 - r0 < 64 or c1 - c0 < 64:
                return None, None

            cap = self.max_window_px
            if r1 - r0 > cap:
                r0 = r0 + (r1 - r0 - cap) // 2
                r1 = r0 + cap
            if c1 - c0 > cap:
                c0 = c0 + (c1 - c0 - cap) // 2
                c1 = c0 + cap

            # Re-sample the georeferencing onto a grid local to the window
            gr = np.linspace(0, r1 - r0, 7)
            gc = np.linspace(0, c1 - c0, 7)
            grid_r, grid_c = [a.ravel() for a in np.meshgrid(gr, gc, indexing="ij")]
            xs, ys = tf.xy((grid_r + r0).tolist(), (grid_c + c0).tolist(), offset="ul")

        gcps = [GroundControlPoint(row=float(r), col=float(c), x=float(x), y=float(y))
                for r, c, x, y in zip(grid_r, grid_c, xs, ys)]
        return Window(c0, r0, c1 - c0, r1 - r0), gcps


def copernicus_credentials_set() -> bool:
    return bool(COPERNICUS_USER and COPERNICUS_PASSWORD
                and COPERNICUS_USER != "your_username")


def make_ingester(source: str = DATA_SOURCE):
    """Return the ingester for 'copernicus', 'planetary' or 'auto'."""
    if source == "auto":
        source = "copernicus" if copernicus_credentials_set() else "planetary"
        log.info("Imagery source: %s%s", source,
                 "" if source == "copernicus"
                 else " (no Copernicus credentials set — using the public "
                      "Planetary Computer archive)")
    if source == "copernicus":
        return SentinelIngester()
    if source == "planetary":
        return PlanetaryComputerIngester()
    raise ValueError(f"Unknown data source '{source}'.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel data ingestion CLI")
    parser.add_argument("--region", default="South_China_Sea",
                        choices=list(REGIONS_OF_INTEREST.keys()))
    parser.add_argument("--sensor", default="S1", choices=["S1", "S2"])
    parser.add_argument("--days",   type=int, default=7)
    parser.add_argument("--max",    type=int, default=1)
    parser.add_argument("--source", default=DATA_SOURCE,
                        choices=["auto", "copernicus", "planetary"])
    args = parser.parse_args()

    ingester = make_ingester(args.source)
    paths = ingester.fetch_latest(args.region, sensor=args.sensor,
                                  days_back=args.days, max_items=args.max)
    print("\nDownloaded:")
    for p in paths:
        print(" ", p)

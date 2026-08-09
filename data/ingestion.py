"""
data/ingestion.py — Satellite imagery ingestion via the Copernicus Data Space (CDSE) API.

Supports:
  • Sentinel-1 GRD (SAR, cloud-independent, best for open ocean)
  • Sentinel-2 L2A (optical, cloud-screened)

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    COPERNICUS_USER, COPERNICUS_PASSWORD,
    CDSE_BASE_URL, CDSE_DOWNLOAD_URL,
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

        now = datetime.utcnow()
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

        url = (
            f"{CDSE_BASE_URL}/Products?"
            f"$filter=Collection/Name eq '{collection}'"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
            f" and ContentDate/Start gt {t_start}"
            f" and ContentDate/Start lt {t_end}"
            f"{extra_filters}"
            f"&$orderby=ContentDate/Start desc"
            f"&$top={max_results}"
            f"&$expand=Assets"
        )

        log.info("[%s] Querying %s products for region '%s' (%s → %s) …",
                 sensor, collection, region_name, t_start, t_end)
        resp = requests.get(url, headers=self._auth_headers(), timeout=60)
        resp.raise_for_status()
        products = resp.json().get("value", [])
        log.info("Found %d product(s).", len(products))
        return products

    def query_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        days_back: int = 5,
        sensor: str = "S1",
    ) -> List[Dict]:
        """Query using a raw (lat_min, lon_min, lat_max, lon_max) bounding box."""
        lat_min, lon_min, lat_max, lon_max = bbox
        roi_key = "_custom_bbox_"
        REGIONS_OF_INTEREST[roi_key] = bbox
        results = self.query_region(roi_key, days_back=days_back, sensor=sensor)
        del REGIONS_OF_INTEREST[roi_key]
        return results

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

            zip_path = output_dir / f"{pname}.zip"
            dl_url   = f"{CDSE_DOWNLOAD_URL}/Products({pid})/$value"
            log.info("Downloading %s …", pname)

            with requests.get(dl_url, headers=self._auth_headers(),
                              stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded_bytes = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total:
                            pct = 100 * downloaded_bytes / total
                            print(f"\r  {pname} — {pct:.1f}%", end="", flush=True)
            print()

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


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel data ingestion CLI")
    parser.add_argument("--region", default="South_China_Sea",
                        choices=list(REGIONS_OF_INTEREST.keys()))
    parser.add_argument("--sensor", default="S1", choices=["S1", "S2"])
    parser.add_argument("--days",   type=int, default=7)
    parser.add_argument("--max",    type=int, default=1)
    args = parser.parse_args()

    ingester = SentinelIngester()
    paths = ingester.fetch_latest(args.region, sensor=args.sensor,
                                  days_back=args.days, max_items=args.max)
    print("\nDownloaded:")
    for p in paths:
        print(" ", p)

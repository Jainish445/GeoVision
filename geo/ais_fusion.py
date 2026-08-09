"""
geo/ais_fusion.py — AIS (Automatic Identification System) data integration.

Cross-references detected ship positions with AIS vessel broadcasts to:
  • Identify vessel name, MMSI, flag state, and ship type.
  • Filter out commercial/cargo vessels (to focus on warships).
  • Correlate positions within a configurable radius.

AIS data sources supported:
  1. MarineTraffic API (commercial, requires API key)
  2. AISHub (free with registration)
  3. Local AIS NMEA log files (for offline/testing)

Usage:
    from geo.ais_fusion import AISFusion
    fusion = AISFusion()
    enriched = fusion.enrich(detections)
"""

import sys
import json
import time
import math
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MARINETRAFFIC_API_KEY, AISHUB_USER
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("ais_fusion")


# ── AIS Vessel Type Codes (ITU-R) that indicate military/government vessels ────
MILITARY_AIS_TYPES = {
    35,   # Military ops
    55,   # Law enforcement
    58,   # Medical transport
}

# Search radius around each detection (km)
DEFAULT_SEARCH_RADIUS_KM = 2.0


# ── AIS Record ─────────────────────────────────────────────────────────────────

class AISRecord:
    """Parsed AIS vessel record."""
    def __init__(self, data: dict):
        self.mmsi       = str(data.get("MMSI", ""))
        self.name       = str(data.get("SHIPNAME", data.get("name", "UNKNOWN"))).strip()
        self.flag       = str(data.get("FLAG", data.get("flag", "")))
        self.ship_type  = int(data.get("SHIPTYPE", data.get("type", 0)))
        self.lat        = float(data.get("LAT", data.get("lat", 0.0)))
        self.lon        = float(data.get("LON", data.get("lon", 0.0)))
        self.speed      = float(data.get("SPEED", data.get("sog", 0.0)))
        self.heading    = float(data.get("HEADING", data.get("cog", 0.0)))
        self.timestamp  = str(data.get("TIMESTAMP", ""))
        self.length     = float(data.get("LENGTH", 0.0))
        self.width      = float(data.get("WIDTH", 0.0))

    @property
    def is_military(self) -> bool:
        return self.ship_type in MILITARY_AIS_TYPES

    def __repr__(self):
        return (f"AISRecord(mmsi={self.mmsi}, name={self.name!r}, "
                f"flag={self.flag}, lat={self.lat:.4f}, lon={self.lon:.4f})")


# ── AIS Fusion ─────────────────────────────────────────────────────────────────

class AISFusion:
    """
    Fetches AIS data and cross-references it with detection positions.

    Supports three backends:
      - "marinetraffic" : Real-time commercial API
      - "aishub"        : Free real-time feed
      - "file"          : Local NMEA or CSV AIS log
    """

    def __init__(
        self,
        backend: str = "marinetraffic",
        ais_file: Optional[Path] = None,
        search_radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
        cache_ttl_minutes: int = 15,
    ):
        self.backend           = backend
        self.ais_file          = ais_file
        self.search_radius_km  = search_radius_km
        self.cache_ttl         = timedelta(minutes=cache_ttl_minutes)
        self._cache:     List[AISRecord] = []
        self._cache_time: Optional[datetime] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def enrich(
        self,
        detections: List[Detection],
        bbox: Optional[tuple] = None,   # (lat_min, lon_min, lat_max, lon_max)
    ) -> List[Detection]:
        """
        For each detection with a known lat/lon, find the nearest AIS vessel
        within search_radius_km and copy vessel metadata into the Detection.

        Args:
            detections: List of Detection objects.
            bbox:       Bounding box to fetch AIS data for.
                        If None, it is computed from detection positions.

        Returns:
            Same list with vessel_name, vessel_mmsi, vessel_flag populated.
        """
        # Compute bbox from detections if not provided
        if bbox is None:
            lats = [d.lat for d in detections if d.lat is not None]
            lons = [d.lon for d in detections if d.lon is not None]
            if not lats:
                log.warning("No valid positions in detections — skipping AIS.")
                return detections
            margin = 0.5  # degrees
            bbox = (min(lats) - margin, min(lons) - margin,
                    max(lats) + margin, max(lons) + margin)

        vessels = self._fetch_vessels(bbox)
        log.info("AIS: %d vessel records fetched for region.", len(vessels))

        for d in detections:
            if d.lat is None or d.lon is None:
                continue
            match = self._nearest_vessel(d.lat, d.lon, vessels)
            if match:
                d.vessel_name = match.name if match.name != "UNKNOWN" else None
                d.vessel_mmsi = match.mmsi
                d.vessel_flag = match.flag
                log.debug(
                    "Detection at (%.4f, %.4f) matched → %s (%s)",
                    d.lat, d.lon, match.name, match.flag,
                )

        return detections

    def filter_military(
        self, vessels: List[AISRecord]
    ) -> List[AISRecord]:
        """Return only vessels with military AIS type codes."""
        return [v for v in vessels if v.is_military]

    # ── Backends ───────────────────────────────────────────────────────────────

    def _fetch_vessels(self, bbox: tuple) -> List[AISRecord]:
        """Route to the appropriate backend, with caching."""
        now = datetime.utcnow()
        if (self._cache_time and (now - self._cache_time) < self.cache_ttl
                and self._cache):
            log.debug("Using cached AIS data (%d records).", len(self._cache))
            return self._cache

        if self.backend == "marinetraffic":
            records = self._fetch_marinetraffic(bbox)
        elif self.backend == "aishub":
            records = self._fetch_aishub(bbox)
        elif self.backend == "file":
            records = self._read_ais_file(self.ais_file)
        else:
            log.error("Unknown AIS backend: %s", self.backend)
            return []

        self._cache      = records
        self._cache_time = now
        return records

    def _fetch_marinetraffic(self, bbox: tuple) -> List[AISRecord]:
        """
        MarineTraffic GetVesselsInArea API.
        Docs: https://www.marinetraffic.com/en/ais-api-services
        Requires a paid API key with the 'vessels_in_area' endpoint access.
        """
        if not MARINETRAFFIC_API_KEY:
            log.warning("MARINETRAFFIC_API_KEY not set — skipping AIS enrichment.")
            return []
        if not HAS_REQUESTS:
            log.error("requests library not installed.")
            return []

        lat_min, lon_min, lat_max, lon_max = bbox
        url = "https://services.marinetraffic.com/api/getVesselsInArea/v:8"
        params = {
            "apikey":   MARINETRAFFIC_API_KEY,
            "MINLAT":   lat_min,
            "MINLON":   lon_min,
            "MAXLAT":   lat_max,
            "MAXLON":   lon_max,
            "protocol": "jsono",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [AISRecord(v) for v in data]
            return []
        except Exception as e:
            log.error("MarineTraffic API error: %s", e)
            return []

    def _fetch_aishub(self, bbox: tuple) -> List[AISRecord]:
        """
        AISHub real-time data feed (free with registration).
        Docs: https://www.aishub.net/api
        """
        if not AISHUB_USER:
            log.warning("AISHUB_USER not set — skipping AIS enrichment.")
            return []
        if not HAS_REQUESTS:
            return []

        lat_min, lon_min, lat_max, lon_max = bbox
        url = "https://data.aishub.net/ws.php"
        params = {
            "username":  AISHUB_USER,
            "format":    "1",
            "output":    "json",
            "compress":  "0",
            "latmin":    lat_min,
            "latmax":    lat_max,
            "lonmin":    lon_min,
            "lonmax":    lon_max,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # AISHub returns [header_dict, [vessel_list]]
            if isinstance(payload, list) and len(payload) > 1:
                vessels = payload[1] if isinstance(payload[1], list) else []
                return [AISRecord(v) for v in vessels]
            return []
        except Exception as e:
            log.error("AISHub API error: %s", e)
            return []

    def _read_ais_file(self, path: Optional[Path]) -> List[AISRecord]:
        """
        Read AIS data from a local JSON or CSV file.
        JSON format: list of vessel dicts.
        CSV format: header row matching AISRecord field names.
        """
        if path is None or not Path(path).exists():
            log.error("AIS file not found: %s", path)
            return []

        path = Path(path)
        records: List[AISRecord] = []

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                records = [AISRecord(v) for v in data]

        elif path.suffix == ".csv":
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        records.append(AISRecord(row))
                    except (ValueError, KeyError):
                        continue

        log.info("Loaded %d AIS records from %s.", len(records), path.name)
        return records

    # ── Spatial matching ───────────────────────────────────────────────────────

    def _nearest_vessel(
        self,
        lat: float,
        lon: float,
        vessels: List[AISRecord],
    ) -> Optional[AISRecord]:
        """Return the nearest AIS vessel within self.search_radius_km, or None."""
        best_dist = self.search_radius_km
        best_vessel = None

        for v in vessels:
            dist = _haversine_km(lat, lon, v.lat, v.lon)
            if dist < best_dist:
                best_dist   = dist
                best_vessel = v

        return best_vessel


# ── Haversine ──────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with local file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ais-file", type=Path, required=True,
                        help="Path to local AIS JSON or CSV file")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    fusion = AISFusion(backend="file", ais_file=args.ais_file)
    vessels = fusion._read_ais_file(args.ais_file)
    match = fusion._nearest_vessel(args.lat, args.lon, vessels)
    if match:
        print(f"Nearest vessel: {match}")
    else:
        print("No vessel found within search radius.")

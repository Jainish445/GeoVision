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
from datetime import datetime, timedelta, timezone

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MARINETRAFFIC_API_KEY, AISHUB_USER, CIVILIAN_LABEL, UNIDENTIFIED_LABEL,
    CROWD_RADIUS_KM, CROWD_MIN_NEIGHBOURS, CROWD_CONFIDENCE,
)
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("ais_fusion")


# ── AIS Vessel Type Codes (ITU-R) that indicate military/government vessels ────
MILITARY_AIS_TYPES = {
    35,   # Military ops
    55,   # Law enforcement
    58,   # Medical transport
}

# AIS type codes that positively identify a non-military vessel
CIVILIAN_AIS_TYPES = (
    {30, 31, 32, 36, 37, 52}          # fishing, towing, sailing, pleasure, tug
    | set(range(60, 90))              # passenger, cargo, tanker
)


def _num(value, cast=float, default=0):
    """Tolerant numeric conversion — CSV exports contain blanks and 'N/A'."""
    try:
        return cast(float(value))
    except (TypeError, ValueError):
        return default


# Search radius around each detection (km)
DEFAULT_SEARCH_RADIUS_KM = 2.0


# ── AIS Record ─────────────────────────────────────────────────────────────────

class AISRecord:
    """Parsed AIS vessel record."""
    def __init__(self, data: dict):
        self.mmsi       = str(data.get("MMSI", ""))
        self.name       = str(data.get("SHIPNAME", data.get("name", "UNKNOWN"))).strip()
        self.flag       = str(data.get("FLAG", data.get("flag", "")))
        self.ship_type  = _num(data.get("SHIPTYPE", data.get("type", 0)), int)
        self.lat        = _num(data.get("LAT", data.get("lat")), default=None)
        self.lon        = _num(data.get("LON", data.get("lon")), default=None)
        self.speed      = _num(data.get("SPEED", data.get("sog", 0.0)))
        self.heading    = _num(data.get("HEADING", data.get("cog", 0.0)))
        self.timestamp  = str(data.get("TIMESTAMP", ""))
        self.length     = _num(data.get("LENGTH", 0.0))
        self.width      = _num(data.get("WIDTH", 0.0))

    @property
    def is_military(self) -> bool:
        return self.ship_type in MILITARY_AIS_TYPES

    @property
    def is_civilian(self) -> bool:
        return self.ship_type in CIVILIAN_AIS_TYPES

    def __repr__(self):
        return (f"AISRecord(mmsi={self.mmsi}, name={self.name!r}, "
                f"flag={self.flag}, lat={self.lat}, lon={self.lon})")


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
        self._cache_bbox: Optional[tuple] = None

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
            Detections matched to a commercial vessel (cargo, tanker,
            passenger, fishing …) are relabelled CIVILIAN_LABEL.
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
                d.vessel_flag = match.flag or None
                if match.is_civilian:
                    d.warship_type = CIVILIAN_LABEL
                    d.warship_conf = None
                    d.type_source  = "ais"
                elif match.is_military and d.warship_type in (None, UNIDENTIFIED_LABEL):
                    # AIS says it is a military vessel; the measured hull says
                    # which kind. Together that supports a warship type.
                    if d.size_class:
                        d.warship_type = d.size_class
                        d.warship_conf = d.size_conf
                        d.type_source  = "ais+size"
                    else:
                        d.warship_type = "Unknown Warship"
                        d.warship_conf = None
                        d.type_source  = "ais"
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
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if (self._cache_time and (now - self._cache_time) < self.cache_ttl
                and self._cache and self._cache_bbox in (None, bbox)):
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
        self._cache_bbox = bbox
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
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = [AISRecord(v) for v in data]

        elif path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8-sig") as f:
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
            if v.lat is None or v.lon is None:
                continue
            dist = _haversine_km(lat, lon, v.lat, v.lon)
            if dist < best_dist:
                best_dist   = dist
                best_vessel = v

        return best_vessel


# ── Context: crowded commercial waters ────────────────────────────────────────

def label_crowded_waters(
    detections: List[Detection],
    radius_km: float = CROWD_RADIUS_KM,
    min_neighbours: int = CROWD_MIN_NEIGHBOURS,
    confidence: float = CROWD_CONFIDENCE,
) -> List[Detection]:
    """
    Mark still-unidentified ships that sit in a crowd of other ships as
    probable civilian vessels.

    Dozens of ships within a few kilometres of each other is what commercial
    anchorages, port approaches and shipping lanes look like; warships in
    company keep kilometres apart. This is an inference from context, not an
    identification — it is recorded as type_source="context" with a modest
    confidence, and anything AIS or the CNN has already labelled is untouched.
    """
    if min_neighbours <= 0:
        return detections

    located = [d for d in detections if d.lat is not None and d.lon is not None]
    n_marked = 0
    for d in located:
        if d.warship_type not in (None, UNIDENTIFIED_LABEL):
            continue
        # Cheap latitude pre-filter before the haversine
        max_dlat = radius_km / 110.0
        neighbours = sum(
            1 for o in located
            if o is not d and abs(o.lat - d.lat) <= max_dlat
            and _haversine_km(d.lat, d.lon, o.lat, o.lon) <= radius_km
        )
        if neighbours >= min_neighbours:
            d.warship_type = CIVILIAN_LABEL
            d.warship_conf = confidence
            d.type_source  = "context"
            n_marked += 1

    if n_marked:
        log.info("Context: %d unidentified ship(s) in crowded waters marked "
                 "probable civilian.", n_marked)
    return detections


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

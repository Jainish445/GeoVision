"""
utils/geo_utils.py — Geographic utility functions used across the pipeline.

Includes:
  • Haversine distance and bearing calculations
  • Bounding-box helpers (expand, to WKT, to GeoJSON)
  • Pixel ↔ geographic coordinate conversion
  • ROI grid generation (split a large ROI into sub-tiles for querying)
  • Speed / course estimation between two positions
"""

import math
import json
from typing import Tuple, List, Dict, Optional


# ── Types ──────────────────────────────────────────────────────────────────────
LatLon = Tuple[float, float]          # (latitude, longitude)
BBox   = Tuple[float, float, float, float]  # (lat_min, lon_min, lat_max, lon_max)


# ── Constants ──────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0


# ── Haversine ──────────────────────────────────────────────────────────────────

def haversine_km(p1: LatLon, p2: LatLon) -> float:
    """
    Calculate the great-circle distance between two lat/lon points (in km).

    >>> round(haversine_km((0, 0), (0, 1)), 3)
    111.195
    """
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_nm(p1: LatLon, p2: LatLon) -> float:
    """Distance in nautical miles (1 nm ≈ 1.852 km)."""
    return haversine_km(p1, p2) / 1.852


def bearing_degrees(p1: LatLon, p2: LatLon) -> float:
    """
    Initial bearing (0–360°) from p1 to p2 (clockwise from North).

    >>> round(bearing_degrees((0, 0), (0, 1)), 1)
    90.0
    """
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination_point(origin: LatLon, bearing_deg: float, distance_km: float) -> LatLon:
    """
    Calculate the destination point given origin, bearing (degrees), and distance (km).
    Useful for projecting ship positions forward.
    """
    lat1 = math.radians(origin[0])
    lon1 = math.radians(origin[1])
    b    = math.radians(bearing_deg)
    d    = distance_km / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


# ── Speed & Course ─────────────────────────────────────────────────────────────

def speed_knots(p1: LatLon, p2: LatLon, elapsed_hours: float) -> float:
    """
    Estimate speed in knots given two positions and elapsed time in hours.
    Returns 0.0 if elapsed_hours <= 0.
    """
    if elapsed_hours <= 0:
        return 0.0
    dist_nm = haversine_nm(p1, p2)
    return dist_nm / elapsed_hours


def estimate_next_position(
    positions: List[Dict],    # [{lat, lon, timestamp (ISO str)}, ...]
    hours_ahead: float = 6.0,
) -> Optional[LatLon]:
    """
    Dead-reckoning: estimate the next position of a vessel using the
    last two known positions to compute speed and bearing.

    Args:
        positions: Chronologically ordered list of position dicts.
        hours_ahead: How many hours into the future to project.

    Returns:
        (lat, lon) of projected position, or None if insufficient data.
    """
    if len(positions) < 2:
        return None

    from datetime import datetime

    p1_d = positions[-2]
    p2_d = positions[-1]

    try:
        t1 = datetime.fromisoformat(p1_d["timestamp"].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(p2_d["timestamp"].replace("Z", "+00:00"))
        elapsed_h = (t2 - t1).total_seconds() / 3600
    except (KeyError, ValueError):
        return None

    if elapsed_h <= 0:
        return None

    p1 = (p1_d["lat"], p1_d["lon"])
    p2 = (p2_d["lat"], p2_d["lon"])

    brg = bearing_degrees(p1, p2)
    spd = speed_knots(p1, p2, elapsed_h)
    dist_km = (spd * hours_ahead) * 1.852

    return destination_point(p2, brg, dist_km)


# ── Bounding Box ───────────────────────────────────────────────────────────────

def bbox_expand(bbox: BBox, margin_deg: float = 0.5) -> BBox:
    """Expand a bounding box by margin_deg on all sides."""
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        max(-90,  lat_min - margin_deg),
        max(-180, lon_min - margin_deg),
        min(90,   lat_max + margin_deg),
        min(180,  lon_max + margin_deg),
    )


def bbox_to_wkt(bbox: BBox) -> str:
    """Convert (lat_min, lon_min, lat_max, lon_max) to WKT POLYGON string."""
    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )


def bbox_to_geojson(bbox: BBox) -> dict:
    """Convert bbox to a GeoJSON Polygon geometry."""
    lat_min, lon_min, lat_max, lon_max = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]],
    }


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate area of a lat/lon bounding box in km²."""
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_c = math.radians((lat_min + lat_max) / 2)
    height_km = haversine_km((lat_min, lon_min), (lat_max, lon_min))
    width_km  = haversine_km((lat_c, lon_min),   (lat_c, lon_max))
    return height_km * width_km


def split_bbox_grid(bbox: BBox, cell_deg: float = 5.0) -> List[BBox]:
    """
    Split a large bounding box into a grid of smaller cells.
    Useful for parallel satellite queries over large regions.

    Args:
        bbox:     (lat_min, lon_min, lat_max, lon_max)
        cell_deg: Size of each grid cell in degrees.

    Returns:
        List of sub-bounding boxes.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    cells: List[BBox] = []

    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            cells.append((
                lat,
                lon,
                min(lat + cell_deg, lat_max),
                min(lon + cell_deg, lon_max),
            ))
            lon += cell_deg
        lat += cell_deg

    return cells


def point_in_bbox(lat: float, lon: float, bbox: BBox) -> bool:
    """Return True if (lat, lon) falls within the bounding box."""
    lat_min, lon_min, lat_max, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


# ── Pixel ↔ Geo ────────────────────────────────────────────────────────────────

def pixel_to_geo(
    px: float, py: float,
    geo_transform: Tuple[float, float, float, float, float, float],
) -> LatLon:
    """
    Convert pixel (col, row) to (lat, lon) using an affine geo-transform tuple.
    Geo-transform format: (lon_origin, pixel_width, 0, lat_origin, 0, -pixel_height)
    This matches the GDAL/rasterio convention.
    """
    lon_origin, pixel_w, _, lat_origin, _, pixel_h = geo_transform
    lon = lon_origin + px * pixel_w
    lat = lat_origin + py * pixel_h
    return lat, lon


def geo_to_pixel(
    lat: float, lon: float,
    geo_transform: Tuple[float, float, float, float, float, float],
) -> Tuple[int, int]:
    """Inverse of pixel_to_geo — returns (col, row) pixel indices."""
    lon_origin, pixel_w, _, lat_origin, _, pixel_h = geo_transform
    col = int((lon - lon_origin) / pixel_w)
    row = int((lat - lat_origin) / pixel_h)
    return col, row


# ── GeoJSON helpers ────────────────────────────────────────────────────────────

def detections_to_geojson(detections: list) -> dict:
    """
    Convert a list of Detection objects (or dicts) to a GeoJSON FeatureCollection.
    Compatible with the Detection dataclass from models/detector.py.
    """
    features = []
    for d in detections:
        if hasattr(d, "to_dict"):
            data = d.to_dict()
        else:
            data = dict(d)

        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            continue

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {k: v for k, v in data.items() if k not in ("lat", "lon")},
        })

    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson: dict, path: str) -> None:
    """Save a GeoJSON dict to a file."""
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:        # Windows consoles default to cp1252, which lacks '≈' and '°'
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Quick sanity checks
    print("haversine_km((0,0),(0,1))     =", round(haversine_km((0, 0), (0, 1)), 3), "km")
    print("haversine_km((51.5,-0.1),(48.9,2.3)) =",
          round(haversine_km((51.5, -0.1), (48.9, 2.3)), 1), "km  (London->Paris ≈340 km)")
    print("bearing (0,0)->(0,1)           =", bearing_degrees((0, 0), (0, 1)), "°  (expect 90)")
    print("bearing (0,0)->(1,0)           =", bearing_degrees((0, 0), (1, 0)), "°  (expect 0)")

    bbox = (0.0, 99.0, 25.0, 125.0)
    cells = split_bbox_grid(bbox, cell_deg=5.0)
    print(f"split_bbox_grid South China Sea (5°) -> {len(cells)} cells")

    proj = estimate_next_position([
        {"lat": 15.0, "lon": 114.0, "timestamp": "2024-01-15T00:00:00Z"},
        {"lat": 15.1, "lon": 114.1, "timestamp": "2024-01-15T06:00:00Z"},
    ], hours_ahead=6.0)
    print(f"estimate_next_position (6h ahead) -> {proj}")

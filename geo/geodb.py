"""
geo/geodb.py — Persistent storage for ship detections and track history.

Uses SQLite with optional PostGIS (PostgreSQL) support for production.
The schema stores every detection event, track history, and region metadata.

Tables:
  • detections   — individual detection events (one row per ship per acquisition)
  • tracks       — aggregated track metadata (one row per unique vessel)
  • acquisitions — satellite acquisition metadata

Usage:
    from geo.geodb import GeoDB
    db = GeoDB()
    db.insert_detections(detections, region="South_China_Sea", acquisition_id="ACQ_001")
    df = db.query_tracks(region="South_China_Sea")
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("geodb")


# ── Schema ─────────────────────────────────────────────────────────────────────

CREATE_ACQUISITIONS = """
CREATE TABLE IF NOT EXISTS acquisitions (
    id            TEXT PRIMARY KEY,
    region        TEXT NOT NULL,
    sensor        TEXT NOT NULL,          -- S1 or S2
    product_name  TEXT,
    acquisition_ts TEXT,                  -- ISO-8601 timestamp from satellite
    ingestion_ts  TEXT NOT NULL,          -- when we processed it
    n_tiles       INTEGER DEFAULT 0,
    n_detections  INTEGER DEFAULT 0
);
"""

CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    acquisition_id TEXT REFERENCES acquisitions(id),
    track_id      INTEGER,
    tile_path     TEXT,
    lat           REAL,
    lon           REAL,
    bbox_x1       REAL,
    bbox_y1       REAL,
    bbox_x2       REAL,
    bbox_y2       REAL,
    confidence    REAL,
    class_name    TEXT,
    warship_type  TEXT,
    warship_conf  REAL,
    vessel_name   TEXT,
    vessel_mmsi   TEXT,
    vessel_flag   TEXT,
    timestamp     TEXT,
    region        TEXT
);
"""

CREATE_TRACKS = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id       INTEGER PRIMARY KEY,
    region         TEXT,
    first_seen     TEXT,
    last_seen      TEXT,
    n_detections   INTEGER DEFAULT 0,
    warship_type   TEXT,
    vessel_name    TEXT,
    vessel_mmsi    TEXT,
    vessel_flag    TEXT,
    trail_json     TEXT    -- JSON array of {lat,lon,timestamp} dicts
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_det_track ON detections(track_id);",
    "CREATE INDEX IF NOT EXISTS idx_det_region ON detections(region);",
    "CREATE INDEX IF NOT EXISTS idx_det_lat    ON detections(lat);",
    "CREATE INDEX IF NOT EXISTS idx_det_lon    ON detections(lon);",
    "CREATE INDEX IF NOT EXISTS idx_acq_region ON acquisitions(region);",
]


# ── GeoDB ──────────────────────────────────────────────────────────────────────

class GeoDB:
    """
    SQLite-backed store for all pipeline outputs.
    Thread-safe (each call opens and closes a connection).
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        log.info("GeoDB initialised: %s", self.db_path)

    # ── Connection helper ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(CREATE_ACQUISITIONS)
            conn.execute(CREATE_DETECTIONS)
            conn.execute(CREATE_TRACKS)
            for idx_sql in CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()

    # ── Acquisitions ───────────────────────────────────────────────────────────

    def insert_acquisition(
        self,
        acq_id:       str,
        region:       str,
        sensor:       str,
        product_name: str = "",
        acq_ts:       str = "",
        n_tiles:      int = 0,
    ) -> None:
        sql = """
        INSERT OR IGNORE INTO acquisitions
            (id, region, sensor, product_name, acquisition_ts, ingestion_ts, n_tiles)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(sql, (acq_id, region, sensor, product_name, acq_ts, now, n_tiles))
            conn.commit()

    def update_acquisition_count(self, acq_id: str, n_detections: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE acquisitions SET n_detections=? WHERE id=?",
                (n_detections, acq_id),
            )
            conn.commit()

    # ── Detections ─────────────────────────────────────────────────────────────

    def insert_detections(
        self,
        detections:    List[Detection],
        region:        str,
        acquisition_id: str,
    ) -> int:
        """
        Insert a list of Detection objects.
        Returns the number of rows inserted.
        """
        if not detections:
            return 0

        sql = """
        INSERT INTO detections
            (acquisition_id, track_id, tile_path, lat, lon,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             confidence, class_name, warship_type, warship_conf,
             vessel_name, vessel_mmsi, vessel_flag, timestamp, region)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox_px
            rows.append((
                acquisition_id,
                d.track_id,
                str(d.tile_path),
                d.lat,
                d.lon,
                x1, y1, x2, y2,
                d.confidence,
                d.class_name,
                d.warship_type,
                d.warship_conf,
                d.vessel_name,
                d.vessel_mmsi,
                d.vessel_flag,
                d.timestamp,
                region,
            ))

        with self._connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()

        self.update_acquisition_count(acquisition_id, len(rows))
        log.info("Inserted %d detection(s) for acquisition '%s'.", len(rows), acquisition_id)
        return len(rows)

    def query_detections(
        self,
        region:   Optional[str] = None,
        track_id: Optional[int] = None,
        since:    Optional[str] = None,    # ISO-8601 timestamp
        limit:    int = 1000,
    ) -> List[dict]:
        """Query detection records with optional filters."""
        clauses = []
        params  = []

        if region:
            clauses.append("region = ?")
            params.append(region)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql   = f"SELECT * FROM detections {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(r) for r in rows]

    # ── Tracks ────────────────────────────────────────────────────────────────

    def upsert_track(
        self,
        track_id:     int,
        region:       str,
        detections:   List[Detection],
        trail:        List[dict],
    ) -> None:
        """
        Insert or update a track record with the latest detection metadata
        and the full position trail.
        """
        if not detections:
            return

        # Determine dominant warship type (most common among confirmed detections)
        types = [d.warship_type for d in detections if d.warship_type]
        dominant_type = max(set(types), key=types.count) if types else None

        # Latest detection for metadata
        latest = sorted(
            [d for d in detections if d.timestamp],
            key=lambda d: d.timestamp or "",
        )
        first  = latest[0]  if latest else detections[0]
        newest = latest[-1] if latest else detections[-1]

        sql = """
        INSERT INTO tracks
            (track_id, region, first_seen, last_seen, n_detections,
             warship_type, vessel_name, vessel_mmsi, vessel_flag, trail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            last_seen    = excluded.last_seen,
            n_detections = excluded.n_detections,
            warship_type = excluded.warship_type,
            vessel_name  = COALESCE(excluded.vessel_name, vessel_name),
            vessel_mmsi  = COALESCE(excluded.vessel_mmsi, vessel_mmsi),
            vessel_flag  = COALESCE(excluded.vessel_flag, vessel_flag),
            trail_json   = excluded.trail_json
        """
        with self._connect() as conn:
            conn.execute(sql, (
                track_id,
                region,
                first.timestamp or "",
                newest.timestamp or "",
                len(detections),
                dominant_type,
                newest.vessel_name,
                newest.vessel_mmsi,
                newest.vessel_flag,
                json.dumps(trail),
            ))
            conn.commit()

    def query_tracks(
        self,
        region: Optional[str] = None,
        active_since: Optional[str] = None,
    ) -> List[dict]:
        """Return all (or filtered) track records with parsed trail JSON."""
        clauses, params = [], []
        if region:
            clauses.append("region = ?")
            params.append(region)
        if active_since:
            clauses.append("last_seen >= ?")
            params.append(active_since)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql   = f"SELECT * FROM tracks {where} ORDER BY last_seen DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            try:
                d["trail"] = json.loads(d.get("trail_json") or "[]")
            except json.JSONDecodeError:
                d["trail"] = []
            result.append(d)
        return result

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary statistics dict."""
        with self._connect() as conn:
            n_acq  = conn.execute("SELECT COUNT(*) FROM acquisitions").fetchone()[0]
            n_det  = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            n_trk  = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            regions = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT region FROM detections"
                ).fetchall()
            ]
        return {
            "acquisitions": n_acq,
            "detections":   n_det,
            "tracks":       n_trk,
            "regions":      regions,
        }

    def export_geojson(self, output_path: Path, region: Optional[str] = None) -> Path:
        """Export all detections as a GeoJSON FeatureCollection."""
        detections = self.query_detections(region=region, limit=100_000)
        features = []
        for d in detections:
            if d.get("lat") and d.get("lon"):
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d["lon"], d["lat"]],
                    },
                    "properties": {
                        k: v for k, v in d.items()
                        if k not in ("lat", "lon")
                    },
                })

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)
        log.info("Exported %d features to %s", len(features), output_path)
        return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db = GeoDB()
    print("DB Summary:", json.dumps(db.summary(), indent=2))

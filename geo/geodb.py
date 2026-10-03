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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, UNIDENTIFIED_LABEL
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("geodb")


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a 'Z' suffix (sorts lexicographically)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_ts(ts: Optional[str]) -> Optional[str]:
    """
    Normalise any ISO-8601 timestamp to 'YYYY-MM-DDTHH:MM:SSZ' (UTC) so that
    string comparison in SQL matches chronological order.
    """
    if not ts:
        return ts
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    length_m      REAL,
    beam_m        REAL,
    size_class    TEXT,           -- warship type with a hull this size
    type_source   TEXT,           -- evidence behind warship_type
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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, and always close it."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(CREATE_ACQUISITIONS)
            conn.execute(CREATE_DETECTIONS)
            conn.execute(CREATE_TRACKS)
            # Migrate databases created before these columns existed
            cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
            for col, kind in (("length_m", "REAL"), ("beam_m", "REAL"),
                              ("size_class", "TEXT"), ("type_source", "TEXT")):
                if col not in cols:
                    conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {kind}")
            for idx_sql in CREATE_INDEXES:
                conn.execute(idx_sql)

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
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(sql, (acq_id, region, sensor, product_name,
                               normalize_ts(acq_ts), now, n_tiles))

    def update_acquisition_count(self, acq_id: str, n_detections: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE acquisitions SET n_detections=? WHERE id=?",
                (n_detections, acq_id),
            )

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
             length_m, beam_m, size_class, type_source,
             vessel_name, vessel_mmsi, vessel_flag, timestamp, region)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                d.length_m,
                d.beam_m,
                d.size_class,
                d.type_source,
                d.vessel_name,
                d.vessel_mmsi,
                d.vessel_flag,
                normalize_ts(d.timestamp),
                region,
            ))

        with self._connect() as conn:
            conn.executemany(sql, rows)

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
            params.append(normalize_ts(since))

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
                normalize_ts(first.timestamp) or "",
                normalize_ts(newest.timestamp) or "",
                len(detections),
                dominant_type,
                newest.vessel_name,
                newest.vessel_mmsi,
                newest.vessel_flag,
                json.dumps(trail),
            ))

    def rebuild_tracks(self, track_ids: Iterable[int]) -> None:
        """
        Recompute track rows from the detections table: first/last seen,
        cumulative detection count, dominant warship type, latest AIS identity
        and the full chronological trail.
        """
        with self._connect() as conn:
            for tid in set(t for t in track_ids if t is not None):
                rows = conn.execute(
                    "SELECT * FROM detections WHERE track_id = ? "
                    "ORDER BY timestamp, id", (tid,)
                ).fetchall()
                if not rows:
                    conn.execute("DELETE FROM tracks WHERE track_id = ?", (tid,))
                    continue

                types = [r["warship_type"] for r in rows if r["warship_type"]]
                # Prefer a specific type over the catch-all labels when both occur
                specific = [t for t in types
                            if t not in ("Unknown Warship", UNIDENTIFIED_LABEL)] or types
                dominant = max(set(specific), key=specific.count) if specific else None

                def latest(col):
                    return next((r[col] for r in reversed(rows) if r[col]), None)

                trail = [
                    {"lat": r["lat"], "lon": r["lon"],
                     "timestamp": r["timestamp"] or "",
                     "confidence": r["confidence"]}
                    for r in rows if r["lat"] is not None and r["lon"] is not None
                ]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tracks
                        (track_id, region, first_seen, last_seen, n_detections,
                         warship_type, vessel_name, vessel_mmsi, vessel_flag,
                         trail_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tid, rows[-1]["region"], rows[0]["timestamp"] or "",
                     rows[-1]["timestamp"] or "", len(rows), dominant,
                     latest("vessel_name"), latest("vessel_mmsi"),
                     latest("vessel_flag"), json.dumps(trail)),
                )

    def next_track_id(self) -> int:
        """Next unused track ID (IDs are global across regions and runs)."""
        with self._connect() as conn:
            a = conn.execute("SELECT MAX(track_id) FROM tracks").fetchone()[0] or 0
            b = conn.execute("SELECT MAX(track_id) FROM detections").fetchone()[0] or 0
        return max(a, b) + 1

    def clear_demo(self) -> int:
        """Remove all synthetic demo rows (sensor='DEMO'). Returns rows deleted."""
        with self._connect() as conn:
            demo = "SELECT id FROM acquisitions WHERE sensor = 'DEMO'"
            tids = [r[0] for r in conn.execute(
                f"SELECT DISTINCT track_id FROM detections "
                f"WHERE acquisition_id IN ({demo})")]
            n = conn.execute(
                f"DELETE FROM detections WHERE acquisition_id IN ({demo})"
            ).rowcount
            conn.execute("DELETE FROM acquisitions WHERE sensor = 'DEMO'")
        self.rebuild_tracks(tids)
        return n

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
            params.append(normalize_ts(active_since))

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
            if d.get("lat") is not None and d.get("lon") is not None:
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

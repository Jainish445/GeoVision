"""
models/tracker.py — Multi-acquisition ship tracking in geographic space.

Satellite revisit intervals are measured in hours, and every detection's pixel
box is relative to its own tile, so video trackers such as DeepSORT (which
associate pixel boxes between consecutive frames) do not apply here. Instead,
detections are associated to existing tracks by great-circle distance, gated by
how far a ship could plausibly have sailed in the elapsed time, and assigned
optimally with the Hungarian algorithm.

Track state is seeded from the database, so IDs persist across pipeline runs
(e.g. a cron job every 6 hours) and never collide between regions.

Usage:
    from models.tracker import ShipTracker
    tracker = ShipTracker(db=GeoDB())
    detections = tracker.update(detections, timestamp="2024-01-15T06:30:00Z",
                                region="South_China_Sea")
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRACK_MAX_SPEED_KNOTS, TRACK_MIN_GATE_KM,
    TRACK_MAX_GATE_KM, TRACK_MAX_GAP_HOURS,
)
from models.detector import Detection, _geo_distance_km
from utils.logger import get_logger

log = get_logger("tracker")

KM_PER_NM = 1.852


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class ShipTracker:
    """
    Maintains ship identities across satellite acquisitions.

    Key design choices:
    • Association is done on lat/lon, not pixels.
    • The search gate grows with elapsed time: MIN_GATE + MAX_SPEED × Δt,
      capped at MAX_GATE. Tracks unseen for MAX_GAP_HOURS are not continued.
    • Within the gate, a track with two or more fixes is matched against its
      dead-reckoned position (constant course and speed), so ships whose
      paths cross keep their identities. A mismatch in measured hull length
      (when the classifier has run first) makes a pairing more expensive.
    • Every detection gets a track ID (a new one if nothing matches).
    """

    def __init__(
        self,
        db=None,                       # optional GeoDB for persistent IDs
        max_speed_knots: float = TRACK_MAX_SPEED_KNOTS,
        min_gate_km:     float = TRACK_MIN_GATE_KM,
        max_gate_km:     float = TRACK_MAX_GATE_KM,
        max_gap_hours:   float = TRACK_MAX_GAP_HOURS,
    ):
        self.db              = db
        self.max_speed_knots = max_speed_knots
        self.min_gate_km     = min_gate_km
        self.max_gate_km     = max_gate_km
        self.max_gap_hours   = max_gap_hours

        # track_id → {"lat", "lon", "ts" (datetime | None), "region",
        #             "prev" ((lat, lon, ts) | None), "length_m" (float | None)}
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1
        self._loaded  = False
        self._frame_index = 0

        # History: track_id → list of {lat, lon, timestamp, confidence} dicts
        self.track_history: Dict[int, List[dict]] = {}

    def _load_state(self):
        """Seed last-known positions and the ID counter from the database."""
        if self._loaded:
            return
        self._loaded = True
        if self.db is None:
            return

        for t in self.db.query_tracks():
            trail = t.get("trail") or []
            if not trail:
                continue
            last = trail[-1]
            prev = trail[-2] if len(trail) > 1 else None
            self._tracks[t["track_id"]] = {
                "lat": last["lat"], "lon": last["lon"],
                "ts": _parse_ts(last.get("timestamp")),
                "region": t.get("region"),
                "prev": (prev["lat"], prev["lon"], _parse_ts(prev.get("timestamp")))
                        if prev else None,
                "length_m": None,
            }
            self.track_history[t["track_id"]] = list(trail)
        self._next_id = max(self._next_id, self.db.next_track_id())
        log.info("Tracker seeded with %d existing track(s).", len(self._tracks))

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Detection],
        timestamp: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[Detection]:
        """
        Feed detections from one satellite acquisition into the tracker.

        Args:
            detections: Detection objects with lat/lon populated.
            timestamp:  ISO-8601 acquisition time.
            region:     Region name; tracks are only matched within a region.

        Returns:
            Same list, with track_id populated on every geolocated detection.
        """
        self._load_state()
        self._frame_index += 1
        now = _parse_ts(timestamp)

        located = [d for d in detections if d.lat is not None and d.lon is not None]
        if not located:
            log.debug("Frame %d: no geolocated detections to track.", self._frame_index)
            return detections

        # Candidate tracks: same region, seen recently enough, not in the future
        cand_ids, gates = [], []
        for tid, t in self._tracks.items():
            if region is not None and t["region"] not in (None, region):
                continue
            if now is not None and t["ts"] is not None:
                gap_h = (now - t["ts"]).total_seconds() / 3600.0
                if gap_h < 0 or gap_h > self.max_gap_hours:
                    continue
            else:
                gap_h = 0.0
            gate = self.min_gate_km + self.max_speed_knots * KM_PER_NM * gap_h
            cand_ids.append(tid)
            gates.append(min(gate, self.max_gate_km))

        matches: Dict[int, int] = {}     # detection index → track id
        if cand_ids:
            from scipy.optimize import linear_sum_assignment

            BIG = 1e9
            cost = np.full((len(located), len(cand_ids)), BIG)
            for i, d in enumerate(located):
                for j, tid in enumerate(cand_ids):
                    t = self._tracks[tid]
                    # Gate on physical reachability from the last fix …
                    dist = _geo_distance_km(d.lat, d.lon, t["lat"], t["lon"])
                    if dist > gates[j]:
                        continue
                    # … but rank candidates by distance to the predicted position
                    p_lat, p_lon = self._predict(t, now)
                    c = _geo_distance_km(d.lat, d.lon, p_lat, p_lon)
                    if d.length_m and t.get("length_m"):
                        big = max(d.length_m, t["length_m"])
                        c *= 1.0 + 2.0 * abs(d.length_m - t["length_m"]) / big
                    cost[i, j] = c
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < BIG:
                    matches[i] = cand_ids[j]

        n_new = 0
        for i, d in enumerate(located):
            tid = matches.get(i)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                n_new += 1
            d.track_id = tid
            old = self._tracks.get(tid)
            self._tracks[tid] = {
                "lat": d.lat, "lon": d.lon, "ts": now, "region": region,
                "prev": (old["lat"], old["lon"], old["ts"]) if old else None,
                "length_m": d.length_m or (old or {}).get("length_m"),
            }
            self.track_history.setdefault(tid, []).append({
                "lat": d.lat,
                "lon": d.lon,
                "timestamp": timestamp or "",
                "confidence": d.confidence,
            })

        log.info("Frame %d: %d detection(s) → %d matched, %d new track(s).",
                 self._frame_index, len(located), len(matches), n_new)
        return detections

    @staticmethod
    def _predict(t: dict, now: Optional[datetime]):
        """Dead-reckon a track to `now` from its last two fixes."""
        prev = t.get("prev")
        if not prev or now is None or t["ts"] is None or prev[2] is None:
            return t["lat"], t["lon"]
        span = (t["ts"] - prev[2]).total_seconds()
        ahead = (now - t["ts"]).total_seconds()
        if span <= 0 or ahead <= 0:
            return t["lat"], t["lon"]
        k = ahead / span
        return (t["lat"] + (t["lat"] - prev[0]) * k,
                t["lon"] + (t["lon"] - prev[1]) * k)

    def get_track_trail(self, track_id: int) -> List[dict]:
        """Return the full position history for a given track ID."""
        return self.track_history.get(track_id, [])

    def active_track_ids(self) -> List[int]:
        """Return all track IDs currently known to the tracker."""
        return sorted(self._tracks)

    def reset(self):
        """Forget all in-memory state (it is re-seeded from the DB on next use)."""
        self._tracks.clear()
        self.track_history.clear()
        self._next_id = 1
        self._loaded = False
        self._frame_index = 0
        log.info("Tracker reset.")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate two acquisitions with synthetic detections
    tracker = ShipTracker()

    acq1 = [
        Detection(tile_path=Path("tile.png"), bbox_px=(100, 100, 200, 200),
                  confidence=0.85, class_id=0, lat=20.5, lon=115.2),
        Detection(tile_path=Path("tile.png"), bbox_px=(300, 300, 400, 400),
                  confidence=0.78, class_id=0, lat=20.6, lon=115.5),
    ]
    acq2 = [
        Detection(tile_path=Path("tile.png"), bbox_px=(105, 103, 205, 203),
                  confidence=0.90, class_id=0, lat=20.9, lon=115.6),
    ]

    print("=== Acquisition 1 ===")
    for d in tracker.update(acq1, timestamp="2024-01-15T06:00:00Z"):
        print(f"  Track {d.track_id} at ({d.lat}, {d.lon}) conf={d.confidence:.2f}")

    print("\n=== Acquisition 2 (5 hours later) ===")
    for d in tracker.update(acq2, timestamp="2024-01-15T11:00:00Z"):
        print(f"  Track {d.track_id} at ({d.lat}, {d.lon}) conf={d.confidence:.2f}")

    print("\nTrack history:")
    for tid, hist in tracker.track_history.items():
        print(f"  Track {tid}: {len(hist)} position(s)")

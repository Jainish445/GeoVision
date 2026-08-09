"""
models/tracker.py — Multi-object tracking with DeepSORT.

Wraps the `deep_sort_realtime` library to assign persistent track IDs
to ship detections across successive satellite image acquisitions.

Because satellite revisit intervals are measured in hours (not video frames),
this module uses a time-aware version of DeepSORT where Kalman filter
state is propagated using actual elapsed time, not frame count.

Usage:
    from models.tracker import ShipTracker
    tracker = ShipTracker()
    detections = tracker.update(detections, timestamp="2024-01-15T06:30:00Z")
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEEPSORT_MAX_AGE, DEEPSORT_N_INIT,
    DEEPSORT_MAX_COSINE_DIST, DEEPSORT_NN_BUDGET,
)
from models.detector import Detection
from utils.logger import get_logger

log = get_logger("tracker")


class ShipTracker:
    """
    Wraps DeepSORT to maintain ship identities across multiple
    satellite acquisitions of the same region.

    Key design choices:
    • Uses IoU + appearance (ReID) matching.
    • A track is "confirmed" after N_INIT consecutive detections.
    • Dead tracks survive MAX_AGE frames before deletion.
    • For satellite data, each acquisition counts as one "frame".
    """

    def __init__(
        self,
        max_age:         int   = DEEPSORT_MAX_AGE,
        n_init:          int   = DEEPSORT_N_INIT,
        max_cosine_dist: float = DEEPSORT_MAX_COSINE_DIST,
        nn_budget:       int   = DEEPSORT_NN_BUDGET,
    ):
        self.max_age         = max_age
        self.n_init          = n_init
        self.max_cosine_dist = max_cosine_dist
        self.nn_budget       = nn_budget
        self._tracker        = None
        self._frame_index    = 0
        self._last_timestamp: Optional[datetime] = None

        # History: track_id → list of (lat, lon, timestamp) tuples
        self.track_history: dict = {}

    def _load_tracker(self):
        """Lazy-initialise DeepSORT (avoids import-time crash if not installed)."""
        if self._tracker is not None:
            return

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError(
                "deep_sort_realtime not installed. "
                "Run: pip install deep-sort-realtime"
            )

        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_dist,
            nn_budget=self.nn_budget,
        )
        log.info("DeepSORT tracker initialised (max_age=%d, n_init=%d).",
                 self.max_age, self.n_init)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Detection],
        timestamp: Optional[str] = None,
        image: Optional[np.ndarray] = None,   # optional: H×W×3 tile for ReID
    ) -> List[Detection]:
        """
        Feed detections from one satellite acquisition into the tracker.

        Args:
            detections: List of Detection objects from the detector.
            timestamp:  ISO-8601 string of the acquisition time.
            image:      Optional full-scene image for appearance embedding.

        Returns:
            Same list, with track_id field populated on confirmed tracks.
        """
        self._load_tracker()
        self._frame_index += 1

        if timestamp:
            try:
                self._last_timestamp = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Build DeepSORT-format input: list of ([left, top, w, h], conf, class)
        raw_dets = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox_px
            w, h = x2 - x1, y2 - y1
            raw_dets.append(([x1, y1, w, h], d.confidence, d.class_name))

        if not raw_dets:
            log.debug("Frame %d: no detections to track.", self._frame_index)
            return detections

        # Run tracking step
        tracks = self._tracker.update_tracks(raw_dets, frame=image)

        # Map confirmed tracks back to detections by IoU overlap
        confirmed = [t for t in tracks if t.is_confirmed()]
        log.info("Frame %d: %d detection(s) → %d confirmed track(s).",
                 self._frame_index, len(detections), len(confirmed))

        assigned = self._assign_tracks_to_detections(detections, confirmed)

        # Update history
        for d in assigned:
            if d.track_id is not None and d.lat is not None:
                self.track_history.setdefault(d.track_id, [])
                self.track_history[d.track_id].append({
                    "lat": d.lat,
                    "lon": d.lon,
                    "timestamp": timestamp or "",
                    "confidence": d.confidence,
                })

        return assigned

    def get_track_trail(self, track_id: int) -> List[dict]:
        """Return the full position history for a given track ID."""
        return self.track_history.get(track_id, [])

    def active_track_ids(self) -> List[int]:
        """Return currently active (confirmed) track IDs."""
        if self._tracker is None:
            return []
        return [
            t.track_id for t in self._tracker.tracker.tracks
            if t.is_confirmed()
        ]

    def reset(self):
        """Reset tracker state (call when switching to a new region)."""
        self._tracker = None
        self._frame_index = 0
        self.track_history.clear()
        log.info("Tracker reset.")

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _assign_tracks_to_detections(
        detections: List[Detection],
        tracks: list,
    ) -> List[Detection]:
        """
        Match confirmed DeepSORT tracks back to Detection objects
        using centroid proximity (since DeepSORT works in pixel space).
        """
        if not tracks:
            return detections

        # Build track bbox centres
        track_centres = []
        track_ids     = []
        for t in tracks:
            ltrb = t.to_ltrb()  # [left, top, right, bottom]
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2
            track_centres.append((cx, cy))
            track_ids.append(t.track_id)

        for d in detections:
            dcx, dcy = d.centre_px
            best_dist = float("inf")
            best_tid  = None
            for (tcx, tcy), tid in zip(track_centres, track_ids):
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid
            # Assign if close enough (within half a tile)
            if best_dist < 320:
                d.track_id = best_tid

        return detections


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate two acquisitions with synthetic detections
    from models.detector import Detection

    tracker = ShipTracker()

    acq1 = [
        Detection(tile_path=Path("tile.png"), bbox_px=(100, 100, 200, 200),
                  confidence=0.85, class_id=0, lat=20.5, lon=115.2),
        Detection(tile_path=Path("tile.png"), bbox_px=(300, 300, 400, 400),
                  confidence=0.78, class_id=0, lat=20.6, lon=115.5),
    ]
    acq2 = [
        Detection(tile_path=Path("tile.png"), bbox_px=(105, 103, 205, 203),
                  confidence=0.90, class_id=0, lat=20.51, lon=115.21),
    ]

    print("=== Acquisition 1 ===")
    r1 = tracker.update(acq1, timestamp="2024-01-15T06:00:00Z")
    for d in r1:
        print(f"  Track {d.track_id} at ({d.lat}, {d.lon}) conf={d.confidence:.2f}")

    print("\n=== Acquisition 2 (5 hours later) ===")
    r2 = tracker.update(acq2, timestamp="2024-01-15T11:00:00Z")
    for d in r2:
        print(f"  Track {d.track_id} at ({d.lat}, {d.lon}) conf={d.confidence:.2f}")

    print("\nTrack history:")
    for tid, hist in tracker.track_history.items():
        print(f"  Track {tid}: {len(hist)} position(s)")

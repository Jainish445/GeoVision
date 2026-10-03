"""
config.py — Centralized configuration for the Warship Detection & Tracking System.
All paths, credentials, hyperparameters, and constants live here.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data" / "raw"
TILES_DIR       = ROOT_DIR / "data" / "tiles"
PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models" / "weights"
RESULTS_DIR     = ROOT_DIR / "results"
DB_PATH         = ROOT_DIR / "results" / "warship_tracks.db"

for d in [DATA_DIR, TILES_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Sentinel / Copernicus Credentials ────────────────────────────────────────
COPERNICUS_USER     = os.getenv("COPERNICUS_USER", "your_username")
COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD", "your_password")

# Where imagery comes from: "copernicus" (needs the credentials above),
# "planetary" (Microsoft Planetary Computer — same Sentinel-1 GRD scenes, no
# account needed) or "auto" (Copernicus when credentials are set, else Planetary).
DATA_SOURCE       = os.getenv("WARSHIP_DATA_SOURCE", "auto")

PC_STAC_URL       = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SIGN_URL       = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
# Planetary Computer scenes are cloud-optimised, so only the part covering the
# area of interest is fetched — at most this many pixels per side (~80 km).
PC_MAX_WINDOW_PX  = 8192

CDSE_BASE_URL     = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"

# ─── AIS Integration ──────────────────────────────────────────────────────────
MARINETRAFFIC_API_KEY = os.getenv("MARINETRAFFIC_API_KEY", "")
AISHUB_USER           = os.getenv("AISHUB_USER", "")

# ─── Sentinel-1 SAR Query Parameters ─────────────────────────────────────────
SAR_PRODUCT_TYPE    = "GRD"
SAR_POLARISATION    = "VV&VH"
SAR_MAX_RESULTS     = 5

# ─── Sentinel-2 Optical Query Parameters ──────────────────────────────────────
OPT_MAX_CLOUD_COVER = 20
OPT_MAX_RESULTS     = 5

# ─── Tiling Parameters ────────────────────────────────────────────────────────
TILE_SIZE_PX   = 640
TILE_OVERLAP_PX = 64
TILE_STRIDE    = TILE_SIZE_PX - TILE_OVERLAP_PX

# ─── Detection Model ──────────────────────────────────────────────────────────
# Model size: yolov8n / s / m / l / x. "n" trains and runs at a usable speed
# on a CPU-only machine; switch to "yolov8m" (env YOLO_MODEL_SIZE) with a GPU.
YOLO_MODEL_SIZE     = os.getenv("YOLO_MODEL_SIZE", "yolov8n")
YOLO_WEIGHTS        = MODELS_DIR / f"warship_{YOLO_MODEL_SIZE}.pt"
YOLO_BASE_WEIGHTS   = MODELS_DIR / f"{YOLO_MODEL_SIZE}.pt"   # generic COCO fallback (auto-downloaded)
YOLO_CONF_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD  = 0.45
YOLO_IMGSZ          = 640
DETECT_CLASSES      = ["ship"]
# Boxes longer than this are not ships (the largest vessels afloat are ~460 m);
# in practice they are islets, breakwaters or piers.
DETECT_MAX_SHIP_M   = 500.0

# ─── Warship Classifier ───────────────────────────────────────────────────────
CLASSIFIER_WEIGHTS  = MODELS_DIR / "warship_classifier.pt"
WARSHIP_CLASSES     = [
    "Aircraft Carrier",
    "Destroyer",
    "Frigate",
    "Cruiser",
    "Submarine (surfaced)",
    "Amphibious Assault",
    "Patrol / Corvette",
    "Unknown Warship",
]
CLASSIFIER_IMG_SIZE = 224
CLASSIFIER_CONF_MIN = 0.50

# Label used when AIS (or the CNN) identifies a detection as a commercial vessel.
CIVILIAN_LABEL      = "Civilian Vessel"

# Label for a ship whose type is not supported by any evidence. Hull size
# alone never justifies a warship type: for every aircraft carrier afloat
# there are hundreds of merchant ships of the same length.
UNIDENTIFIED_LABEL  = "Unidentified Vessel"

# Ships packed this closely are commercial traffic (anchorages, port
# approaches, shipping lanes); warships in company keep kilometres apart.
# An unidentified ship with at least CROWD_MIN_NEIGHBOURS other ships within
# CROWD_RADIUS_KM is labelled a probable civilian vessel. 0 disables the rule.
CROWD_RADIUS_KM      = 3.0
CROWD_MIN_NEIGHBOURS = 6
CROWD_CONFIDENCE     = 0.6

# Classes the CNN is trained on: the warship types plus a civilian class, so
# merchant ships are not forced into a warship category.
CLASSIFIER_CLASSES  = WARSHIP_CLASSES + [CIVILIAN_LABEL]

# The CNN is trained on high-resolution optical crops (≈1–2 m/pixel). On
# coarser imagery (Sentinel is 10 m/pixel) a ship is only a few pixels long
# and the CNN's output is meaningless, so the size-based method is used instead.
CLASSIFIER_CNN_MAX_GSD_M = 4.0

# Size-based fallback used when no trained classifier weights are available.
# (class, min length m, max length m) — typical hull lengths per type.
SIZE_CLASS_BANDS = [
    ("Patrol / Corvette",  20.0, 105.0),
    ("Frigate",           105.0, 145.0),
    ("Destroyer",         145.0, 170.0),
    ("Cruiser",           170.0, 200.0),
    ("Amphibious Assault", 200.0, 265.0),
    ("Aircraft Carrier",  265.0, 360.0),
]
# Amphibious ships / carriers are beamy; cruisers & destroyers are slender.
SIZE_WIDE_BEAM_M     = 28.0
# Fallback ground sample distance when a tile has no geo sidecar (Sentinel = 10 m).
DEFAULT_GSD_M        = 10.0

# ─── Tracking (geographic nearest-neighbour association) ──────────────────────
TRACK_MAX_SPEED_KNOTS = 35.0    # fastest plausible surface ship
TRACK_MIN_GATE_KM     = 2.0     # position noise allowance when elapsed time ≈ 0
TRACK_MAX_GATE_KM     = 400.0   # never associate further than this
TRACK_MAX_GAP_HOURS   = 72.0    # tracks unseen for longer are not continued

# ─── Training Hyperparameters ─────────────────────────────────────────────────
TRAIN_EPOCHS      = 80
TRAIN_BATCH_SIZE  = 16
TRAIN_LR0         = 1e-3
TRAIN_LRF         = 1e-2
TRAIN_IMG_SIZE    = 640
TRAIN_PATIENCE    = 20
TRAIN_WORKERS     = 4
TRAIN_DEVICE      = os.getenv("TRAIN_DEVICE", "auto")   # "auto" → GPU 0 if available, else CPU
TRAIN_AUGMENT     = True

# ─── Regions of Interest (lat_min, lon_min, lat_max, lon_max) ─────────────────
REGIONS_OF_INTEREST = {
    "South_China_Sea":   (0.0,   99.0,  25.0, 125.0),
    "Persian_Gulf":      (22.0,  48.0,  30.0,  60.0),
    "Mediterranean_Sea": (30.0,   5.0,  46.0,  36.0),
    "Arabian_Sea":       (10.0,  55.0,  25.0,  75.0),
    "East_China_Sea":    (25.0, 118.0,  35.0, 130.0),
    "Black_Sea":         (40.0,  27.5,  46.5,  41.5),
    "Baltic_Sea":        (54.0,  10.0,  65.0,  30.0),
    "North_Pacific":     (30.0, 130.0,  50.0, 180.0),
    "North_Atlantic":    (25.0, -80.0,  60.0,   0.0),
    "Red_Sea":           (12.0,  32.0,  30.0,  43.5),
}

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_HOST     = "0.0.0.0"
DASHBOARD_PORT     = 8501
MAP_DEFAULT_CENTER = [20.0, 0.0]
MAP_DEFAULT_ZOOM   = 2

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = ROOT_DIR / "results" / "pipeline.log"

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
YOLO_MODEL_SIZE     = "yolov8m"
YOLO_WEIGHTS        = MODELS_DIR / "warship_yolov8m.pt"
YOLO_CONF_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD  = 0.45
YOLO_IMGSZ          = 640
DETECT_CLASSES      = ["ship"]

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

# ─── DeepSORT Tracking ────────────────────────────────────────────────────────
DEEPSORT_MAX_AGE         = 5
DEEPSORT_N_INIT          = 2
DEEPSORT_MAX_IOU_DIST    = 0.7
DEEPSORT_MAX_COSINE_DIST = 0.4
DEEPSORT_NN_BUDGET       = 100

# ─── Training Hyperparameters ─────────────────────────────────────────────────
TRAIN_EPOCHS      = 80
TRAIN_BATCH_SIZE  = 16
TRAIN_LR0         = 1e-3
TRAIN_LRF         = 1e-2
TRAIN_IMG_SIZE    = 640
TRAIN_PATIENCE    = 20
TRAIN_WORKERS     = 4
TRAIN_DEVICE      = "0"
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

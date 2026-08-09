# 🛰️ GeoVision — Warship Detection & Tracking System

A production-grade satellite image processing pipeline for detecting and tracking
major surface warships in near-real-time using Sentinel-1 SAR and Sentinel-2 optical
imagery, YOLOv8 detection, DeepSORT multi-object tracking, and AIS data fusion.

---

## 📂 Project Structure

```
warship-tracker/
├── config.py               ← All constants, paths, hyperparams
├── pipeline.py             ← End-to-end orchestration (main entry point)
├── dashboard.py            ← Streamlit visualization dashboard
├── requirements.txt
│
├── data/
│   ├── ingestion.py        ← Copernicus / Sentinel data download
│   └── preprocessing.py    ← Band reading, cloud masking, tiling
│
├── models/
│   ├── detector.py         ← YOLOv8 ship detection
│   ├── tracker.py          ← DeepSORT multi-object tracking
│   └── classifier.py       ← ResNet-50 warship type classifier
│
├── geo/
│   ├── ais_fusion.py       ← AIS vessel data enrichment
│   └── geodb.py            ← SQLite persistence layer
│
├── train/
│   └── finetune.py         ← YOLOv8 fine-tuning & dataset preparation
│
└── utils/
    └── logger.py           ← Unified logging
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

> **Note on rasterio:** On Linux, install GDAL first:
> `sudo apt-get install libgdal-dev` then `pip install rasterio`
>
> On macOS: `brew install gdal` then `pip install rasterio`

### 2. Set credentials

```bash
# Copernicus Data Space (free at https://dataspace.copernicus.eu)
export COPERNICUS_USER="your_username"
export COPERNICUS_PASSWORD="your_password"

# Optional AIS enrichment
export MARINETRAFFIC_API_KEY="your_key"   # https://www.marinetraffic.com/en/ais-api-services
# OR
export AISHUB_USER="your_username"        # https://www.aishub.net
```

---

## 🚀 Quick Start

### Demo mode (no credentials needed)

```bash
python pipeline.py --demo
streamlit run dashboard.py
```

This loads synthetic ship detections into the DB and launches the dashboard.

### Real data — single region

```bash
# Sentinel-1 SAR (cloud-independent, best for ocean monitoring)
python pipeline.py --region South_China_Sea --sensor S1 --days 7

# Sentinel-2 optical (higher resolution, cloud-dependent)
python pipeline.py --region Persian_Gulf --sensor S2 --days 5
```

### All regions (full world scan)

```bash
python pipeline.py --all --sensor S1 --days 7
```

### Process a locally downloaded product (skip download)

```bash
python pipeline.py --local-dir data/raw/S1A_IW_GRDH_1SDV_20240115.SAFE --sensor S1
```

### Launch dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## 🧠 Model Training

### Prepare SAR-Ship dataset

Download the SAR-Ship dataset from:
https://github.com/CAESAR-Radi/SAR-Ship-Dataset

```bash
python train/finetune.py prepare \
    --raw-dir /path/to/SAR-Ship-Dataset \
    --output-dir data/dataset
```

### Fine-tune YOLOv8

```bash
python train/finetune.py train \
    --data data/dataset/data.yaml \
    --epochs 80 \
    --device 0          # GPU 0, or "cpu"
```

### Evaluate the model

```bash
python train/finetune.py eval \
    --data data/dataset/data.yaml \
    --weights models/weights/warship_yolov8m.pt
```

### Export to ONNX (for deployment)

```bash
python train/finetune.py export \
    --export onnx \
    --weights models/weights/warship_yolov8m.pt
```

---

## 🗂️ Recommended Datasets

| Dataset | Type | Classes | Size | Link |
|---|---|---|---|---|
| SAR-Ship | Sentinel-1 SAR | ship | 43,000+ chips | [GitHub](https://github.com/CAESAR-Radi/SAR-Ship-Dataset) |
| DOTA v1.5 | Aerial optical | 16 (incl. ship) | 11,268 images | [Website](https://captain-whu.github.io/DOTA/) |
| HRSC2016 | Optical | ship subtypes | 1,061 images | [GitHub](https://www.kaggle.com/datasets/guofeng/hrsc2016) |
| xView | Satellite | 60 classes | 1,413 km² | [Website](http://xviewdataset.org/) |
| Roboflow Maritime | Mixed | ships | various | [Roboflow](https://universe.roboflow.com) |

---

## 🗺️ Dashboard Features

| Tab | Content |
|---|---|
| **Map** | Interactive dark Folium map with ship markers, colour-coded by warship type. Click any marker for details. |
| **Detections** | Sortable/filterable table of all detection events. |
| **Tracks** | Per-track summary with full position trail. |
| **Export** | One-click GeoJSON and CSV download. |

**Sidebar controls:**
- Region filter
- Time window (last N days)
- Toggle track trails
- Warship type multi-select
- Confidence threshold slider
- Load Demo Data button

---

## 🔗 Pipeline Architecture

```
Copernicus CDSE API (Sentinel-1/2)
         │
         ▼
  SentinelIngester          ← OAuth2 token + OData query + ZIP download
         │
         ▼
   Preprocessor             ← Band read → cloud mask → tile (640×640 PNG + sidecar JSON)
         │
         ▼
   ShipDetector             ← YOLOv8m inference → NMS across tiles → Detection objects
         │
         ▼
   ShipTracker              ← DeepSORT → persistent track IDs across acquisitions
         │
         ▼
 WarshipClassifier          ← ResNet-50 crop classifier → warship type + confidence
         │
         ▼
    AISFusion               ← MarineTraffic / AISHub → vessel name, MMSI, flag
         │
         ▼
      GeoDB                 ← SQLite (detections, tracks, acquisitions tables)
         │
         ▼
   dashboard.py             ← Streamlit + Folium map visualization
```

---

## 📡 Supported Regions

| Region | Coverage |
|---|---|
| South China Sea | 0°–25°N, 99°–125°E |
| Persian Gulf | 22°–30°N, 48°–60°E |
| Mediterranean Sea | 30°–46°N, 5°–36°E |
| Arabian Sea | 10°–25°N, 55°–75°E |
| East China Sea | 25°–35°N, 118°–130°E |
| Black Sea | 40°–46.5°N, 27.5°–41.5°E |
| Baltic Sea | 54°–65°N, 10°–30°E |
| North Pacific | 30°–50°N, 130°–180°E |
| North Atlantic | 25°–60°N, 80°W–0° |
| Red Sea | 12°–30°N, 32°–43.5°E |

Custom regions can be added to `REGIONS_OF_INTEREST` in `config.py`.

---

## ⚠️ Known Limitations

| Issue | Mitigation |
|---|---|
| Cloud cover blocking optical | Use Sentinel-1 SAR as primary sensor |
| 10m/pixel free imagery resolution | Use commercial Planet/Maxar for higher res |
| ~5 day Sentinel revisit gap | Use multiple sensors and ROIs |
| False positives from oil rigs | Fine-tune on labeled maritime data |
| AIS spoofing by military vessels | Cross-reference visual + AIS |
| No live streaming | Cron-schedule `pipeline.py` every 6 hours |

---

## 📅 7-Week Milestone Plan

| Week | Goal |
|---|---|
| 1 | Set up ingestion pipeline, authenticate with Copernicus, download first product |
| 2 | Preprocessing pipeline: tiling, cloud masking, verify tile quality |
| 3 | Fine-tune YOLOv8 on SAR-Ship dataset, achieve mAP > 0.70 |
| 4 | Integrate DeepSORT tracker, validate track persistence across acquisitions |
| 5 | Add warship classifier (ResNet-50), add AIS fusion |
| 6 | Build Streamlit dashboard with Folium map, GeoJSON export |
| 7 | System integration testing, documentation, portfolio write-up |

---

## 📄 License

MIT — for academic and research use.

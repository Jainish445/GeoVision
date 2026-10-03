# 🛰️ GeoIntel — Warship Detection & Tracking System

A production-grade satellite image processing pipeline for detecting and tracking
major surface warships in near-real-time using Sentinel-1 SAR and Sentinel-2 optical
imagery, YOLOv8 detection, geographic multi-acquisition tracking, warship type
classification, and AIS data fusion.

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
│   ├── preprocessing.py    ← Band reading, cloud masking, georeferencing, tiling
│   └── dataset.py          ← Training-dataset preparation & statistics
│
├── models/
│   ├── detector.py         ← YOLOv8 ship detection
│   ├── tracker.py          ← Geographic track association across acquisitions
│   └── classifier.py       ← Warship type classifier (ResNet-50 / hull-size fallback)
│
├── geo/
│   ├── ais_fusion.py       ← AIS vessel data enrichment
│   └── geodb.py            ← SQLite persistence layer
│
├── train/
│   └── finetune.py         ← YOLOv8 fine-tuning & dataset preparation
│
├── tests/                  ← pytest suite (no network, GPU or weights needed)
│
└── utils/
    ├── geo_utils.py        ← Distance / bearing / bbox helpers
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

On Windows PowerShell use `$env:COPERNICUS_USER = "your_username"` etc.

**No account? It still works.** When no Copernicus credentials are set, the
pipeline fetches the same Sentinel-1 GRD scenes from Microsoft's public
[Planetary Computer](https://planetarycomputer.microsoft.com) archive, which
needs no sign-up. Only the part of the scene covering your area is downloaded
(at most ~80 × 80 km), so a run takes a couple of minutes instead of a 1.7 GB
download. Force a source with `--source copernicus` or `--source planetary`.
Sentinel-2 (optical) is only available through Copernicus.

### 3. Run the tests

```bash
python -m pytest tests -q
```

---

## 🚀 Quick Start

### Demo mode (no credentials needed)

```bash
python pipeline.py --demo
streamlit run dashboard.py
```

This simulates three satellite passes (6 h apart, ending now) over seven ships.
A synthetic SAR tile and a synthetic AIS feed are generated for every pass, and
the classifier, AIS fusion, tracker, database and dashboard all run for real on
them — only the download and YOLO detection steps are skipped. The ships show
each way a label comes about: three warships broadcasting military AIS (typed
from AIS + hull size), two warships running dark (left **Unidentified**, with a
size class), and two merchant ships (**Civilian Vessel**). Re-running replaces
the previous demo data.

### Real data — single region

```bash
# Sentinel-1 SAR (cloud-independent, best for ocean monitoring)
python pipeline.py --region South_China_Sea --sensor S1 --days 7

# Sentinel-2 optical (higher resolution, cloud-dependent)
python pipeline.py --region Persian_Gulf --sensor S2 --days 5
```

### Any custom area

```bash
# lat_min lon_min lat_max lon_max — e.g. the Singapore Strait
python pipeline.py --bbox 1.05 103.60 1.35 104.10 --local-region Singapore_Strait --days 30
```

### All regions (full world scan)

```bash
python pipeline.py --all --sensor S1 --days 7
```

### Process a locally downloaded product (skip download)

```bash
python pipeline.py --local-dir data/raw/S1A_IW_GRDH_1SDV_20240115.SAFE --sensor S1
```

`--local-dir` also accepts a single GeoTIFF. Add `--local-region NAME` to label
the results, `--no-clf` / `--no-ais` to skip stages, `--no-geojson` to skip the
GeoJSON export.

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
    --raw-dir data/SAR-Ship-Dataset \
    --output-dir data/dataset
```

The dataset's own layout (`<name>.jpg` + YOLO `<name>.txt` in one folder) is
read directly; PASCAL VOC (`JPEGImages/` + `Annotations/*.xml`) works too.

### Fine-tune YOLOv8

```bash
python train/finetune.py train \
    --data data/dataset/data.yaml \
    --epochs 80 \
    --device 0          # GPU 0, or "cpu" (default: GPU if available)
```

The pretrained base weights are downloaded to `models/weights/` on first use.

**CPU-only machines:** the default model is the small `yolov8n` (set in
`config.py`), which trains and runs at a usable speed without a GPU. Train at
the chips' native 256 px on part of the data:

```bash
python train/finetune.py train --data data/dataset/data.yaml \
    --imgsz 256 --fraction 0.3 --epochs 6 --device cpu
```

With a GPU, set the environment variable `YOLO_MODEL_SIZE=yolov8m` and train on
everything for the full 80 epochs. Fine-tuned weights are saved as
`models/weights/warship_<model>.pt` and picked up automatically.

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

## 🚢 Warship Classification

A label is only as strong as the evidence behind it. Each detection gets a
`warship_type`, the `type_source` it rests on, and (separately) a `size_class`:

| Evidence (`type_source`) | Label | When |
|---|---|---|
| `cnn` — ResNet-50 on the image crop | one of `WARSHIP_CLASSES` or **Civilian Vessel** | trained weights exist **and** the imagery is sharper than 4 m/pixel |
| `ais+size` — AIS says military, hull size says which kind | the size class, e.g. **Destroyer** | a military AIS broadcast matches the detection |
| `ais` — AIS says commercial | **Civilian Vessel** | a cargo / tanker / passenger / fishing AIS broadcast matches |
| `context` — crowded commercial waters | **Civilian Vessel** (probable, conf 0.6) | still unidentified and ≥ 6 other ships within 3 km (anchorage, port approach, shipping lane) |
| `size` — nothing but hull dimensions | **Unidentified Vessel** | everything else |

**Hull size is never treated as identification.** On Sentinel's 10 m pixels the
only thing measurable is length and beam, and for every aircraft carrier afloat
there are hundreds of merchant ships of the same length. So the hull is measured
(`length_m`, `beam_m`) and given a `size_class` — "Destroyer-sized" — which is
shown in the dashboard and used by the tracker, but the ship stays
**Unidentified Vessel** until AIS or the CNN says more. On a real scene of the
Singapore Strait this turns 306 detections into 234 probable civilian and 72
unidentified vessels, instead of 89 "aircraft carriers".

An unidentified vessel with no AIS match is exactly what a "dark" warship looks
like — and also what a merchant ship looks like when you have no AIS feed. To
get warship types on real Sentinel data you need AIS (`MARINETRAFFIC_API_KEY` or
`AISHUB_USER`, or `AISFusion(backend="file", ...)` with your own AIS log).

Size bands are in `SIZE_CLASS_BANDS`, the crowd rule in `CROWD_RADIUS_KM` /
`CROWD_MIN_NEIGHBOURS` (set the latter to 0 to switch it off), both in `config.py`.

**Why the CNN is not used on Sentinel imagery** — it is trained on
high-resolution optical crops (~1–2 m/pixel) where deck layout and
superstructure are visible. At 10 m/pixel a destroyer is ~15 × 2 pixels of
radar return; there is nothing for a CNN to recognise, so hull size is the
only honest cue. The CNN takes over automatically on sharper imagery
(commercial satellites, aerial photos), set via `CLASSIFIER_CNN_MAX_GSD_M`.

### Train the CNN

[FGSC-23](https://huggingface.co/datasets/jbourcier/fgsc23) (4,080 real ship
crops in 23 fine-grained classes, research use only) maps cleanly onto this
project's classes:

```bash
# 1. Download FGSC-23.zip from the link above and unzip into data/classifier_raw/
# 2. Arrange it into one folder per class (keeps the dataset's train/test split)
python data/dataset.py prepare-fgsc --raw-dir data/classifier_raw --output-dir data/classifier

# 3. Train, then measure accuracy on the untouched test split
python models/classifier.py train --data data/classifier/train --epochs 12
python models/classifier.py eval  --data data/classifier/test

# Classify one crop (prints the method used and the measured hull size):
python models/classifier.py predict crop.jpg --gsd 1
```

The CNN has a ninth class, **Civilian Vessel**, so merchant ships are not
forced into a warship category. Your own data works too: one folder of images
per class (names are matched loosely, e.g. `patrol_corvette`).

---

## 🛰️ Tracking

Revisits are hours apart and every detection box is relative to its own tile,
so video trackers that match pixel boxes between frames don't apply.
`models/tracker.py` associates detections with tracks in lat/lon instead:

- a detection can only continue a track it could physically have reached
  (`TRACK_MAX_SPEED_KNOTS` × elapsed time, capped by `TRACK_MAX_GATE_KM`);
- candidates are ranked by distance to the track's dead-reckoned position and
  by hull-length agreement, and assigned optimally (Hungarian algorithm);
- tracks unseen for `TRACK_MAX_GAP_HOURS` are not continued;
- track IDs are seeded from the database, so they persist across runs.

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
- Vessel type multi-select
- Confidence threshold slider
- Load Demo Data button

---

## 🔗 Pipeline Architecture

```
Copernicus CDSE API (Sentinel-1/2)   or   Planetary Computer (Sentinel-1, no account)
         │
         ▼
  SentinelIngester / PlanetaryComputerIngester   ← query + download
         │
         ▼
   Preprocessor             ← Band read → cloud mask → tile (640×640 PNG + sidecar JSON)
         │
         ▼
   ShipDetector             ← YOLOv8 inference → NMS across tiles → Detection objects
         │
         ▼
 WarshipClassifier          ← ResNet-50 or hull-size fallback → warship type + confidence
         │
         ▼
   ShipTracker              ← Geographic association → persistent track IDs
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
| No AIS feed configured → no warship types on Sentinel data | Set a MarineTraffic / AISHub key, or supply an AIS file |
| Crowd rule can mislabel warships moored together in port | Tune or disable `CROWD_MIN_NEIGHBOURS` |
| No live streaming | Cron-schedule `pipeline.py` every 6 hours |
| Type recognition impossible at 10 m/pixel | Ships stay "Unidentified" with a size class unless AIS identifies them; CNN on sharper imagery |
| Generic COCO weights find nothing in SAR imagery | Run `train/finetune.py` on SAR-Ship first (weights are not stored in git) |
| Bright land / harbour structures can be detected as ships | Raise `--conf`, or restrict `--bbox` to open water |
| Tracking across multi-day gaps is ambiguous | Gate limits + AIS identity; treat long-gap associations as tentative |

---

## 📅 7-Week Milestone Plan

| Week | Goal |
|---|---|
| 1 | Set up ingestion pipeline, authenticate with Copernicus, download first product |
| 2 | Preprocessing pipeline: tiling, cloud masking, verify tile quality |
| 3 | Fine-tune YOLOv8 on SAR-Ship dataset, achieve mAP > 0.70 |
| 4 | Integrate tracker, validate track persistence across acquisitions |
| 5 | Add warship classifier (ResNet-50), add AIS fusion |
| 6 | Build Streamlit dashboard with Folium map, GeoJSON export |
| 7 | System integration testing, documentation, portfolio write-up |

---

## 📄 License

MIT — for academic and research use.

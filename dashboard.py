"""
dashboard.py — Streamlit dashboard for the Warship Detection & Tracking System.

Displays:
  • Live world map with all detected ship positions (colour-coded by type)
  • Track trails showing vessel movement over time
  • Per-region detection statistics
  • Filterable data table
  • GeoJSON export button

Run with:
    streamlit run dashboard.py
"""

import sys
import json
import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config must come before any other st call ────────────────────────────
try:
    import streamlit as st
    st.set_page_config(
        page_title="GeoIntel — Warship Tracker",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

import pandas as pd

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from config import WARSHIP_CLASSES, REGIONS_OF_INTEREST, MAP_DEFAULT_CENTER
from geo.geodb import GeoDB


# ── Colour scheme for warship types ───────────────────────────────────────────
WARSHIP_COLORS = {
    "Aircraft Carrier":       "#e74c3c",
    "Destroyer":              "#e67e22",
    "Frigate":                "#f39c12",
    "Cruiser":                "#d35400",
    "Submarine (surfaced)":   "#8e44ad",
    "Amphibious Assault":     "#2980b9",
    "Patrol / Corvette":      "#27ae60",
    "Unknown Warship":        "#7f8c8d",
}

WARSHIP_ICONS = {
    "Aircraft Carrier":       "⬡",
    "Destroyer":              "▲",
    "Frigate":                "●",
    "Cruiser":                "★",
    "Submarine (surfaced)":   "◆",
    "Amphibious Assault":     "■",
    "Patrol / Corvette":      "▸",
    "Unknown Warship":        "?",
}


# ── Cached DB access ───────────────────────────────────────────────────────────

@st.cache_resource
def get_db() -> GeoDB:
    return GeoDB()


@st.cache_data(ttl=60)
def load_detections(region: Optional[str], since: Optional[str]) -> pd.DataFrame:
    db = get_db()
    rows = db.query_detections(region=region or None, since=since, limit=5000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lat", "lon"])
    return df


@st.cache_data(ttl=60)
def load_tracks(region: Optional[str]) -> List[dict]:
    db = get_db()
    return db.query_tracks(region=region or None)


@st.cache_data(ttl=120)
def load_summary() -> dict:
    return get_db().summary()


# ── CSS ────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }
    .stApp {
        background: #0a0f1e;
        color: #c9d1d9;
    }
    .metric-card {
        background: linear-gradient(135deg, #141b2d 0%, #1a2332 100%);
        border: 1px solid #253048;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.4rem;
        color: #58a6ff;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #3fb950;
        box-shadow: 0 0 6px #3fb950;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; }
        50%      { opacity: 0.4; }
    }
    .sidebar-header {
        font-family: 'Share Tech Mono', monospace;
        color: #58a6ff;
        font-size: 1.1rem;
        border-bottom: 1px solid #253048;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Map building ───────────────────────────────────────────────────────────────

def build_map(df: pd.DataFrame, tracks: List[dict], show_trails: bool) -> "folium.Map":
    """Build a dark-themed Folium map with ship markers and track trails."""

    m = folium.Map(
        location=MAP_DEFAULT_CENTER,
        zoom_start=3,
        tiles=None,
        prefer_canvas=True,
    )

    # Dark tile layer
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="CartoDB Dark",
        max_zoom=19,
    ).add_to(m)

    # Track trails
    if show_trails:
        for track in tracks:
            trail = track.get("trail", [])
            if len(trail) < 2:
                continue
            coords = [[p["lat"], p["lon"]] for p in trail if "lat" in p and "lon" in p]
            if len(coords) >= 2:
                wtype = track.get("warship_type", "Unknown Warship") or "Unknown Warship"
                color = WARSHIP_COLORS.get(wtype, "#7f8c8d")
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Track {track['track_id']} — {wtype}",
                ).add_to(m)

    # Ship markers
    for _, row in df.iterrows():
        wtype = row.get("warship_type") or "Unknown Warship"
        color = WARSHIP_COLORS.get(wtype, "#7f8c8d")
        conf  = float(row.get("confidence", 0))

        popup_html = f"""
        <div style='font-family:monospace;font-size:12px;min-width:200px;'>
          <b style='color:{color};font-size:14px;'>{wtype}</b><br>
          <hr style='border-color:#333;margin:4px 0;'>
          Track ID:    <b>{row.get('track_id','—')}</b><br>
          Confidence:  <b>{conf:.1%}</b><br>
          Lat / Lon:   <b>{row['lat']:.4f}, {row['lon']:.4f}</b><br>
          Vessel:      <b>{row.get('vessel_name') or '—'}</b><br>
          MMSI:        <b>{row.get('vessel_mmsi') or '—'}</b><br>
          Flag:        <b>{row.get('vessel_flag') or '—'}</b><br>
          Time:        <b>{(row.get('timestamp') or '')[:19]}</b>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            tooltip=f"{wtype} ({conf:.0%})",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    return m


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-header">🛰️ GeoIntel Controls</div>',
                        unsafe_allow_html=True)

    region = st.sidebar.selectbox(
        "Region of Interest",
        ["All Regions"] + list(REGIONS_OF_INTEREST.keys()),
        help="Filter map and table to a specific maritime region.",
    )

    days = st.sidebar.slider("Show last N days", 1, 30, 7)
    since = (datetime.datetime.utcnow()
             - datetime.timedelta(days=days)).isoformat()

    show_trails = st.sidebar.toggle("Show Track Trails", value=True)

    st.sidebar.markdown("---")
    wtype_filter = st.sidebar.multiselect(
        "Warship Types",
        options=WARSHIP_CLASSES,
        default=WARSHIP_CLASSES,
    )

    conf_min = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.40, 0.05)

    st.sidebar.markdown("---")
    run_demo = st.sidebar.button("▶ Load Demo Data", type="primary",
                                 help="Insert synthetic detections into the DB")
    if run_demo:
        with st.spinner("Loading demo data …"):
            from pipeline import WarshipPipeline
            pipe = WarshipPipeline(enable_ais=False, enable_classifier=False)
            pipe.run_demo()
        st.success("Demo data loaded! Refresh the page.")
        st.cache_data.clear()

    return {
        "region":       None if region == "All Regions" else region,
        "since":        since,
        "show_trails":  show_trails,
        "wtype_filter": wtype_filter,
        "conf_min":     conf_min,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Header
    col_title, col_status = st.columns([5, 1])
    with col_title:
        st.markdown("## 🛰️ GeoIntel — Warship Detection & Tracking")
    with col_status:
        st.markdown(
            '<div style="text-align:right;padding-top:12px;">'
            '<span class="status-dot"></span>'
            '<span style="font-size:0.8rem;color:#3fb950;">LIVE</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    filters = render_sidebar()

    # ── Load data
    df     = load_detections(filters["region"], filters["since"])
    tracks = load_tracks(filters["region"])
    summary = load_summary()

    # Apply filters
    if not df.empty:
        df = df[df["confidence"] >= filters["conf_min"]]
        if filters["wtype_filter"]:
            df = df[df["warship_type"].isin(filters["wtype_filter"])
                    | df["warship_type"].isna()]

    # ── KPI cards
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, val, label):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    kpi(k1, summary.get("detections", 0), "Total Detections")
    kpi(k2, summary.get("tracks", 0),     "Unique Tracks")
    kpi(k3, summary.get("acquisitions", 0), "Acquisitions")
    kpi(k4, len(df) if not df.empty else 0, "Filtered Ships")
    st.markdown("---")

    # ── Map + table tabs
    tab_map, tab_table, tab_tracks, tab_export = st.tabs(
        ["🗺️ Map", "📋 Detections", "🔍 Tracks", "⬇️ Export"]
    )

    with tab_map:
        if not HAS_FOLIUM:
            st.warning("Install folium & streamlit-folium for the map view:\n"
                       "`pip install folium streamlit-folium`")
        elif df.empty:
            st.info("No detections match the current filters. "
                    "Try clicking **Load Demo Data** in the sidebar.")
        else:
            fmap = build_map(df, tracks, filters["show_trails"])
            st_folium(fmap, use_container_width=True, height=560)

            # Legend
            st.markdown("**Legend:**")
            cols = st.columns(len(WARSHIP_COLORS))
            for col, (wtype, color) in zip(cols, WARSHIP_COLORS.items()):
                col.markdown(
                    f'<span style="color:{color};font-size:1.2rem;">●</span> '
                    f'<span style="font-size:0.75rem;">{wtype}</span>',
                    unsafe_allow_html=True,
                )

    with tab_table:
        if df.empty:
            st.info("No detections to display.")
        else:
            display_cols = [
                "timestamp", "region", "track_id", "warship_type",
                "confidence", "lat", "lon", "vessel_name", "vessel_flag",
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].sort_values("timestamp", ascending=False),
                use_container_width=True,
                height=420,
            )
            st.caption(f"{len(df)} row(s) shown.")

    with tab_tracks:
        if not tracks:
            st.info("No tracks found.")
        else:
            track_df = pd.DataFrame([
                {
                    "Track ID":     t["track_id"],
                    "Warship Type": t.get("warship_type") or "Unknown",
                    "Vessel Name":  t.get("vessel_name") or "—",
                    "Flag":         t.get("vessel_flag") or "—",
                    "MMSI":         t.get("vessel_mmsi") or "—",
                    "First Seen":   (t.get("first_seen") or "")[:19],
                    "Last Seen":    (t.get("last_seen") or "")[:19],
                    "# Detections": t.get("n_detections", 0),
                    "Trail Points": len(t.get("trail", [])),
                }
                for t in tracks
            ])
            st.dataframe(track_df, use_container_width=True)

            # Show trail for selected track
            if not track_df.empty:
                selected_tid = st.selectbox(
                    "View trail for track",
                    track_df["Track ID"].tolist(),
                )
                selected_track = next(
                    (t for t in tracks if t["track_id"] == selected_tid), None
                )
                if selected_track and selected_track.get("trail"):
                    trail_df = pd.DataFrame(selected_track["trail"])
                    st.dataframe(trail_df, use_container_width=True)

    with tab_export:
        st.subheader("Export Data")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**GeoJSON Export**")
            if not df.empty:
                features = []
                for _, row in df.iterrows():
                    if row["lat"] and row["lon"]:
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point",
                                         "coordinates": [row["lon"], row["lat"]]},
                            "properties": {
                                k: (v if not pd.isna(v) else None)
                                for k, v in row.items()
                                if k not in ("lat", "lon")
                            },
                        })
                geojson_str = json.dumps({
                    "type": "FeatureCollection",
                    "features": features,
                }, indent=2)
                st.download_button(
                    "⬇️ Download GeoJSON",
                    data=geojson_str,
                    file_name="warship_detections.geojson",
                    mime="application/geo+json",
                )

        with c2:
            st.markdown("**CSV Export**")
            if not df.empty:
                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="warship_detections.csv",
                    mime="text/csv",
                )

        st.markdown("---")
        st.markdown("**Pipeline Quick Launch**")
        st.code("""
# Run detection on South China Sea (SAR):
python pipeline.py --region South_China_Sea --sensor S1

# Run in demo mode (no credentials needed):
python pipeline.py --demo

# Process all regions:
python pipeline.py --all --sensor S1 --days 7

# Fine-tune the YOLOv8 model:
python train/finetune.py train --data data/dataset/data.yaml

# Launch this dashboard:
streamlit run dashboard.py
        """, language="bash")


if __name__ == "__main__":
    main()

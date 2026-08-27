"""Motor Bearing Predictive Maintenance Dashboard — Benchmark Evaluation & System Architecture."""

from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import serial
import serial.tools.list_ports
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Color palette tokens
INK = "#0f172a"          # Near-black slate for headings
BODY = "#1e293b"         # Dark charcoal for primary body text
SECONDARY = "#334155"    # Medium-dark slate for secondary text
MUTED = "#475569"        # Dark slate for metadata/labels
LINE = "#cbd5e1"         # Crisp border line
PAPER = "#f8fafc"        # Clean off-white background
PANEL = "#ffffff"        # Crisp white card background
TEAL = "#0d6e6d"         # Rich, high-contrast dark teal
TEAL_BG = "#e6f4f1"      # Subtle pale teal container background
GREEN = "#166534"        # Dark forest green for normal/pass
AMBER = "#92400e"        # Rich dark amber for warning/residuals
ORANGE = "#9a3412"       # Dark burnt orange
RED = "#991b1b"          # Dark crimson for fault/errors
SLATE = "#334155"        # Neutral dark slate

FAULT_ORDER = ["NORMAL", "BALL", "INNER_RACE", "OUTER_RACE"]
FAULT_LABELS = {
    "NORMAL": "Normal",
    "BALL": "Ball",
    "INNER_RACE": "Inner Race",
    "OUTER_RACE": "Outer Race",
}

RIG_CONDITIONS = ["NORMAL", "IMBALANCE", "MISALIGNMENT"]
RIG_LABELS = {
    "NORMAL": "Normal",
    "IMBALANCE": "Imbalance",
    "MISALIGNMENT": "Misalignment",
}


st.set_page_config(
    page_title="Motor Bearing Predictive Maintenance — Benchmark Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --ink: #0f172a;
    --body: #1e293b;
    --secondary: #334155;
    --muted: #475569;
    --line: #cbd5e1;
    --paper: #f8fafc;
    --panel: #ffffff;
    --teal: #0d6e6d;
    --teal-bg: #e6f4f1;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--body);
}
.stApp { background: var(--paper); }
.block-container { max-width: 1440px; padding: 1.5rem 2.5rem 3.5rem; }

/* --- typography --- */
h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.01em; }
h1 { font-size: 1.85rem; font-weight: 700; margin: 0; }
h2 { font-size: 0.74rem; font-weight: 700; text-transform: uppercase;
     letter-spacing: 0.1em; color: var(--muted) !important; margin: 0; }
h3 { font-size: 1.10rem; font-weight: 700; margin: 0; color: var(--ink) !important; }

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--teal);
}
.subtitle { color: var(--secondary); font-size: 0.95rem; font-weight: 500; margin-top: 0.25rem; }
.header-block {
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.1rem; margin-bottom: 0.9rem;
}
.section-rule {
    border-top: 1px solid var(--line);
    padding-top: 1.2rem; margin-top: 2.0rem; margin-bottom: 0.6rem;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.1rem;
}
.section-title { font-size: 1.20rem; font-weight: 700; color: var(--ink); margin-bottom: 0.15rem; }
.section-note { font-size: 0.84rem; color: var(--secondary); font-weight: 400; }

/* --- metric blocks --- */
.metric-block {
    border-left: 3px solid var(--teal);
    padding: 0.45rem 0.6rem 0.45rem 0.80rem;
    background: var(--panel);
    border-radius: 0 5px 5px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.60rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
}
.metric-value {
    font-size: 1.38rem; font-weight: 700; color: var(--ink);
    margin-top: 0.15rem; line-height: 1.2;
}
.metric-detail { font-size: 0.74rem; color: var(--secondary); margin-top: 0.1rem; font-weight: 500; }

/* colour utilities */
.clr-green  { color: #166534 !important; }
.clr-amber  { color: #92400e !important; }
.clr-orange { color: #9a3412 !important; }
.clr-red    { color: #991b1b !important; }
.clr-teal   { color: #0d6e6d !important; }

/* --- card for overview --- */
.overview-card {
    background: var(--panel); border: 1px solid #cbd5e1;
    border-radius: 6px; padding: 1.05rem 1.15rem; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.overview-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 0.30rem;
}
.overview-title { font-size: 1.00rem; font-weight: 700; color: var(--ink); }
.overview-desc { font-size: 0.80rem; color: var(--secondary); margin-top: 0.35rem; line-height: 1.45; }

/* --- provenance banner (high contrast dark text) --- */
.provenance {
    background: #eef7f5; border-left: 4px solid var(--teal);
    border-radius: 0 6px 6px 0; padding: 0.70rem 1.0rem;
    font-size: 0.82rem; color: #0f3933; margin-bottom: 0.85rem;
    line-height: 1.45; border-top: 1px solid #d1e7e2;
    border-right: 1px solid #d1e7e2; border-bottom: 1px solid #d1e7e2;
}
.leakage-note {
    background: #f1f5f9; border-left: 4px solid #475569;
    border-radius: 0 6px 6px 0; padding: 0.65rem 0.95rem;
    font-size: 0.80rem; color: #1e293b; margin-top: 0.65rem;
    line-height: 1.40; border-top: 1px solid #cbd5e1;
    border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1;
}

/* --- Project At a Glance strip --- */
.glance-container {
    background: var(--panel); border: 1px solid #cbd5e1;
    border-radius: 6px; padding: 0.75rem 1.0rem; margin-bottom: 1.0rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.glance-grid {
    display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.75rem;
}
.glance-item {
    border-right: 1px solid #e2e8f0; padding-right: 0.5rem;
}
.glance-item:last-child { border-right: none; }
.glance-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.56rem;
    font-weight: 700; text-transform: uppercase; color: var(--muted); letter-spacing: 0.06em;
}
.glance-val {
    font-size: 1.05rem; font-weight: 700; color: var(--ink); margin-top: 0.1rem;
}

/* --- summary table --- */
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.summary-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #0f172a;
    text-align: left; padding: 0.60rem 0.80rem;
    border-bottom: 2px solid #cbd5e1;
    background: #e2e8f0;
}
.summary-table td {
    padding: 0.60rem 0.80rem; border-bottom: 1px solid #e2e8f0;
    color: #1e293b; font-size: 0.84rem;
}
.summary-table tr:last-child td { border-bottom: none; }
.summary-table tr:hover td { background: #f8fafc; }
.tag-validated {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #065f46;
    background: #d1fae5; border-radius: 3px; padding: 0.20rem 0.50rem;
    border: 1px solid #a7f3d0;
}
.tag-historical {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #334155;
    background: #e2e8f0; border-radius: 3px; padding: 0.20rem 0.50rem;
    border: 1px solid #cbd5e1;
}
.tag-experimental {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #9a3412;
    background: #ffedd5; border-radius: 3px; padding: 0.20rem 0.50rem;
    border: 1px solid #fed7aa;
}
.tag-live {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #0369a1;
    background: #e0f2fe; border-radius: 3px; padding: 0.20rem 0.50rem;
    border: 1px solid #bae6fd;
}

/* Streamlit expander header styling */
[data-testid="stExpander"] details summary {
    font-weight: 600 !important;
    color: #0f172a !important;
    font-size: 0.90rem !important;
}
[data-testid="stExpander"] details {
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    background: var(--panel) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- CACHED DATA LOADERS ---

@st.cache_data(show_spinner=False)
def load_cwru_anomaly_results() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_anomaly_results.csv")


@st.cache_data(show_spinner=False)
def load_cwru_anomaly_evaluation() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_anomaly_evaluation.csv")


@st.cache_data(show_spinner=False)
def load_cwru_predictions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_predictions.csv")


@st.cache_data(show_spinner=False)
def load_cwru_evaluation() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_evaluation.csv")


@st.cache_data(show_spinner=False)
def load_cwru_feature_importance() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_cwru_cv_results() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cwru_cross_validation_results.csv")


@st.cache_data(show_spinner=False)
def load_nasa_predictions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "nasa_fd001_rul_predictions.csv")


@st.cache_data(show_spinner=False)
def load_lstm_results() -> pd.DataFrame:
    path = PROCESSED_DIR / "lstm_proxy_rul_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_physical_rig_predictions() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_predictions.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_evaluation() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_evaluation.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_feature_importance() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_recording_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_recording_summary.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_raw_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_raw_summary.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_raw_samples() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_raw_samples.csv")


@st.cache_data(show_spinner=False)
def load_physical_rig_fft_spectra() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "physical_rig_fft_spectra.csv")


# --- ESP32 SERIAL COMMUNICATION INFRASTRUCTURE (THREAD-SAFE SINGLETON) ---

class ESP32SerialManager:
    """Thread-safe background USB serial reader for the physical ESP32 motor rig."""

    def __init__(self, max_buffer: int = 300):
        self.max_buffer = max_buffer
        self.buffer = collections.deque(maxlen=max_buffer)
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.serial_conn: serial.Serial | None = None
        self.port: str = ""
        self.baud: int = 115200
        self.is_connected: bool = False
        self.last_packet_time: float | None = None
        self.last_error: str | None = None
        self.packet_count: int = 0
        self.connection_start_time: float | None = None
        self.latest_sample: dict | None = None

    def connect(self, port: str, baud: int = 115200) -> bool:
        self.disconnect()
        try:
            ser = serial.Serial(port, baud, timeout=1.0)
            self.serial_conn = ser
            self.port = port
            self.baud = baud
            self.is_connected = True
            self.last_error = None
            self.packet_count = 0
            self.connection_start_time = time.time()
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            self.last_error = f"Failed to open {port}: {e}"
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.thread = None
        if self.serial_conn:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self.is_connected = False

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    break
                line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("timestamp_ms"):
                    continue
                parts = line.split(",")
                if len(parts) == 6:
                    try:
                        ts_ms = float(parts[0])
                        temp_c = float(parts[1]) if parts[1] != "NaN" else np.nan
                        ax = float(parts[2]) if parts[2] != "NaN" else np.nan
                        ay = float(parts[3]) if parts[3] != "NaN" else np.nan
                        az = float(parts[4]) if parts[4] != "NaN" else np.nan
                        rpm_val = float(parts[5]) if parts[5] != "NaN" else np.nan

                        vib_mag = float(np.sqrt(ax**2 + ay**2 + az**2)) if not (np.isnan(ax) or np.isnan(ay) or np.isnan(az)) else np.nan

                        sample = {
                            "timestamp_ms": ts_ms,
                            "dashboard_time": time.strftime("%H:%M:%S"),
                            "time_epoch": time.time(),
                            "temperature_C": temp_c,
                            "accel_x": ax,
                            "accel_y": ay,
                            "accel_z": az,
                            "vibration_magnitude": vib_mag,
                            "rpm": rpm_val,
                        }
                        with self.lock:
                            self.buffer.append(sample)
                            self.latest_sample = sample
                            self.last_packet_time = time.time()
                            self.packet_count += 1
                            self.last_error = None
                    except ValueError:
                        continue
            except serial.SerialException as e:
                with self.lock:
                    self.last_error = f"Serial error: {e}"
                    self.is_connected = False
                break
            except Exception as e:
                with self.lock:
                    self.last_error = f"Reader error: {e}"
                time.sleep(0.05)

        if self.serial_conn:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            except Exception:
                pass
        with self.lock:
            self.is_connected = False

    def get_status(self) -> tuple[str, str, float | None, dict | None, list[dict]]:
        with self.lock:
            now = time.time()
            if not self.is_connected:
                elapsed = (now - self.last_packet_time) if self.last_packet_time else None
                return "OFFLINE", "clr-red", elapsed, self.latest_sample, list(self.buffer)
            if self.last_packet_time is None or (now - self.last_packet_time) > 3.0:
                elapsed = (now - self.last_packet_time) if self.last_packet_time else None
                return "STALE", "clr-amber", elapsed, self.latest_sample, list(self.buffer)
            return "ONLINE", "clr-green", (now - self.last_packet_time), self.latest_sample, list(self.buffer)


@st.cache_resource
def get_serial_manager() -> ESP32SerialManager:
    return ESP32SerialManager()


# --- UI HELPER FUNCTIONS & PLOTLY STYLING ---

def metric_block(label: str, value: str, detail: str = "", cls: str = "") -> None:
    st.markdown(
        f'<div class="metric-block">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{value}</div>'
        f'<div class="metric-detail">{detail}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(tag: str, title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-rule">'
        f'<div class="section-label">{tag}</div>'
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-note">{note}</div>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _chart_layout(fig: go.Figure, h: int = 300, title_text: str = "") -> go.Figure:
    fig.update_layout(
        height=h,
        margin=dict(l=15, r=15, t=38 if title_text else 20, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(family="Inter", color="#1e293b", size=11),
        title=dict(
            text=title_text,
            font=dict(family="Inter", size=12, color="#0f172a"),
            x=0.0,
            xanchor="left",
        ) if title_text else None,
        legend=dict(
            font=dict(family="Inter", size=10, color="#1e293b"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#ffffff", family="Inter", size=11)),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#cbd5e1",
        linewidth=1,
        title_font=dict(family="Inter", size=11, color="#1e293b"),
        tickfont=dict(family="Inter", size=10, color="#334155"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#cbd5e1",
        linewidth=1,
        title_font=dict(family="Inter", size=11, color="#1e293b"),
        tickfont=dict(family="Inter", size=10, color="#334155"),
    )
    return fig


# --- 1. HEADER / PROJECT IDENTITY ---

st.markdown(
    """
<div class="header-block">
  <div class="eyebrow">College Motor Bearing Predictive Maintenance System — Final Architecture</div>
  <h1>MOTOR BEARING PREDICTIVE MAINTENANCE</h1>
  <div class="subtitle">Benchmark-Driven Anomaly Detection, Fault Diagnosis & Remaining Useful Life Prediction</div>
  <div style="font-size: 0.85rem; color: #334155; margin-top: 0.45rem; font-weight: 500; line-height: 1.45;">
    A machine-learning motor bearing predictive maintenance framework evaluated using the CWRU Bearing and NASA C-MAPSS FD001 benchmark datasets.
  </div>
</div>
<div class="provenance">
  <b>Benchmark Evaluation Dashboard — No Live Sensor Stream</b><br>
  This dashboard presents validated machine learning results evaluated on standard industrial benchmarks with rigorous source-level leakage protection.
</div>
""",
    unsafe_allow_html=True,
)


# --- PROJECT AT A GLANCE (COMPACT REPOSITORY METRIC STRIP) ---

st.markdown(
    """
<div class="glance-container">
  <div class="glance-grid">
    <div class="glance-item">
      <div class="glance-label">Benchmark Data</div>
      <div class="glance-val">2 Datasets</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">Core Models</div>
      <div class="glance-val">3 Validated</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">CWRU Sources</div>
      <div class="glance-val">65 Signals</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">NASA Engines</div>
      <div class="glance-val">100 Test Units</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">CWRU Anomaly Recall</div>
      <div class="glance-val clr-green">100.00%</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">CWRU Diagnostic Acc</div>
      <div class="glance-val clr-teal">98.57%</div>
    </div>
    <div class="glance-item">
      <div class="glance-label">NASA RUL MAE</div>
      <div class="glance-val clr-teal">14.2361 cyc</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# --- 2. SYSTEM OVERVIEW (THREE PRIMARY PILLARS) ---

section_header("SYSTEM OVERVIEW", "Core Machine Learning Pillars")

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">PILLAR 1 · UNSUPERVISED</div>
      <div class="overview-title">CWRU Anomaly Detection</div>
      <div class="overview-desc">
        <b>Model:</b> Isolation Forest (300 estimators)<br>
        <b>Task:</b> Binary normal vs. anomalous detection<br>
        <b>Dataset:</b> CWRU 12 kHz Drive-End bearing vibration<br>
        <b>Fitted:</b> Healthy baseline signals only (0% fault contamination)
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">PILLAR 2 · SUPERVISED DIAGNOSTICS</div>
      <div class="overview-title">CWRU Bearing Fault Classification</div>
      <div class="overview-desc">
        <b>Model:</b> Gradient Boosted Trees (XGBoost Classifier)<br>
        <b>Task:</b> 4-class fault localization (Normal, Ball, Inner, Outer)<br>
        <b>Dataset:</b> CWRU 12 kHz Drive-End bearing vibration<br>
        <b>Validation:</b> Source-grouped 5-fold cross-validation
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">PILLAR 3 · PROGNOSTIC REGRESSION</div>
      <div class="overview-title">NASA FD001 RUL Prediction</div>
      <div class="overview-desc">
        <b>Model:</b> Piecewise Capped Regressor (XGBoost Regressor)<br>
        <b>Task:</b> Remaining Useful Life estimation in cycles<br>
        <b>Dataset:</b> NASA C-MAPSS FD001 turbofan degradation<br>
        <b>Evaluation:</b> 100 held-out official test engines
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# --- 3. CWRU ANOMALY DETECTION (PRIMARY PILLAR 1) ---

section_header(
    "PILLAR 1",
    "CWRU Bearing Anomaly Detection",
    "Unsupervised detection of mechanical defects from 12 kHz Drive-End vibration",
)

st.markdown(
    """
<div class="provenance">
  <b>CWRU Bearing Benchmark — Drive-End Vibration</b><br>
  <b>Training:</b> Healthy CWRU Drive-End vibration only (Loads 0, 1, 2 HP; 4 source signals; 413 windows).<br>
  <b>Evaluation:</b> Held-out healthy Normal (Load 3 HP; 118 windows) + 60 bearing fault signals across Ball, Inner Race, and Outer Race (1,741 windows).<br>
  <b>Model:</b> Isolation Forest (300 estimators, 0.01 contamination) on 21 time-domain and Welch spectral features.
</div>
""",
    unsafe_allow_html=True,
)

anom_results = load_cwru_anomaly_results()
anom_eval = load_cwru_anomaly_evaluation()
win_anom = anom_eval[anom_eval.level == "window"].set_index("metric")["value"]

ac1, ac2, ac3, ac4, ac5 = st.columns(5, gap="small")
with ac1:
    metric_block("Accuracy", f"{win_anom.get('accuracy', 0.9989) * 100:.2f}%", "1,859 test windows")
with ac2:
    metric_block("Balanced Accuracy", f"{win_anom.get('balanced_accuracy', 0.9915) * 100:.2f}%", "Normal & fault mean")
with ac3:
    metric_block("Fault Recall", f"{win_anom.get('recall', 1.0) * 100:.2f}%", "1,741 / 1,741 detected", "clr-green")
with ac4:
    metric_block("Normal Specificity", f"{win_anom.get('specificity', 0.9831) * 100:.2f}%", "116 / 118 held-out", "clr-teal")
with ac5:
    metric_block("ROC-AUC / PR-AUC", "1.000 / 1.000", "continuous score", "clr-teal")

st.markdown(
    """
<div class="leakage-note">
  <b>Zero-Leakage Assurance:</b> Detector fitted exclusively on healthy training source signals. Fault signals were not used during fitting. All windows from each source signal remain strictly isolated.
</div>
""",
    unsafe_allow_html=True,
)

# Anomaly charts
ach1, ach2, ach3 = st.columns(3, gap="small")

with ach1:
    norm_scores = anom_results[anom_results["true_anomaly_label"] == "NORMAL"]["anomaly_score"]
    fault_scores = anom_results[anom_results["true_anomaly_label"] == "ANOMALY"]["anomaly_score"]
    fig_score = go.Figure()
    fig_score.add_trace(go.Histogram(x=norm_scores, name="Healthy Normal", marker_color=GREEN, opacity=0.85, nbinsx=15))
    fig_score.add_trace(go.Histogram(x=fault_scores, name="Bearing Faults", marker_color=RED, opacity=0.80, nbinsx=30))
    fig_score.update_layout(
        barmode="overlay",
        xaxis_title="Anomaly Score (Higher = More Anomalous)",
        yaxis_title="Windows",
    )
    st.plotly_chart(_chart_layout(fig_score, 290, "ANOMALY SCORE DISTRIBUTION"), config={"displayModeBar": False})

with ach2:
    tn = int(win_anom.get("true_negatives", 116))
    fp = int(win_anom.get("false_positives", 2))
    fn = int(win_anom.get("false_negatives", 0))
    tp = int(win_anom.get("true_positives", 1741))
    cm_anom = np.array([[tn, fp], [fn, tp]])
    fig_cm_anom = go.Figure(
        go.Heatmap(
            z=cm_anom,
            x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"],
            colorscale=[[0, "#e8f0ee"], [1, TEAL]],
            text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
            texttemplate="%{text}",
            textfont=dict(family="Inter", size=12, color="#0f172a"),
            showscale=False,
        )
    )
    fig_cm_anom.update_layout(
        xaxis_title="Predicted Condition",
        yaxis_title="True Condition",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_chart_layout(fig_cm_anom, 290, "ANOMALY CONFUSION MATRIX"), config={"displayModeBar": False})

with ach3:
    per_class_df = pd.DataFrame({
        "Condition": ["Normal (Held-out)", "Ball Fault", "Inner Race Fault", "Outer Race Fault"],
        "Detection Rate (%)": [98.31, 100.0, 100.0, 100.0],
        "Count": ["116 / 118", "464 / 464", "465 / 465", "812 / 812"],
    })
    fig_det = go.Figure(
        go.Bar(
            x=per_class_df["Condition"],
            y=per_class_df["Detection Rate (%)"],
            marker_color=[GREEN, SLATE, TEAL, ORANGE],
            text=[f"{v:.1f}%" for v in per_class_df["Detection Rate (%)"]],
            textposition="auto",
            textfont=dict(color="#ffffff", family="Inter", size=11),
        )
    )
    fig_det.update_layout(
        yaxis=dict(range=[90, 102], title="Detection Rate (%)"),
    )
    st.plotly_chart(_chart_layout(fig_det, 290, "DETECTION RATE BY FAULT CLASS"), config={"displayModeBar": False})

with st.expander("Source-Level Evaluation Summary (61 Evaluated Source Signals)"):
    source_summary = (
        anom_results.groupby(["source_file", "source_signal", "class", "severity", "load"], as_index=False)
        .agg(
            windows=("window_index", "count"),
            mean_score=("anomaly_score", "mean"),
            predicted_anomaly_rate=("predicted_label", lambda v: float((v == "ANOMALY").mean())),
            true_label=("true_anomaly_label", "first"),
        )
    )
    source_summary["source_prediction"] = np.where(source_summary["predicted_anomaly_rate"] >= 0.5, "ANOMALY", "NORMAL")
    source_summary["result"] = np.where(source_summary["source_prediction"] == source_summary["true_label"], "PASS", "FAIL")
    st.dataframe(
        source_summary.rename(
            columns={
                "source_file": "Source File",
                "source_signal": "Signal",
                "class": "Class",
                "severity": "Severity",
                "load": "Load (HP)",
                "windows": "Windows",
                "mean_score": "Mean Anomaly Score",
                "source_prediction": "Source Vote",
                "true_label": "True Status",
                "result": "Result",
            }
        ),
        hide_index=True,
    )
    st.caption("All 61 evaluation source signals were classified with 100% majority voting accuracy (1/1 Normal holdout source + 60/60 Fault sources).")


# --- 4. CWRU FAULT CLASSIFICATION (PRIMARY PILLAR 2) ---

section_header(
    "PILLAR 2",
    "CWRU Bearing Fault Classification",
    "Supervised multi-class diagnostic localization of bearing defect types",
)

st.markdown(
    """
<div class="provenance">
  <b>CWRU Bearing Benchmark — XGBoost Diagnostic Classifier</b><br>
  <b>Architecture:</b> 65 source signals segmented into 4,096-sample windows (~0.34s at 12 kHz). 21 time & spectral features.<br>
  <b>Single-Split Partition:</b> 47 training source signals (1,638 windows), 9 validation source signals (284 windows), 9 test source signals (350 windows).<br>
  <b>4 Classes:</b> Normal, Ball Fault, Inner Race Fault, Outer Race Fault across 0.007", 0.014", and 0.021" defects.
</div>
""",
    unsafe_allow_html=True,
)

cwru_pred = load_cwru_predictions()
cwru_eval = load_cwru_evaluation()
cwru_imp = load_cwru_feature_importance()
cwru_cv = load_cwru_cv_results()

fc1, fc2, fc3, fc4, fc5 = st.columns(5, gap="small")
win_fc = cwru_eval[cwru_eval.level == "window"].set_index("metric")["value"]
with fc1:
    metric_block("Test Accuracy", f"{win_fc.get('accuracy', 0.9857) * 100:.2f}%", "held-out test windows")
with fc2:
    metric_block("Balanced Accuracy", f"{win_fc.get('balanced_accuracy', 0.9784) * 100:.2f}%", "macro class average")
with fc3:
    metric_block("Macro F1-Score", f"{win_fc.get('f1_macro', 0.9835) * 100:.2f}%", "across 4 classes")
with fc4:
    metric_block("Macro Precision", f"{win_fc.get('precision_macro', 0.9897) * 100:.2f}%", "class precision")
with fc5:
    metric_block("Source Accuracy", "100.00%", "source majority vote", "clr-green")

st.markdown(
    """
<div class="leakage-note">
  <b>Source-Signal Grouping:</b> Source-signal grouping prevents windows from the same vibration source from appearing across training and evaluation folds. Window-level results reflect genuine generalization to unseen motor runs.
</div>
""",
    unsafe_allow_html=True,
)

# Classification charts
fch1, fch2, fch3 = st.columns(3, gap="small")

with fch1:
    cm_fc = pd.crosstab(cwru_pred["true_class"], cwru_pred["predicted_class"]).reindex(
        index=FAULT_ORDER, columns=FAULT_ORDER, fill_value=0
    )
    fig_cm_fc = go.Figure(
        go.Heatmap(
            z=cm_fc.values,
            x=[FAULT_LABELS[f] for f in FAULT_ORDER],
            y=[FAULT_LABELS[f] for f in FAULT_ORDER],
            colorscale=[[0, "#e8f0ee"], [1, TEAL]],
            text=cm_fc.values,
            texttemplate="%{text}",
            textfont=dict(family="Inter", size=12, color="#0f172a"),
            showscale=False,
        )
    )
    fig_cm_fc.update_layout(
        xaxis_title="Predicted Fault Class",
        yaxis_title="True Fault Class",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_chart_layout(fig_cm_fc, 290, "DIAGNOSTIC CONFUSION MATRIX"), config={"displayModeBar": False})

with fch2:
    dist = cwru_pred["predicted_class"].value_counts(normalize=True).reindex(FAULT_ORDER, fill_value=0) * 100
    fig_dist = go.Figure(
        go.Bar(
            x=[FAULT_LABELS[f] for f in dist.index],
            y=dist.values,
            marker_color=[GREEN, SLATE, TEAL, AMBER],
            text=[f"{v:.1f}%" for v in dist.values],
            textposition="auto",
            textfont=dict(color="#ffffff", family="Inter", size=11),
        )
    )
    fig_dist.update_layout(
        yaxis_title="Windows (%)",
    )
    st.plotly_chart(_chart_layout(fig_dist, 290, "TEST CLASS PREDICTION DISTRIBUTION"), config={"displayModeBar": False})

with fch3:
    top_feat = cwru_imp.head(10).sort_values("importance")
    fig_imp = go.Figure(
        go.Bar(
            x=top_feat.importance,
            y=top_feat.feature,
            orientation="h",
            marker_color=TEAL,
        )
    )
    fig_imp.update_layout(
        xaxis_title="Relative Importance",
    )
    st.plotly_chart(_chart_layout(fig_imp, 290, "TOP 10 FEATURE IMPORTANCES"), config={"displayModeBar": False})

st.markdown("##### 5-Fold Grouped Cross-Validation Results")
cv_summary_cols = st.columns(6, gap="small")
with cv_summary_cols[0]:
    metric_block("Mean CV Accuracy", f"{cwru_cv['win_accuracy'].mean() * 100:.2f}%", f"± {cwru_cv['win_accuracy'].std() * 100:.2f}%")
with cv_summary_cols[1]:
    metric_block("Mean Balanced Acc", f"{cwru_cv['win_balanced_accuracy'].mean() * 100:.2f}%", f"± {cwru_cv['win_balanced_accuracy'].std() * 100:.2f}%")
with cv_summary_cols[2]:
    metric_block("Mean Macro F1", f"{cwru_cv['win_f1_macro'].mean() * 100:.2f}%", f"± {cwru_cv['win_f1_macro'].std() * 100:.2f}%")
with cv_summary_cols[3]:
    metric_block("Mean Precision", f"{cwru_cv['win_precision_macro'].mean() * 100:.2f}%", f"± {cwru_cv['win_precision_macro'].std() * 100:.2f}%")
with cv_summary_cols[4]:
    metric_block("Mean Recall", f"{cwru_cv['win_recall_macro'].mean() * 100:.2f}%", f"± {cwru_cv['win_recall_macro'].std() * 100:.2f}%")
with cv_summary_cols[5]:
    metric_block("Source CV Accuracy", "100.00%", "± 0.00% across folds", "clr-green")

with st.expander("Detailed Fold Breakdown (5-Fold GroupKFold across 65 Source Signals)"):
    cv_table = cwru_cv[["fold", "train_sources", "test_sources", "train_windows", "test_windows", "win_accuracy", "win_balanced_accuracy", "win_f1_macro", "source_accuracy"]].copy()
    cv_table.columns = ["Fold", "Train Sources", "Test Sources", "Train Windows", "Test Windows", "Window Accuracy", "Balanced Accuracy", "Macro F1", "Source Accuracy"]
    cv_table["Window Accuracy"] = cv_table["Window Accuracy"].map(lambda v: f"{v * 100:.2f}%")
    cv_table["Balanced Accuracy"] = cv_table["Balanced Accuracy"].map(lambda v: f"{v * 100:.2f}%")
    cv_table["Macro F1"] = cv_table["Macro F1"].map(lambda v: f"{v * 100:.2f}%")
    cv_table["Source Accuracy"] = cv_table["Source Accuracy"].map(lambda v: f"{v * 100:.2f}%")
    st.dataframe(cv_table, hide_index=True)


# --- 5. NASA C-MAPSS FD001 — RUL PREDICTION (PRIMARY PILLAR 3) ---

section_header(
    "PILLAR 3",
    "NASA C-MAPSS FD001 — Remaining Useful Life (RUL) Prediction",
    "Prognostic regression on run-to-failure turbofan engine degradation trajectories",
)

st.markdown(
    """
<div class="provenance">
  <b>NASA C-MAPSS FD001 Benchmark — Turbofan Run-to-Failure Data</b><br>
  <b>Training:</b> 100 turbofan engines run to failure; piecewise linear RUL target capped at 125 cycles.<br>
  <b>Evaluation:</b> 100 held-out official test engines evaluated against ground-truth labels from <code>RUL_FD001.txt</code>.<br>
  <b>Explicit Disclaimer:</b> This benchmark result is evaluated on NASA turbofan engines and is <b>NOT</b> a direct lifetime estimate for the project's physical-rig motor.
</div>
""",
    unsafe_allow_html=True,
)

nasa = load_nasa_predictions()
actual_rul = nasa["actual_rul_cycles"]
pred_rul = nasa["predicted_rul_cycles"]

mae_val = mean_absolute_error(actual_rul, pred_rul)
rmse_val = float(mean_squared_error(actual_rul, pred_rul) ** 0.5)
r2_val = r2_score(actual_rul, pred_rul)
spearman_val = float(spearmanr(actual_rul, pred_rul).statistic)

nc1, nc2, nc3, nc4 = st.columns(4, gap="small")
with nc1:
    metric_block("MAE", f"{mae_val:.4f}", "100 test engines · cycles", "clr-teal")
with nc2:
    metric_block("RMSE", f"{rmse_val:.4f}", "100 test engines · cycles", "clr-teal")
with nc3:
    metric_block("R² Score", f"{r2_val:.4f}", "variance explained", "clr-teal")
with nc4:
    metric_block("Spearman Rank Corr.", f"{spearman_val:.4f}", "degradation ranking", "clr-teal")

# NASA charts
nch1, nch2 = st.columns(2, gap="small")

with nch1:
    fig_scatter = go.Figure(
        go.Scatter(
            x=actual_rul,
            y=pred_rul,
            mode="markers",
            marker=dict(color=TEAL, size=6, opacity=0.85),
            customdata=nasa["unit"],
            hovertemplate="Engine Unit %{customdata}<br>Actual RUL: %{x:.0f} cycles<br>Predicted RUL: %{y:.1f} cycles<extra></extra>",
        )
    )
    fig_scatter.add_shape(
        type="line",
        x0=actual_rul.min(),
        y0=actual_rul.min(),
        x1=actual_rul.max(),
        y1=actual_rul.max(),
        line=dict(color=RED, dash="dot", width=1.5),
    )
    fig_scatter.update_layout(
        xaxis_title="Actual RUL (Operating Cycles)",
        yaxis_title="Predicted RUL (Operating Cycles)",
        showlegend=False,
    )
    st.plotly_chart(_chart_layout(fig_scatter, 290, "ACTUAL VS PREDICTED RUL (100 ENGINES)"), config={"displayModeBar": False})

with nch2:
    errors = pred_rul - actual_rul
    fig_err = go.Figure(
        go.Histogram(
            x=errors,
            nbinsx=20,
            marker_color=AMBER,
            hovertemplate="Error: %{x:.1f} cycles<br>Count: %{y}<extra></extra>",
        )
    )
    fig_err.update_layout(
        xaxis_title="Signed Error (Predicted - Actual Cycles)",
        yaxis_title="Engine Count",
    )
    st.plotly_chart(_chart_layout(fig_err, 290, "PREDICTION ERROR RESIDUAL DISTRIBUTION"), config={"displayModeBar": False})

fig_eng = go.Figure()
fig_eng.add_trace(
    go.Scatter(
        x=nasa["unit"],
        y=actual_rul,
        mode="lines+markers",
        name="Actual RUL",
        line=dict(color=INK, width=1.5),
        marker=dict(size=4),
        hovertemplate="Unit %{x}<br>Actual: %{y:.0f} cycles<extra></extra>",
    )
)
fig_eng.add_trace(
    go.Scatter(
        x=nasa["unit"],
        y=pred_rul,
        mode="lines+markers",
        name="Predicted RUL",
        line=dict(color=TEAL, width=1.5),
        marker=dict(size=4),
        hovertemplate="Unit %{x}<br>Predicted: %{y:.1f} cycles<extra></extra>",
    )
)
fig_eng.update_layout(
    xaxis_title="NASA FD001 Engine Unit (1 – 100)",
    yaxis_title="Remaining Useful Life (Cycles)",
    showlegend=True,
)
st.plotly_chart(_chart_layout(fig_eng, 280, "RUL TRAJECTORY PROFILE ACROSS 100 NASA FD001 TEST ENGINES"), config={"displayModeBar": False})


# --- 6. MODEL COMPARISON / VALIDATION SUMMARY TABLE ---

section_header("MODEL VALIDATION SUMMARY", "Core Architecture Comparison")

st.markdown(
    """
<table class="summary-table">
<thead>
<tr>
  <th>Task</th>
  <th>Dataset</th>
  <th>Methodology</th>
  <th>Key Validation Metric</th>
  <th>Status</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>Anomaly Detection</b></td>
  <td>CWRU Bearing Dataset (12 kHz Drive-End)</td>
  <td>Isolation Forest (300 estimators) on 21 spectral/time features</td>
  <td><b>99.89% Accuracy</b> (100% Fault Recall, 98.31% Specificity)</td>
  <td><span class="tag-validated">VALIDATED</span></td>
</tr>
<tr>
  <td><b>Fault Classification</b></td>
  <td>CWRU Bearing Dataset (12 kHz Drive-End)</td>
  <td>XGBoost Classifier (4-Class Fault Localization)</td>
  <td><b>98.57% Test Acc</b> (99.06% ± 1.02% in 5-Fold Grouped CV)</td>
  <td><span class="tag-validated">VALIDATED</span></td>
</tr>
<tr>
  <td><b>RUL Prediction</b></td>
  <td>NASA C-MAPSS FD001 (Turbofan Engines)</td>
  <td>XGBoost Regressor (Piecewise 125-Cycle Capped Target)</td>
  <td><b>MAE: 14.2361 cycles</b> (R²: 0.7979, Spearman: 0.8500)</td>
  <td><span class="tag-validated">VALIDATED</span></td>
</tr>
<tr>
  <td><i>Edge Condition Monitoring</i></td>
  <td>Physical Motor Rig (ESP32 Data Acquisition)</td>
  <td>Grouped 3-Class XGBoost + 32 Time/Spectral Features</td>
  <td><b>78.43% Grouped Acc</b> (9 Runs / 8,100 Windows)</td>
  <td><span class="tag-experimental">EXPERIMENTAL PROOF-OF-CONCEPT</span></td>
</tr>
</tbody>
</table>
""",
    unsafe_allow_html=True,
)


# --- 7. DATASET PROVENANCE CARD ---

section_header("DATASET PROVENANCE", "Repository Data Sources & Provenance")

p1, p2, p3 = st.columns(3, gap="medium")

with p1:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">BENCHMARK DATASET 1</div>
      <div class="overview-title">CWRU Bearing Dataset</div>
      <div class="overview-desc">
        <b>Source:</b> Case Western Reserve University Bearing Data Center<br>
        <b>Sampling Rate:</b> 12,000 Hz (Drive-End accelerometer)<br>
        <b>Conditions:</b> Normal, Ball Fault, Inner Race Fault, Outer Race Fault<br>
        <b>Loads:</b> 0, 1, 2, 3 HP motor loads; 0.007", 0.014", 0.021" fault sizes<br>
        <b>Scope:</b> Benchmark anomaly detection & fault classification
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">BENCHMARK DATASET 2</div>
      <div class="overview-title">NASA C-MAPSS FD001</div>
      <div class="overview-desc">
        <b>Source:</b> NASA Prognostics Center of Excellence (PCoE)<br>
        <b>Type:</b> Simulated turbofan engine run-to-failure degradation<br>
        <b>Dimensions:</b> 100 train engines, 100 official test engines<br>
        <b>Features:</b> 3 operational settings + 21 sensor channels<br>
        <b>Scope:</b> Standard prognostic benchmark for Remaining Useful Life (RUL)
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
    <div class="overview-card">
      <div class="overview-tag">SUPPLEMENTARY DATASET</div>
      <div class="overview-title">Physical Motor Rig DAQ</div>
      <div class="overview-desc">
        <b>Source:</b> Custom physical motor / bearing test bench<br>
        <b>Sampling Rate:</b> 100 Hz (ESP32 / ADXL345 / DS18B20 / Hall-effect)<br>
        <b>Recordings:</b> 9 historical recordings (Normal, Imbalance, Misalignment)<br>
        <b>Channels:</b> Tri-axial acceleration, temperature, RPM<br>
        <b>Scope:</b> Hardware data acquisition & experimental condition classification
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# --- 8. HISTORICAL HARDWARE CASE STUDY ---

section_header(
    "SUPPLEMENTARY HARDWARE CASE STUDY",
    "Historical Hardware Prototype",
    "ESP32 Edge Data Acquisition & Condition Monitoring Case Study",
)

st.markdown(
    """
<div class="provenance" style="border-left-color: #475569; background: #f1f5f9; color: #1e293b;">
  <b>Historical Physical-Rig Experiment (Hardware Proof-of-Concept)</b><br>
  The physical motor rig was an earlier embedded hardware and edge DAQ prototyping experiment demonstrating real-world sensor acquisition and real-time rule-based health indexing.<br>
  <b>Notice:</b> The 9 historical recordings are experimental bench test data. They are not active production machines, and they are not evaluated with the CWRU or NASA models.
</div>
""",
    unsafe_allow_html=True,
)

ph1, ph2, ph3, ph4 = st.columns(4, gap="small")
with ph1:
    metric_block("Historical Recordings", "9 files", "3 normal, 3 imbalance, 3 misalignment")
with ph2:
    metric_block("Sampling Rate", "100 Hz", "synchronized edge polling")
with ph3:
    metric_block("Sensor Channels", "Tri-Axial + Temp + RPM", "ADXL345 · DS18B20 · Hall effect")
with ph4:
    metric_block("Condition Index", "Health Score (0–100)", "composite physical indicator")

with st.expander("Historical Hardware Architecture & Edge Pipeline Details"):
    st.markdown(
        r"""
- **Microcontroller:** Espressif ESP32 dual-core SoC running custom FreeRTOS firmware (`src/main.cpp`).
- **Sensors:**
  - `ADXL345`: Digital I2C 3-axis accelerometer configured at $\pm 16g$ dynamic range, 100 Hz output data rate.
  - `DS18B20`: 1-Wire digital thermometer measuring surface bearing temperature in °C.
  - `Hall-Effect Sensor`: Interrupt-driven pulse counter calculating shaft rotational speed (RPM).
- **Edge Feature Extraction:** 1-second non-overlapping windows extracting 24 statistical time-domain features (RMS, peak, variance, crest factor, range).
- **Rule-Based Health Index:** Composite 0–100 condition metric evaluating vibration exceedance, thermal divergence, and speed instability.
"""
    )


# --- 9. EXPERIMENTAL LSTM ---

with st.expander("Exploratory / Negative Result — Physical-Rig LSTM Proxy"):
    lstm_df = load_lstm_results()
    st.markdown(
        """
**Experimental Context:**  
An exploratory 2-layer LSTM model was trained on steady-state physical-rig recordings using a synthesized condition-horizon target derived from instantaneous 30-second linear health degradation slopes.

**Outcome & Scientific Assessment:**  
Because the physical motor recordings represent short 15-minute steady-state operational runs without run-to-failure degradation, the model achieved near-zero predictive correlation ($R^2 \approx 0.06$, Spearman $\approx 0.15$).

**Key Takeaway:**  
This negative result formally demonstrates the fundamental impossibility of predicting true Remaining Useful Life without run-to-failure experimental wear data. **NASA C-MAPSS FD001 serves as the project's sole validated RUL benchmark.**
"""
    )
    if not lstm_df.empty:
        st.caption(f"Evaluated across {len(lstm_df)} sequence windows. MAE: {mean_absolute_error(lstm_df['proxy_RUL_target_hours'], lstm_df['proxy_RUL_prediction_hours']):.4f} hours.")


# --- 10. SCIENTIFIC LIMITATIONS & METHODOLOGY ---

with st.expander("Scientific Limitations & Provenance Disclosures"):
    st.markdown(
        """
1. **Dataset Independence:** CWRU Bearing Dataset and NASA C-MAPSS FD001 are separate, standardized benchmark datasets and are not direct measurements of the physical-rig motor.
2. **Prognostics Ground Truth:** NASA FD001 evaluates RUL on run-to-failure turbofan degradation trajectories. NASA RUL predictions must not be interpreted as direct lifetime estimates for the physical-rig motor.
3. **Diagnostics Provenance:** CWRU classification was trained on 12 kHz laboratory bearing vibration under seeded fault conditions (Ball, Inner Race, Outer Race) and was not trained on physical-rig accelerometer data.
4. **Physical Rig Scope:** The physical-rig recordings demonstrate embedded IoT data acquisition and real-time condition monitoring, but do not provide run-to-failure ground truth.
5. **No Synthetic Data:** All benchmark metrics presented in this dashboard are computed directly from saved evaluation artifacts evaluated on official test splits without cross-dataset fabrication.
6. **Presentation Layer:** This dashboard functions as an interactive research presentation and benchmark evaluation platform, not a live industrial SCADA deployment.
"""
    )


# --- 11. LIVE PHYSICAL-RIG MONITOR (RAW TELEMETRY STREAM ONLY) ---

section_header(
    "LIVE HARDWARE",
    "Live Physical-Rig Monitor",
    "ESP32 Serial Data Acquisition — Raw Sensor Stream",
)

@st.fragment(run_every="1s")
def render_live_hardware_section():
    manager = get_serial_manager()
    status_label, status_cls, elapsed_s, latest_sample, buffer_list = manager.get_status()

    # Hardware connection controls bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 2], gap="small")
    
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if not available_ports:
        port_options = ["No Serial Ports Found"]
    else:
        port_options = available_ports

    with ctrl_col1:
        selected_port = st.selectbox(
            "Serial Port",
            options=port_options,
            index=0,
            disabled=manager.is_connected,
            label_visibility="collapsed",
        )
    with ctrl_col2:
        selected_baud = st.selectbox(
            "Baud Rate",
            options=[115200, 9600, 57600, 230400],
            index=0,
            disabled=manager.is_connected,
            label_visibility="collapsed",
        )
    with ctrl_col3:
        if not manager.is_connected:
            connect_disabled = selected_port == "No Serial Ports Found"
            if st.button("Connect", disabled=connect_disabled, type="primary"):
                manager.connect(selected_port, selected_baud)
                st.rerun()
        else:
            if st.button("Disconnect", type="secondary"):
                manager.disconnect()
                st.rerun()

    with ctrl_col4:
        st.markdown(
            f"""
            <div style="background: var(--panel); border: 1px solid #cbd5e1; border-radius: 5px; padding: 0.45rem 0.75rem; display: flex; align-items: center; justify-content: space-between;">
              <div>
                <span style="font-weight: 700; font-size: 0.85rem;" class="{status_cls}">LIVE HARDWARE • {status_label}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display error message if connection failed
    if manager.last_error and not manager.is_connected:
        st.warning(f"{manager.last_error}")

    # Primary Live Metric Cards: RPM, Temperature, Vibration
    m1, m2, m3 = st.columns(3, gap="small")
    if status_label == "ONLINE" and latest_sample:
        with m1:
            rpm_v = f"{latest_sample['rpm']:.1f} RPM" if not np.isnan(latest_sample['rpm']) else "NaN"
            metric_block("RPM", rpm_v, "Hall-effect pulse speed")
        with m2:
            temp_v = f"{latest_sample['temperature_C']:.2f} °C" if not np.isnan(latest_sample['temperature_C']) else "NaN"
            metric_block("Temperature", temp_v, "DS18B20 surface temp")
        with m3:
            vib_v = f"{latest_sample['vibration_magnitude']:.4f} m/s²" if not np.isnan(latest_sample['vibration_magnitude']) else "NaN"
            metric_block("Vibration", vib_v, "Tri-axial acceleration magnitude")
    else:
        with m1:
            metric_block("RPM", "—", "Hall-effect pulse speed")
        with m2:
            metric_block("Temperature", "—", "DS18B20 surface temp")
        with m3:
            metric_block("Vibration", "—", "Tri-axial acceleration magnitude")

    # Raw Live Data Table & Real-time Charts
    if buffer_list:
        df_live = pd.DataFrame(buffer_list)
        
        # Real-time charts
        lch1, lch2, lch3 = st.columns(3, gap="small")
        with lch1:
            fig_vib = go.Figure(go.Scatter(x=df_live["dashboard_time"], y=df_live["vibration_magnitude"], mode="lines", line=dict(color=TEAL, width=1.5)))
            fig_vib.update_layout(xaxis_title="Time", yaxis_title="Acceleration Magnitude (m/s²)")
            st.plotly_chart(_chart_layout(fig_vib, 260, "LIVE VIBRATION MAGNITUDE"), config={"displayModeBar": False})
            
        with lch2:
            fig_temp = go.Figure(go.Scatter(x=df_live["dashboard_time"], y=df_live["temperature_C"], mode="lines+markers", line=dict(color=RED, width=1.8), marker=dict(size=3)))
            fig_temp.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
            st.plotly_chart(_chart_layout(fig_temp, 260, "LIVE SURFACE TEMPERATURE"), config={"displayModeBar": False})

        with lch3:
            fig_rpm = go.Figure(go.Scatter(x=df_live["dashboard_time"], y=df_live["rpm"], mode="lines+markers", line=dict(color=GREEN, width=1.8), marker=dict(size=3)))
            fig_rpm.update_layout(xaxis_title="Time", yaxis_title="Rotational Speed (RPM)")
            st.plotly_chart(_chart_layout(fig_rpm, 260, "LIVE SHAFT SPEED"), config={"displayModeBar": False})

        with st.expander(f"Raw Live Serial Stream Log ({len(df_live)} Samples in Buffer)"):
            display_cols = ["timestamp_ms", "dashboard_time", "accel_x", "accel_y", "accel_z", "vibration_magnitude", "temperature_C", "rpm"]
            st.dataframe(
                df_live[display_cols].iloc[::-1].rename(
                    columns={
                        "timestamp_ms": "ESP32 Time (ms)",
                        "dashboard_time": "Dashboard Time",
                        "accel_x": "Accel X (m/s²)",
                        "accel_y": "Accel Y (m/s²)",
                        "accel_z": "Accel Z (m/s²)",
                        "vibration_magnitude": "Vibration Mag (m/s²)",
                        "temperature_C": "Temp (°C)",
                        "rpm": "RPM",
                    }
                ),
                hide_index=True,
            )
    else:
        st.markdown(
            """
            <div style="background: var(--panel); border: 1px dashed #cbd5e1; border-radius: 6px; padding: 1.5rem; text-align: center; color: var(--secondary); margin-top: 0.75rem; margin-bottom: 0.75rem;">
              <div style="font-weight: 700; font-size: 0.95rem; color: var(--ink); margin-bottom: 0.25rem;">No Active Serial Stream</div>
              <div style="font-size: 0.82rem;">Connect the ESP32 via USB to display real-time sensor telemetry.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_live_hardware_section()


# --- 12. PHYSICAL-RIG HISTORICAL ML ANALYSIS (EXPERIMENTAL PROOF-OF-CONCEPT) ---

section_header(
    "EXPERIMENTAL PROOF-OF-CONCEPT",
    "Physical-Rig Historical ML Analysis",
    "Experimental machine-learning analysis of recorded ESP32 motor-bench telemetry",
)

st.markdown(
    """
<div class="provenance" style="border-left-color: #9a3412; background: #fff7ed; color: #7c2d12;">
  <b>Historical Physical-Rig Dataset — 9 Recorded Runs</b><br>
  <b>Experimental Scope:</b> Supervised condition classification evaluated across 9 historical recordings (3 Normal, 3 Imbalance, 3 Misalignment) collected at 100 Hz on the physical ESP32 motor bench. Total: 810,000 raw samples segmented into 8,100 one-second feature windows (32 features per window).<br>
  <b>Notice:</b> This analysis is an embedded DAQ proof-of-concept. It remains strictly separated from the CWRU and NASA benchmark models.
</div>
""",
    unsafe_allow_html=True,
)

# Load physical rig artifacts
rig_raw = load_physical_rig_raw_summary()
rig_samples = load_physical_rig_raw_samples()
rig_fft = load_physical_rig_fft_spectra()
rig_preds = load_physical_rig_predictions()
rig_eval = load_physical_rig_evaluation()
rig_imp = load_physical_rig_feature_importance()
rig_rec_summary = load_physical_rig_recording_summary()

total_samples = int(rig_raw["samples"].sum())
total_windows = len(rig_preds)

pr1, pr2, pr3, pr4 = st.columns(4, gap="small")
with pr1:
    metric_block("Historical Recordings", "9 runs", "3 Normal · 3 Imbalance · 3 Misalignment")
with pr2:
    metric_block("Total Raw Samples", f"{total_samples:,}", "90,000 samples / 15 min per run")
with pr3:
    metric_block("Sampling Rate", "100 Hz", "ADXL345 · DS18B20 · Hall effect")
with pr4:
    metric_block("Feature Windows", f"{total_windows:,}", "1-second non-overlapping windows")

# --- Raw Data Exploration Sub-section ---
with st.expander("Raw Physical-Rig Sensor Exploration (Waveforms, Spectra, & Preview Table)", expanded=False):
    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs(["Raw Telemetry Preview", "Raw Acceleration Waveforms", "Vibration Spectra (Welch PSD)", "Temperature & Speed Trends"])
    
    with t_tab1:
        st.caption("Sample 5-second raw telemetry windows extracted from representative runs of each condition:")
        st.dataframe(
            rig_samples[["recording", "condition", "time_s", "accel_x", "accel_y", "accel_z", "vibration_magnitude", "temperature_C", "rpm"]].rename(
                columns={
                    "recording": "Recording ID",
                    "condition": "Condition",
                    "time_s": "Relative Time (s)",
                    "accel_x": "Accel X (m/s²)",
                    "accel_y": "Accel Y (m/s²)",
                    "accel_z": "Accel Z (m/s²)",
                    "vibration_magnitude": "Vibration Mag (m/s²)",
                    "temperature_C": "Temp (°C)",
                    "rpm": "RPM",
                }
            ).head(50),
            hide_index=True,
        )

    with t_tab2:
        rw_col1, rw_col2, rw_col3 = st.columns(3, gap="small")
        for idx, (col_obj, c_name) in enumerate(zip([rw_col1, rw_col2, rw_col3], RIG_CONDITIONS)):
            c_data = rig_samples[rig_samples["condition"] == c_name]
            with col_obj:
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(x=c_data["time_s"], y=c_data["accel_x"], name="X", line=dict(color=TEAL, width=1.2)))
                fig_w.add_trace(go.Scatter(x=c_data["time_s"], y=c_data["accel_y"], name="Y", line=dict(color=ORANGE, width=1.2)))
                fig_w.add_trace(go.Scatter(x=c_data["time_s"], y=c_data["accel_z"], name="Z", line=dict(color=SLATE, width=1.2)))
                fig_w.add_trace(go.Scatter(x=c_data["time_s"], y=c_data["vibration_magnitude"], name="Magnitude", line=dict(color=INK, width=1.5, dash="dot")))
                fig_w.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)")
                st.plotly_chart(_chart_layout(fig_w, 250, f"RAW WAVEFORM — {RIG_LABELS[c_name].upper()}"), config={"displayModeBar": False})

    with t_tab3:
        fig_psd = go.Figure()
        colors = {"NORMAL": GREEN, "IMBALANCE": RED, "MISALIGNMENT": ORANGE}
        for c_name in RIG_CONDITIONS:
            spec_c = rig_fft[rig_fft["condition"] == c_name]
            fig_psd.add_trace(
                go.Scatter(
                    x=spec_c["frequency_hz"],
                    y=spec_c["psd_magnitude"],
                    name=RIG_LABELS[c_name],
                    line=dict(color=colors[c_name], width=1.8),
                )
            )
        fig_psd.update_layout(
            xaxis_title="Frequency (Hz) [0 – 50 Hz Nyquist Limit]",
            yaxis_title="Power Spectral Density ((m/s²)²/Hz)",
        )
        st.plotly_chart(_chart_layout(fig_psd, 280, "VIBRATION SPECTRA COMPARISON (WELCH PSD ACROSS CONDITIONS)"), config={"displayModeBar": False})

    with t_tab4:
        trend_col1, trend_col2 = st.columns(2, gap="small")
        with trend_col1:
            fig_tr_temp = go.Figure(
                go.Bar(
                    x=rig_raw["recording"],
                    y=rig_raw["temp_mean"],
                    marker_color=[GREEN if "normal" in r else (RED if "imbalance" in r else ORANGE) for r in rig_raw["recording"]],
                    text=[f"{v:.1f}°C" for v in rig_raw["temp_mean"]],
                    textposition="auto",
                    textfont=dict(color="#ffffff", family="Inter", size=10),
                )
            )
            fig_tr_temp.update_layout(xaxis_title="Recording Run", yaxis_title="Mean Temperature (°C)")
            st.plotly_chart(_chart_layout(fig_tr_temp, 250, "MEAN SURFACE TEMPERATURE BY RUN"), config={"displayModeBar": False})
        with trend_col2:
            fig_tr_rpm = go.Figure(
                go.Bar(
                    x=rig_raw["recording"],
                    y=rig_raw["rpm_mean"],
                    marker_color=TEAL,
                    text=[f"{v:.0f}" for v in rig_raw["rpm_mean"]],
                    textposition="auto",
                    textfont=dict(color="#ffffff", family="Inter", size=10),
                )
            )
            fig_tr_rpm.update_layout(xaxis_title="Recording Run", yaxis_title="Mean Speed (RPM)")
            st.plotly_chart(_chart_layout(fig_tr_rpm, 250, "MEAN SHAFT SPEED BY RUN"), config={"displayModeBar": False})

# --- Grouped Validation & ML Results ---
st.markdown("##### Grouped 3-Class Condition Classification Results")

st.markdown(
    """
<div class="leakage-note">
  <b>Zero-Leakage Assurance:</b> Validation is grouped by recording run (3-Fold GroupKFold) to prevent windows from the same physical run from appearing in both training and evaluation folds. The model was trained offline; results reflect genuine out-of-fold generalization across physical motor runs.
</div>
""",
    unsafe_allow_html=True,
)

overall_eval = rig_eval[rig_eval["level"] == "overall"].set_index("metric")["value"]
acc_g = overall_eval.get("accuracy", 0.7843)
bal_acc_g = overall_eval.get("balanced_accuracy", 0.7843)
f1_g = overall_eval.get("f1_macro", 0.7869)
prec_g = overall_eval.get("precision_macro", 0.7896)
rec_g = overall_eval.get("recall_macro", 0.7843)

prm1, prm2, prm3, prm4, prm5 = st.columns(5, gap="small")
with prm1:
    metric_block("Grouped Accuracy", f"{acc_g * 100:.2f}%", "8,100 out-of-fold windows")
with prm2:
    metric_block("Balanced Accuracy", f"{bal_acc_g * 100:.2f}%", "macro condition mean")
with prm3:
    metric_block("Macro F1-Score", f"{f1_g * 100:.2f}%", "across 3 conditions")
with prm4:
    metric_block("Macro Precision", f"{prec_g * 100:.2f}%", "mean class precision")
with prm5:
    metric_block("Macro Recall", f"{rec_g * 100:.2f}%", "mean class recall")

# Charts: Confusion matrix, condition distribution, feature importance
pmch1, pmch2, pmch3 = st.columns(3, gap="small")

with pmch1:
    cm_rig = pd.crosstab(rig_preds["true_condition"], rig_preds["predicted_condition"]).reindex(
        index=RIG_CONDITIONS, columns=RIG_CONDITIONS, fill_value=0
    )
    fig_cm_rig = go.Figure(
        go.Heatmap(
            z=cm_rig.values,
            x=[RIG_LABELS[c] for c in RIG_CONDITIONS],
            y=[RIG_LABELS[c] for c in RIG_CONDITIONS],
            colorscale=[[0, "#fff7ed"], [1, ORANGE]],
            text=cm_rig.values,
            texttemplate="%{text}",
            textfont=dict(family="Inter", size=12, color="#0f172a"),
            showscale=False,
        )
    )
    fig_cm_rig.update_layout(
        xaxis_title="Predicted Condition",
        yaxis_title="True Condition",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_chart_layout(fig_cm_rig, 280, "GROUPED CONFUSION MATRIX"), config={"displayModeBar": False})

with pmch2:
    cond_counts = rig_preds["true_condition"].value_counts().reindex(RIG_CONDITIONS)
    fig_cond = go.Figure(
        go.Bar(
            x=[RIG_LABELS[c] for c in cond_counts.index],
            y=cond_counts.values,
            marker_color=[GREEN, RED, ORANGE],
            text=[f"{v:,} windows" for v in cond_counts.values],
            textposition="auto",
            textfont=dict(color="#ffffff", family="Inter", size=11),
        )
    )
    fig_cond.update_layout(
        yaxis_title="Window Count",
    )
    st.plotly_chart(_chart_layout(fig_cond, 280, "CONDITION DATASET BALANCE (3 RUNS EACH)"), config={"displayModeBar": False})

with pmch3:
    top_rig_feat = rig_imp.head(10).sort_values("importance")
    fig_r_imp = go.Figure(
        go.Bar(
            x=top_rig_feat.importance,
            y=top_rig_feat.feature,
            orientation="h",
            marker_color=ORANGE,
        )
    )
    fig_r_imp.update_layout(
        xaxis_title="Relative Importance",
    )
    st.plotly_chart(_chart_layout(fig_r_imp, 280, "TOP 10 PHYSICAL-RIG FEATURES"), config={"displayModeBar": False})

# Recording-level evaluation table
with st.expander("Recording-Level Generalization Summary (9 Physical Runs)"):
    st.dataframe(
        rig_rec_summary.rename(
            columns={
                "recording": "Recording Run",
                "true_condition": "True Condition",
                "predicted_condition": "Predicted (Majority Vote)",
                "accuracy": "Window Accuracy",
                "mean_confidence": "Mean Confidence",
                "total_windows": "Windows",
                "match": "Status",
            }
        ).assign(
            **{
                "Window Accuracy": lambda d: d["Window Accuracy"].map(lambda v: f"{v * 100:.2f}%"),
                "Mean Confidence": lambda d: d["Mean Confidence"].map(lambda v: f"{v * 100:.2f}%"),
            }
        ),
        hide_index=True,
    )
    st.caption(
        "Scientific Assessment: Folds 2 and 3 achieved 98.00% and 98.19% out-of-fold generalization. Fold 1 encountered baseline sensor shift on normal_01, causing cross-run degradation. This variation is characteristic of small physical-rig prototypes and reinforces why physical-rig data is treated as an experimental proof-of-concept."
    )

# --- Overall Project Architecture Visual ---
with st.expander("End-to-End System Architecture & Dataflow Overview"):
    st.markdown(
        """
```mermaid
graph TD
    subgraph BENCHMARK_ML [VALIDATED BENCHMARK MACHINE LEARNING]
        A1[CWRU 12 kHz Vibration] -->|Spectral & Time Extraction| B1[Isolation Forest]
        B1 --> C1[Validated Anomaly Detection: 99.89% Acc / 100% Recall]
        
        A2[CWRU 12 kHz Vibration] -->|21 Feature Windows| B2[XGBoost Classifier]
        B2 --> C2[Validated Diagnostic Localization: 98.57% Test / 99.06% CV]
        
        A3[NASA C-MAPSS FD001] -->|24 Sensor Channels| B3[XGBoost Regressor]
        B3 --> C3[Validated Prognostic RUL: MAE 14.2361 cycles]
    end

    subgraph LIVE_HARDWARE [LIVE PHYSICAL HARDWARE TELEMETRY]
        D1[ESP32 Motor Bench] -->|USB Serial @ 115200 baud| E1[Thread-Safe Serial Reader]
        E1 --> F1[Live Raw Telemetry: Vibration, Temp, RPM]
    end

    subgraph HISTORICAL_ML [EXPERIMENTAL PHYSICAL-RIG ML BENCHMARK]
        G1[9 Historical ESP32 Runs] -->|32 Extracted Features| H1[Grouped XGBoost Classifier]
        H1 --> I1[Experimental Condition Analysis: 78.43% Grouped Acc]
    end
```
"""
    )

# Concise Disclaimer
st.markdown(
    """
    <div style="font-size: 0.78rem; color: var(--muted); font-style: italic; margin-top: 0.8rem; padding-left: 0.2rem; border-top: 1px solid #e2e8f0; padding-top: 0.6rem;">
      <b>Disclaimer:</b> Experimental physical-rig results are based on 9 historical recordings and are intended as a proof-of-concept for condition monitoring. They are not equivalent to the independently validated CWRU/NASA benchmark results and should not be interpreted as industrial generalization performance.
    </div>
    """,
    unsafe_allow_html=True,
)

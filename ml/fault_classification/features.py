"""Signal processing and feature extraction."""

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 100
WINDOW_SECONDS = 1
WINDOW_SIZE = SAMPLE_RATE_HZ * WINDOW_SECONDS

RAW_COLUMNS = [
    "timestamp_ms",
    "temperature_C",
    "accel_x",
    "accel_y",
    "accel_z",
    "rpm",
]

FEATURE_COLUMNS = [
    "temperature_mean",
    "temperature_std",
    "rpm_mean",
    "rpm_std",
    "rpm_min",
    "rpm_max",
    "accel_x_mean",
    "accel_x_std",
    "accel_x_rms",
    "accel_x_peak",
    "accel_y_mean",
    "accel_y_std",
    "accel_y_rms",
    "accel_y_peak",
    "accel_z_mean",
    "accel_z_std",
    "accel_z_rms",
    "accel_z_peak",
    "vibration_mean",
    "vibration_std",
    "vibration_rms",
    "vibration_peak",
    "vibration_min",
    "vibration_range",
    "timestamp_start_ms",
    "timestamp_end_ms",
    "recording",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def extract_window_features(window: pd.DataFrame, recording: str) -> dict:
    temperature = window["temperature_C"].to_numpy(dtype=np.float64)
    rpm = window["rpm"].to_numpy(dtype=np.float64)
    accel_x = window["accel_x"].to_numpy(dtype=np.float64)
    accel_y = window["accel_y"].to_numpy(dtype=np.float64)
    accel_z = window["accel_z"].to_numpy(dtype=np.float64)
    vibration = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    return {
        "temperature_mean": float(np.mean(temperature)),
        "temperature_std": float(np.std(temperature, ddof=0)),
        "rpm_mean": float(np.mean(rpm)),
        "rpm_std": float(np.std(rpm, ddof=0)),
        "rpm_min": float(np.min(rpm)),
        "rpm_max": float(np.max(rpm)),
        "accel_x_mean": float(np.mean(accel_x)),
        "accel_x_std": float(np.std(accel_x, ddof=0)),
        "accel_x_rms": rms(accel_x),
        "accel_x_peak": float(np.max(np.abs(accel_x))),
        "accel_y_mean": float(np.mean(accel_y)),
        "accel_y_std": float(np.std(accel_y, ddof=0)),
        "accel_y_rms": rms(accel_y),
        "accel_y_peak": float(np.max(np.abs(accel_y))),
        "accel_z_mean": float(np.mean(accel_z)),
        "accel_z_std": float(np.std(accel_z, ddof=0)),
        "accel_z_rms": rms(accel_z),
        "accel_z_peak": float(np.max(np.abs(accel_z))),
        "vibration_mean": float(np.mean(vibration)),
        "vibration_std": float(np.std(vibration, ddof=0)),
        "vibration_rms": rms(vibration),
        "vibration_peak": float(np.max(vibration)),
        "vibration_min": float(np.min(vibration)),
        "vibration_range": float(np.max(vibration) - np.min(vibration)),
        "timestamp_start_ms": float(window["timestamp_ms"].iloc[0]),
        "timestamp_end_ms": float(window["timestamp_ms"].iloc[-1]),
        "recording": recording,
    }


def process_recording(csv_path: Path) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(csv_path)
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    df = df[RAW_COLUMNS].copy()
    sample_count = len(df)
    usable = (sample_count // WINDOW_SIZE) * WINDOW_SIZE
    df = df.iloc[:usable].reset_index(drop=True)

    recording = csv_path.stem
    rows = []
    for start in range(0, usable, WINDOW_SIZE):
        window = df.iloc[start : start + WINDOW_SIZE]
        rows.append(extract_window_features(window, recording))

    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return features, sample_count, len(features)


def run() -> pd.DataFrame:
    root = project_root()
    raw_dir = root / "data" / "raw"
    output_path = root / "data" / "processed" / "features.csv"

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    frames = []
    stats = []
    for csv_path in csv_files:
        features, sample_count, window_count = process_recording(csv_path)
        frames.append(features)
        stats.append((csv_path.stem, sample_count, window_count))

    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("FEATURE ENGINEERING COMPLETE")
    print()
    for recording, sample_count, window_count in stats:
        print(f"{recording}:")
        print(f"  samples: {sample_count}")
        print(f"  windows: {window_count}")
        print()

    print(f"Total recordings: {len(stats)}")
    print(f"Total windows: {len(result)}")
    print(f"Missing values: {int(result.isna().sum().sum())}")
    print(f"Feature count: {len(FEATURE_COLUMNS) - 3}")
    print(f"Output path: {output_path}")

    return result


if __name__ == "__main__":
    run()

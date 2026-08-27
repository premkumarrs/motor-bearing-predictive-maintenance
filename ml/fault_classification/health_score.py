"""Phase 10 machine condition assessment from existing physical-data outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

BASELINE_RECORDINGS = ["normal_01", "normal_02"]
TARGET_RECORDINGS = [
    "normal_03",
    "imbalance_01",
    "imbalance_02",
    "imbalance_03",
    "misalignment_01",
    "misalignment_02",
    "misalignment_03",
]
PHYSICAL_FEATURES = FEATURE_COLUMNS[:-3]
VIBRATION_FEATURES = ["vibration_rms", "vibration_std", "vibration_peak", "vibration_range"]
TEMPERATURE_FEATURES = ["temperature_mean", "temperature_std"]
RPM_FEATURES = ["rpm_mean", "rpm_std"]
RAW_COLUMNS = [
    "timestamp_start_ms",
    "timestamp_end_ms",
    "recording",
    "time_s",
    "true_condition",
    "anomaly_score",
    "anomaly_prediction",
]


class HealthValidationError(Exception):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_inputs(features: pd.DataFrame, results: pd.DataFrame, model_path: Path) -> None:
    required_features = set(PHYSICAL_FEATURES + ["timestamp_start_ms", "timestamp_end_ms", "recording"])
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise HealthValidationError(f"features.csv is missing columns: {missing_features}")
    missing_results = sorted(set(RAW_COLUMNS) - set(results.columns))
    if missing_results:
        raise HealthValidationError(f"anomaly_results.csv is missing columns: {missing_results}")
    if len(results) != 6300:
        raise HealthValidationError(f"Expected 6300 anomaly results, found {len(results)}.")
    expected = {f"{recording}.csv" for recording in TARGET_RECORDINGS}
    if set(results["recording"]) != expected:
        raise HealthValidationError("Anomaly results do not contain exactly the seven target recordings.")
    counts = results["recording"].value_counts()
    if not (counts == 900).all():
        raise HealthValidationError(f"Unexpected windows per recording: {counts.to_dict()}")
    required_values = results[RAW_COLUMNS].isna().any()
    if required_values.any():
        raise HealthValidationError(f"Missing required anomaly result values: {required_values[required_values].index.tolist()}")
    if len(features) != 8100:
        raise HealthValidationError(f"Expected 8100 feature rows, found {len(features)}.")
    if not model_path.exists():
        raise HealthValidationError(f"Anomaly detector does not exist: {model_path}")


def robust_reference(baseline: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    reference = pd.DataFrame(index=columns)
    reference["median"] = baseline[columns].median()
    reference["iqr"] = baseline[columns].quantile(0.75) - baseline[columns].quantile(0.25)
    reference["scale"] = reference["iqr"].replace(0, np.nan)
    return reference


def robust_score(frame: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> pd.Series:
    deviations = []
    for column in columns:
        scale = reference.loc[column, "scale"]
        if pd.isna(scale) or scale == 0:
            deviations.append(pd.Series(0.0, index=frame.index))
        else:
            deviations.append((frame[column] - reference.loc[column, "median"]).abs() / scale)
    mean_deviation = pd.concat(deviations, axis=1).mean(axis=1)
    return 100.0 / (1.0 + mean_deviation)


def assign_operating_state(frame: pd.DataFrame, rpm_threshold: float, rpm_variation_threshold: float) -> pd.Series:
    rpm = frame["rpm_mean"].to_numpy()
    steady_candidate = (rpm >= rpm_threshold) & (frame["rpm_std"].to_numpy() <= rpm_variation_threshold)
    if not steady_candidate.any():
        return pd.Series("STEADY", index=frame.index)
    first_steady = int(np.flatnonzero(steady_candidate)[0])
    last_steady = int(np.flatnonzero(steady_candidate)[-1])
    states = np.full(len(frame), "STEADY", dtype=object)
    for index in range(first_steady):
        states[index] = "STARTUP" if rpm[index] <= rpm_threshold else "STEADY"
    for index in range(last_steady + 1, len(frame)):
        states[index] = "SHUTDOWN" if rpm[index] <= rpm_threshold else "STEADY"
    return pd.Series(states, index=frame.index)


def add_health_components(
    frame: pd.DataFrame,
    physical_reference: pd.DataFrame,
    rpm_reference: pd.DataFrame,
    rpm_threshold: float,
    rpm_variation_threshold: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["operating_state"] = assign_operating_state(frame, rpm_threshold, rpm_variation_threshold)
    frame["vibration_score"] = robust_score(frame, physical_reference, VIBRATION_FEATURES)
    frame["temperature_score"] = robust_score(frame, physical_reference, TEMPERATURE_FEATURES)
    steady_rpm = frame[RPM_FEATURES].copy()
    frame["rpm_score"] = robust_score(steady_rpm, rpm_reference, RPM_FEATURES)
    frame.loc[frame["operating_state"] != "STEADY", "rpm_score"] = 100.0

    frame["anomaly_flag"] = frame["anomaly_prediction"].eq("anomaly").astype(float)
    frame["anomaly_severity"] = (-frame["anomaly_score"] / 0.15).clip(0.0, 1.0)
    frame["anomaly_component"] = 100.0 - 30.0 * frame["anomaly_flag"] * frame["anomaly_severity"]
    frame["persistence_ratio"] = frame["anomaly_flag"].rolling(10, min_periods=1).mean()
    frame["persistence_score"] = 100.0 * (1.0 - frame["persistence_ratio"])

    frame["health_score_raw"] = (
        0.40 * frame["vibration_score"]
        + 0.15 * frame["temperature_score"]
        + 0.10 * frame["rpm_score"]
        + 0.20 * frame["anomaly_component"]
        + 0.15 * frame["persistence_score"]
    ).clip(0.0, 100.0)
    frame["health_score"] = frame["health_score_raw"].rolling(5, min_periods=1).median().clip(0.0, 100.0)
    frame["health_status"] = pd.cut(
        frame["health_score"],
        bins=[-np.inf, 30.0, 60.0, 80.0, np.inf],
        labels=["Critical", "Degraded", "Warning", "Healthy"],
        right=False,
    ).astype(str)
    return frame


def summary_row(recording: str, frame: pd.DataFrame) -> dict:
    steady = frame[frame["operating_state"] == "STEADY"]
    early = steady.head(max(1, min(30, len(steady))))["health_score"]
    late = steady.tail(max(1, min(30, len(steady))))["health_score"]
    return {
        "recording": recording,
        "true_condition": frame["true_condition"].iloc[0],
        "total_windows": len(frame),
        "steady_windows": int((frame["operating_state"] == "STEADY").sum()),
        "startup_windows": int((frame["operating_state"] == "STARTUP").sum()),
        "shutdown_windows": int((frame["operating_state"] == "SHUTDOWN").sum()),
        "mean_health_score": frame["health_score"].mean(),
        "median_health_score": frame["health_score"].median(),
        "minimum_health_score": frame["health_score"].min(),
        "maximum_health_score": frame["health_score"].max(),
        "healthy_windows": int((frame["health_status"] == "Healthy").sum()),
        "warning_windows": int((frame["health_status"] == "Warning").sum()),
        "degraded_windows": int((frame["health_status"] == "Degraded").sum()),
        "critical_windows": int((frame["health_status"] == "Critical").sum()),
        "early_health_score": early.mean() if len(early) else np.nan,
        "late_health_score": late.mean() if len(late) else np.nan,
        "health_change": (late.mean() - early.mean()) if len(early) and len(late) else np.nan,
    }


def health_events(frame: pd.DataFrame) -> list[dict]:
    events = []
    active = []
    for _, row in frame.iterrows():
        if row["health_status"] in ("Degraded", "Critical"):
            if not active or row["time_s"] - active[-1]["time_s"] > 1.01:
                if len(active) >= 3:
                    events.append(active)
                active = []
            active.append(row)
        elif len(active) >= 3:
            events.append(active)
            active = []
        else:
            active = []
    if len(active) >= 3:
        events.append(active)
    return [
        {
            "recording": event[0]["recording"],
            "start_time_s": event[0]["time_s"],
            "end_time_s": event[-1]["time_s"],
            "duration_s": event[-1]["time_s"] - event[0]["time_s"],
            "minimum_health_score": min(row["health_score"] for row in event),
            "mean_health_score": np.mean([row["health_score"] for row in event]),
            "worst_status": "Critical" if any(row["health_status"] == "Critical" for row in event) else "Degraded",
        }
        for event in events
    ]


def run() -> None:
    root = project_root()
    features_path = root / "data" / "processed" / "features.csv"
    results_path = root / "data" / "processed" / "anomaly_results.csv"
    output_path = root / "data" / "processed" / "health_results.csv"
    summary_path = root / "data" / "processed" / "health_summary.csv"
    detector_path = root / "models" / "anomaly_detector.pkl"
    protected = [
        root / "models" / "fault_classifier.pkl",
        detector_path,
        features_path,
        results_path,
        root / "data" / "processed" / "anomaly_summary.csv",
        root / "data" / "processed" / "classification_dataset.csv",
        root / "data" / "processed" / "train.csv",
        root / "data" / "processed" / "test.csv",
    ] + sorted((root / "data" / "raw").glob("*.csv"))
    before_hashes = {path: digest(path) for path in protected}

    features = pd.read_csv(features_path)
    results = pd.read_csv(results_path)
    validate_inputs(features, results, detector_path)
    bundle = joblib.load(detector_path)
    if len(bundle.get("feature_columns", [])) != 24 or bundle.get("baseline_recordings") != ["normal_01.csv", "normal_02.csv"]:
        raise HealthValidationError("Phase 9 baseline metadata is invalid.")

    baseline = features[features["recording"].isin(BASELINE_RECORDINGS)].copy()
    if len(baseline) != 1800:
        raise HealthValidationError(f"Healthy baseline has {len(baseline)} windows, expected 1800.")
    baseline_rpm = baseline[baseline["rpm_mean"] >= baseline["rpm_mean"].median() * 0.8]
    rpm_threshold = float(baseline["rpm_mean"].median() * 0.8)
    rpm_variation_threshold = float(baseline_rpm["rpm_std"].quantile(0.95))
    physical_reference = robust_reference(baseline, VIBRATION_FEATURES + TEMPERATURE_FEATURES)
    rpm_reference = robust_reference(baseline_rpm, RPM_FEATURES)

    frames = []
    for recording in TARGET_RECORDINGS:
        filename = f"{recording}.csv"
        result_frame = results[results["recording"] == filename].copy().reset_index(drop=True)
        feature_frame = features[features["recording"] == recording].copy().reset_index(drop=True)
        if len(result_frame) != len(feature_frame) or len(result_frame) != 900:
            raise HealthValidationError(f"Feature/result alignment failed for {filename}.")
        feature_frame = feature_frame.drop(columns=["recording"])
        combined = feature_frame.merge(
            result_frame[["timestamp_start_ms", "timestamp_end_ms", "recording", "true_condition", "anomaly_score", "anomaly_prediction"]],
            on=["timestamp_start_ms", "timestamp_end_ms"],
            how="inner",
            validate="one_to_one",
        )
        if len(combined) != 900:
            raise HealthValidationError(f"Timestamp alignment failed for {filename}.")
        combined = combined.sort_values("timestamp_start_ms").reset_index(drop=True)
        combined["time_s"] = combined["timestamp_start_ms"] / 1000.0
        combined = add_health_components(combined, physical_reference, rpm_reference, rpm_threshold, rpm_variation_threshold)
        frames.append(combined)
    all_health = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame([summary_row(f"{recording}.csv", all_health[all_health["recording"] == f"{recording}.csv"]) for recording in TARGET_RECORDINGS])
    output_columns = [
        "timestamp_start_ms", "timestamp_end_ms", "time_s", "recording", "true_condition",
        "operating_state", "anomaly_score", "anomaly_prediction", "persistence_ratio",
        "vibration_score", "temperature_score", "rpm_score", "anomaly_component",
        "persistence_score", "health_score_raw", "health_score", "health_status",
    ]
    all_health[output_columns].to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    events = []
    for recording in TARGET_RECORDINGS:
        events.extend(health_events(all_health[all_health["recording"] == f"{recording}.csv"]))
    normal = all_health[all_health["recording"] == "normal_03.csv"]
    fault_summary = summary[summary["true_condition"] != "normal"]
    worst = all_health.nsmallest(10, "health_score")
    distributions = all_health.groupby("recording")["health_score"].agg(
        mean="mean", median="median", std="std", minimum="min",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75), maximum="max",
    )

    print("============================================================")
    print("PHASE 10 - MACHINE HEALTH / CONDITION ASSESSMENT")
    print("============================================================\n")
    print("DATA VALIDATION\n---------------")
    print(f"features.csv rows: {len(features)}")
    print(f"anomaly_results.csv rows: {len(results)}")
    print(f"Target recordings: {len(TARGET_RECORDINGS)}")
    print(f"Windows processed: {len(all_health)}")
    print("Recording boundaries preserved: True")
    print("Required values present: True")
    print()
    print("HEALTHY BASELINE\n----------------")
    print("normal_01.csv and normal_02.csv")
    print(f"Baseline windows: {len(baseline)}")
    print(f"Baseline RPM median: {baseline['rpm_mean'].median():.6f}")
    print(f"RPM operating threshold: {rpm_threshold:.6f}")
    print(f"Healthy steady RPM variation 95th percentile: {rpm_variation_threshold:.6f}")
    print("Fault recordings used for baseline statistics: 0")
    print()
    print("OPERATING CONDITIONS\n--------------------")
    print(f"RPM thresholds: steady RPM >= {rpm_threshold:.6f}; steady rpm_std <= {rpm_variation_threshold:.6f}")
    print(summary[["recording", "startup_windows", "steady_windows", "shutdown_windows"]].to_string(index=False))
    print()
    print("HEALTH SCORE FORMULA\n--------------------")
    print("health_score_raw = 0.40*vibration + 0.15*temperature + 0.10*rpm + 0.20*anomaly + 0.15*persistence")
    print("Each physical component uses 100 / (1 + mean(abs(value - healthy_median) / healthy_IQR)).")
    print("Anomaly component = 100 - 30 * anomaly_flag * clip(-anomaly_score / 0.15, 0, 1).")
    print("Persistence uses the current/previous 10-window anomaly ratio; smoothing is a per-recording 5-window median.")
    print()
    print("RECORDING HEALTH SUMMARY\n------------------------")
    print(summary.to_string(index=False))
    print()
    print("NORMAL_03 HEALTH CHECK\n----------------------")
    print(normal[["health_score", "health_status"]].describe().to_string())
    print(f"Healthy: {(normal['health_status'] == 'Healthy').mean() * 100:.4f}%")
    print(f"Warning: {(normal['health_status'] == 'Warning').mean() * 100:.4f}%")
    print(f"Degraded: {(normal['health_status'] == 'Degraded').mean() * 100:.4f}%")
    print(f"Critical: {(normal['health_status'] == 'Critical').mean() * 100:.4f}%")
    print()
    print("FAULT CONDITION COMPARISON\n--------------------------")
    print(fault_summary[["recording", "true_condition", "mean_health_score", "median_health_score"]].to_string(index=False))
    print()
    print("HEALTH SCORE DISTRIBUTION\n-------------------------")
    print(distributions.to_string())
    print()
    print("TEMPORAL HEALTH EVENTS\n----------------------")
    print(pd.DataFrame(events).to_string(index=False) if events else "No sustained degraded/critical events.")
    print()
    print("WORST HEALTH WINDOWS\n--------------------")
    for _, row in worst.iterrows():
        components = {name: row[name] for name in ["vibration_score", "temperature_score", "rpm_score", "anomaly_component", "persistence_score"]}
        contributors = ", ".join(sorted(components, key=components.get)[:2])
        print(f"{row['recording']} t={row['time_s']:.1f}s score={row['health_score']:.4f} status={row['health_status']} dominant={contributors}")
    print()
    print("WITHIN-RECORDING CONDITION CHANGE\n----------------------------------")
    print(summary[["recording", "early_health_score", "late_health_score", "health_change"]].to_string(index=False))
    print()
    print("SCIENTIFIC INTERPRETATION\n-------------------------")
    print("The health score is a data-driven condition indicator, not a probability, RUL estimate, or time-to-failure measure.")
    print("Startup and shutdown RPM behavior is labeled separately and receives an RPM score of 100 rather than a transient penalty.")
    print("Isolated anomaly flags have limited influence; repeated flags lower the 10-window persistence score and must persist before events are reported.")
    print("Healthy references use only normal_01.csv and normal_02.csv; normal_03.csv remains an unseen evaluation recording.")
    print(f"normal_03 shows a mean condition score of {normal['health_score'].mean():.4f}; this is a condition result, not proof of clean operation.")
    print("Misalignment shows lower condition scores than normal_03 in these recordings; imbalance is more mixed and overlaps healthy behavior.")
    print("Thresholds and weights are specific to this physical dataset and are not a standardized industrial index.")
    print("The results do not establish cross-machine generalization, unknown-fault detection, production readiness, RUL, or time-to-failure.")
    print("This layer may inform future work only after additional validation and operating-condition review.")
    print()
    print(f"Protected files unchanged during run: {all(digest(path) == value for path, value in before_hashes.items())}")
    print(f"Saved health results: {output_path.relative_to(root)}")
    print(f"Saved health summary: {summary_path.relative_to(root)}")
    print()
    print("============================================================")
    print("PHASE 10 COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    try:
        run()
    except HealthValidationError as error:
        print("PHASE 10 STOPPED")
        print("---------------")
        print(error)
        raise SystemExit(1)

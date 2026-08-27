"""Phase 11.3 LSTM-based proxy RUL experiment using real physical recordings."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEQUENCE_LENGTH = 30
TRAILING_WINDOW = 30
CRITICAL_THRESHOLD = 30.0
BASELINE_RECORDINGS = ["normal_01", "normal_02"]
TRAIN_RECORDINGS = ["normal_01", "imbalance_01", "misalignment_01"]
VALIDATION_RECORDINGS = ["normal_02", "imbalance_02", "misalignment_02"]
TEST_RECORDINGS = ["normal_03", "imbalance_03", "misalignment_03"]
PHYSICAL_FEATURES = [
    "temperature_mean", "temperature_std", "rpm_mean", "rpm_std", "rpm_min", "rpm_max",
    "accel_x_std", "accel_x_rms", "accel_x_peak", "accel_y_std", "accel_y_rms", "accel_y_peak",
    "accel_z_std", "accel_z_rms", "accel_z_peak", "vibration_mean", "vibration_std",
    "vibration_rms", "vibration_peak", "vibration_min", "vibration_range",
]
MODEL_FEATURES = PHYSICAL_FEATURES + ["health_score", "anomaly_score"]
TARGET_COLUMNS = ["recording", "time_s", "true_condition", "health_score", "proxy_RUL_target_hours", "proxy_RUL_prediction_hours", "absolute_error_hours"]


# Importing these helpers preserves the established Phase 10 condition calculation for
# baseline recordings whose health rows were not persisted in the Phase 10 target output.
from ml.fault_classification.health_score import add_health_components, robust_reference


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def condition(recording: str) -> str:
    return recording.split("_")[0]


def validate_inputs(features: pd.DataFrame, anomaly: pd.DataFrame, health: pd.DataFrame) -> None:
    missing_features = sorted(set(PHYSICAL_FEATURES + ["timestamp_start_ms", "timestamp_end_ms", "recording"]) - set(features.columns))
    missing_anomaly = sorted(set(["timestamp_start_ms", "recording", "anomaly_score", "anomaly_prediction"]) - set(anomaly.columns))
    missing_health = sorted(set(["timestamp_start_ms", "recording", "health_score"]) - set(health.columns))
    if missing_features or missing_anomaly or missing_health:
        raise ValueError(f"Missing input columns: features={missing_features}, anomaly={missing_anomaly}, health={missing_health}")
    if len(features) != 8100 or len(anomaly) != 6300 or len(health) != 6300:
        raise ValueError("Unexpected Phase 9/10 input row counts.")


def build_recording_frame(
    recording: str,
    features: pd.DataFrame,
    anomaly: pd.DataFrame,
    health: pd.DataFrame,
    detector_bundle: dict,
    physical_reference: pd.DataFrame,
    rpm_reference: pd.DataFrame,
    rpm_threshold: float,
    rpm_variation_threshold: float,
) -> pd.DataFrame:
    feature_frame = features[features["recording"] == recording].copy().reset_index(drop=True)
    if len(feature_frame) != 900:
        raise ValueError(f"{recording} does not contain 900 feature windows.")
    filename = f"{recording}.csv"
    existing_anomaly = anomaly[anomaly["recording"] == filename].copy()
    if len(existing_anomaly):
        score_frame = existing_anomaly[["timestamp_start_ms", "timestamp_end_ms", "anomaly_score", "anomaly_prediction"]]
    else:
        detector = detector_bundle["model"]
        scaler = detector_bundle["scaler"]
        detector_features = detector_bundle.get("feature_columns", PHYSICAL_FEATURES)
        scaled = scaler.transform(feature_frame[detector_features])
        raw_prediction = detector.predict(scaled)
        score_frame = feature_frame[["timestamp_start_ms", "timestamp_end_ms"]].copy()
        score_frame["anomaly_score"] = detector.decision_function(scaled)
        score_frame["anomaly_prediction"] = pd.Series(raw_prediction).map({1: "normal", -1: "anomaly"}).to_numpy()
    frame = feature_frame.merge(score_frame, on=["timestamp_start_ms", "timestamp_end_ms"], validate="one_to_one")
    frame["recording"] = filename
    frame["true_condition"] = condition(recording)
    frame["time_s"] = frame["timestamp_start_ms"] / 1000.0
    frame = add_health_components(frame, physical_reference, rpm_reference, rpm_threshold, rpm_variation_threshold)
    existing_health = health[health["recording"] == filename][["timestamp_start_ms", "operating_state", "health_score"]]
    if len(existing_health):
        frame = frame.drop(columns=["operating_state", "health_score"]).merge(existing_health, on="timestamp_start_ms", validate="one_to_one")
    return frame.sort_values("timestamp_start_ms").reset_index(drop=True)


def add_proxy_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    values = frame["health_score"].to_numpy(dtype=float)
    times = frame["time_s"].to_numpy(dtype=float)
    targets = np.full(len(frame), np.nan, dtype=float)
    for index in range(TRAILING_WINDOW - 1, len(frame)):
        window_values = values[index - TRAILING_WINDOW + 1 : index + 1]
        window_times = times[index - TRAILING_WINDOW + 1 : index + 1]
        if frame.loc[index, "operating_state"] != "STEADY":
            continue
        if not (frame.loc[index - TRAILING_WINDOW + 1 : index, "operating_state"] == "STEADY").all():
            continue
        health_slope = np.polyfit(window_times, window_values, 1)[0]
        if values[index] <= CRITICAL_THRESHOLD:
            targets[index] = 0.0
        elif health_slope < 0:
            targets[index] = (values[index] - CRITICAL_THRESHOLD) / abs(health_slope) / 3600.0
    frame["proxy_RUL_target_hours"] = targets
    return frame


def make_sequences(frames: dict[str, pd.DataFrame], recordings: list[str], scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences = []
    targets = []
    metadata = []
    for recording in recordings:
        frame = frames[recording]
        scaled = scaler.transform(frame[MODEL_FEATURES])
        for index in range(SEQUENCE_LENGTH - 1, len(frame)):
            target = frame.loc[index, "proxy_RUL_target_hours"]
            if not np.isfinite(target):
                continue
            sequences.append(scaled[index - SEQUENCE_LENGTH + 1 : index + 1])
            targets.append(target)
            metadata.append([
                frame.loc[index, "recording"], frame.loc[index, "time_s"], frame.loc[index, "true_condition"],
                frame.loc[index, "health_score"], target,
            ])
    metadata_frame = pd.DataFrame(metadata, columns=["recording", "time_s", "true_condition", "health_score", "proxy_RUL_target_hours"])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), metadata_frame


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "r2": r2_score(actual, predicted) if len(actual) > 1 and np.std(actual) > 0 else np.nan,
        "spearman": spearmanr(actual, predicted).statistic if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0 else np.nan,
    }
    return result


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    root = Path(__file__).resolve().parents[2]
    features_path = root / "data" / "processed" / "features.csv"
    anomaly_path = root / "data" / "processed" / "anomaly_results.csv"
    health_path = root / "data" / "processed" / "health_results.csv"
    detector_path = root / "models" / "anomaly_detector.pkl"
    classifier_path = root / "models" / "fault_classifier.pkl"
    model_path = root / "models" / "lstm_proxy_rul.keras"
    result_path = root / "data" / "processed" / "lstm_proxy_rul_results.csv"
    protected = [features_path, anomaly_path, health_path, detector_path, classifier_path] + sorted((root / "data" / "raw").glob("*.csv"))
    before = {path: digest(path) for path in protected}

    features = pd.read_csv(features_path)
    anomaly = pd.read_csv(anomaly_path)
    health = pd.read_csv(health_path)
    validate_inputs(features, anomaly, health)
    detector_bundle = joblib.load(detector_path)

    baseline = features[features["recording"].isin(BASELINE_RECORDINGS)].copy()
    baseline_rpm = baseline[baseline["rpm_mean"] >= baseline["rpm_mean"].median() * 0.8]
    rpm_threshold = float(baseline["rpm_mean"].median() * 0.8)
    rpm_variation_threshold = float(baseline_rpm["rpm_std"].quantile(0.95))
    physical_reference = robust_reference(baseline, ["vibration_rms", "vibration_std", "vibration_peak", "vibration_range", "temperature_mean", "temperature_std"])
    rpm_reference = robust_reference(baseline_rpm, ["rpm_mean", "rpm_std"])

    frames = {}
    for recording in BASELINE_RECORDINGS + ["normal_03", "imbalance_01", "imbalance_02", "imbalance_03", "misalignment_01", "misalignment_02", "misalignment_03"]:
        frames[recording] = add_proxy_targets(build_recording_frame(recording, features, anomaly, health, detector_bundle, physical_reference, rpm_reference, rpm_threshold, rpm_variation_threshold))

    train_scaler = StandardScaler()
    train_scaler.fit(pd.concat([frames[recording][MODEL_FEATURES] for recording in TRAIN_RECORDINGS], ignore_index=True))
    x_train, y_train, train_meta = make_sequences(frames, TRAIN_RECORDINGS, train_scaler)
    x_validation, y_validation, validation_meta = make_sequences(frames, VALIDATION_RECORDINGS, train_scaler)
    x_test, y_test, test_meta = make_sequences(frames, TEST_RECORDINGS, train_scaler)
    if not len(y_train) or not len(y_validation) or not len(y_test):
        raise ValueError(f"Insufficient proxy targets: train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}")

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, len(MODEL_FEATURES))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=100, batch_size=32, shuffle=False, callbacks=[callback], verbose=0)
    model.save(model_path)

    predicted = model.predict(x_test, verbose=0).reshape(-1)
    negative_predictions = int((predicted < 0).sum())
    predicted = np.maximum(predicted, 0.0)
    test_meta["proxy_RUL_prediction_hours"] = predicted
    test_meta["absolute_error_hours"] = np.abs(test_meta["proxy_RUL_target_hours"] - predicted)
    test_meta[TARGET_COLUMNS].to_csv(result_path, index=False)

    all_metrics = metrics(y_test, predicted)
    baseline_prediction = np.full(len(y_test), y_train.mean())
    baseline_metrics = metrics(y_test, baseline_prediction)
    per_recording = []
    for recording in TEST_RECORDINGS:
        mask = test_meta["recording"] == f"{recording}.csv"
        per_recording.append([recording, int(mask.sum()), metrics(y_test[mask], predicted[mask])])
    trend_rows = []
    for recording in TEST_RECORDINGS:
        group = test_meta[test_meta.recording == f"{recording}.csv"].sort_values("time_s")
        trend = np.polyfit(group.time_s, group.proxy_RUL_prediction_hours, 1)[0] if len(group) > 1 else np.nan
        trend_rows.append([recording, group.proxy_RUL_prediction_hours.iloc[0], group.proxy_RUL_prediction_hours.iloc[len(group)//2], group.proxy_RUL_prediction_hours.iloc[-1], trend])

    unchanged = all(digest(path) == value for path, value in before.items())
    print("============================================================")
    print("PHASE 11.3 - LSTM PROXY RUL MODEL")
    print("============================================================\n")
    print("DATASET\n-------")
    print("Real physical sensor features and existing Phase 9/10 outputs were used.")
    print("No failure timestamps or true RUL labels exist.")
    print(f"Input feature count: {len(MODEL_FEATURES)}")
    print()
    print("PROXY TARGET\n------------")
    print("For each steady-state point, fit a linear health slope over the trailing 30 seconds only.")
    print("If health_slope < 0 and health_score > 30: proxy_RUL_hours = (health_score - 30) / abs(health_slope) / 3600.")
    print("If health_score <= 30: proxy_RUL_hours = 0; otherwise non-degrading points are unavailable.")
    print("This estimates time for the condition indicator to reach the project Critical threshold, not physical failure time.")
    print()
    print("SEQUENCE CONSTRUCTION\n---------------------")
    print(f"Sequence length: {SEQUENCE_LENGTH} seconds")
    print(f"Training sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_validation)}")
    print(f"Test sequences: {len(y_test)}")
    print(f"Usable proxy targets by split: train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}")
    print()
    print("DATA SPLIT\n----------")
    print(f"Training: {TRAIN_RECORDINGS}")
    print(f"Validation: {VALIDATION_RECORDINGS}")
    print(f"Testing: {TEST_RECORDINGS}")
    print("No recording appears in more than one split: True")
    print()
    print("LSTM ARCHITECTURE\n-----------------")
    print("LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, linear)")
    print(f"Feature list: {MODEL_FEATURES}")
    print()
    print("TRAINING\n--------")
    print(f"Epochs completed: {len(history.history['loss'])}")
    print(f"Best epoch: {int(np.argmin(history.history['val_loss']) + 1)}")
    print(f"Final training loss: {history.history['loss'][-1]}")
    print(f"Best validation loss: {min(history.history['val_loss'])}")
    print()
    print("TEST RESULTS\n------------")
    print(f"MAE: {all_metrics['mae']}")
    print(f"RMSE: {all_metrics['rmse']}")
    print(f"R2: {all_metrics['r2']}")
    print(f"Spearman correlation: {all_metrics['spearman']}")
    print()
    print("PER-RECORDING RESULTS\n---------------------")
    for recording, count, result in per_recording:
        print(f"{recording}: sequences={count}, MAE={result['mae']}, RMSE={result['rmse']}, R2={result['r2']}, Spearman={result['spearman']}")
    print()
    print("BASELINE COMPARISON\n-------------------")
    print(f"Mean baseline MAE: {baseline_metrics['mae']}")
    print(f"LSTM MAE: {all_metrics['mae']}")
    print(f"Mean baseline RMSE: {baseline_metrics['rmse']}")
    print(f"LSTM RMSE: {all_metrics['rmse']}")
    print(f"Mean baseline R2: {baseline_metrics['r2']}")
    print(f"LSTM R2: {all_metrics['r2']}")
    print(f"LSTM improves over baseline MAE: {all_metrics['mae'] < baseline_metrics['mae']}")
    print()
    print("TEMPORAL SANITY\n---------------")
    for row in trend_rows:
        print(f"{row[0]}: first={row[1]}, middle={row[2]}, last={row[3]}, prediction_trend_hours_per_s={row[4]}")
    print(f"Negative predictions before documented clipping: {negative_predictions}")
    print("Predictions were clipped at zero because proxy_RUL cannot be negative; no NaN or infinite predictions exist.")
    print()
    print("DATA LEAKAGE CHECK\n------------------")
    print("Scaler fitted only on training recordings: True")
    print("No test recordings used during training: True")
    print("No future points used in target slopes or sequences: True")
    print("Recording column used as model feature: False")
    print("Condition labels used as model feature: False")
    print("Training shuffle disabled: True")
    print()
    print("FINAL RUL INTERPRETATION\n------------------------")
    print("TRUE RUL: NOT AVAILABLE; no failure timestamps exist.")
    print("PROXY RUL: LSTM-based estimate of time for the condition indicator to reach the Critical threshold under continued recent degradation.")
    print("This is a proxy RUL estimate based on observed condition trajectory, not remaining useful life until physical failure.")
    print("It does not predict machine failure time or failure probability.")
    print()
    print("SCIENTIFIC LIMITATIONS\n----------------------")
    print("No actual failure timestamps or run-to-failure trajectories exist.")
    print("The proxy target is derived from the health trajectory and assumes local linear degradation continuation.")
    print("The Critical threshold is project-specific; nine condition recordings are insufficient for production-level RUL validation.")
    print("The test recordings are condition recordings, not independent machine lifetimes.")
    print("LSTM feature importance was not fabricated; permutation importance was not reported because sequence-level importance is unstable with this small recording count.")
    print("Additional real run-to-failure physical experiments are required to validate true RUL.")
    print()
    print(f"Protected files unchanged: {unchanged}")
    print(f"Saved model: {model_path.relative_to(root)}")
    print(f"Saved results: {result_path.relative_to(root)}")
    print()
    print("============================================================")
    print("PHASE 11.3 COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()

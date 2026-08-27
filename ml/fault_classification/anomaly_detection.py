"""Phase 9 unsupervised anomaly detection using a healthy physical baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, RAW_COLUMNS, process_recording

BASELINE_RECORDINGS = ["normal_01.csv", "normal_02.csv"]
TARGET_RECORDINGS = [
    "normal_03.csv",
    "imbalance_01.csv",
    "imbalance_02.csv",
    "imbalance_03.csv",
    "misalignment_01.csv",
    "misalignment_02.csv",
    "misalignment_03.csv",
]
EXPECTED_RECORDINGS = sorted(BASELINE_RECORDINGS + TARGET_RECORDINGS)
EXPECTED_SAMPLES_PER_RECORDING = 90000
EXPECTED_WINDOWS_PER_RECORDING = 900
SAMPLE_RATE_HZ = 1000
CLUSTER_GAP_S = 10.0
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS[:-3]


class AnalysisValidationError(Exception):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_condition(recording: str) -> str:
    stem = Path(recording).stem
    for condition in ("normal", "imbalance", "misalignment"):
        if stem.startswith(f"{condition}_"):
            return condition
    raise AnalysisValidationError(f"Cannot derive condition from recording: {recording}")


def validate_raw_recordings(raw_dir: Path) -> dict[str, str]:
    actual = sorted(path.name for path in raw_dir.glob("*.csv"))
    if actual != EXPECTED_RECORDINGS:
        raise AnalysisValidationError(
            f"Expected raw recordings {EXPECTED_RECORDINGS}, found {actual}."
        )

    digests = {}
    for recording in EXPECTED_RECORDINGS:
        path = raw_dir / recording
        frame = pd.read_csv(path)
        missing = [column for column in RAW_COLUMNS if column not in frame.columns]
        if missing:
            raise AnalysisValidationError(f"{recording} is missing columns: {missing}")
        if len(frame) != EXPECTED_SAMPLES_PER_RECORDING:
            raise AnalysisValidationError(
                f"{recording} has {len(frame)} samples, expected {EXPECTED_SAMPLES_PER_RECORDING}."
            )
        digests[recording] = file_digest(path)
    return digests


def load_recordings(raw_dir: Path, recordings: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    counts = {}
    for recording in recordings:
        features, sample_count, window_count = process_recording(raw_dir / recording)
        if sample_count != EXPECTED_SAMPLES_PER_RECORDING or window_count != EXPECTED_WINDOWS_PER_RECORDING:
            raise AnalysisValidationError(f"Unexpected feature count for {recording}.")
        frames.append(features)
        counts[recording] = window_count
    return pd.concat(frames, ignore_index=True), counts


def build_clusters(anomalies: pd.DataFrame) -> pd.DataFrame:
    clusters = []
    for recording, group in anomalies.groupby("recording", sort=False):
        ordered = group.sort_values("timestamp_start_ms")
        current = None
        for _, row in ordered.iterrows():
            time_s = float(row["time_s"])
            if current is None or time_s - current["cluster_end_s"] > CLUSTER_GAP_S:
                if current is not None:
                    clusters.append(current)
                current = {
                    "recording": recording,
                    "cluster_start_s": time_s,
                    "cluster_end_s": time_s,
                    "number_of_anomalous_windows": 1,
                }
            else:
                current["cluster_end_s"] = time_s
                current["number_of_anomalous_windows"] += 1
        if current is not None:
            clusters.append(current)
    return pd.DataFrame(
        clusters,
        columns=[
            "recording",
            "cluster_start_s",
            "cluster_end_s",
            "number_of_anomalous_windows",
        ],
    )


def score_summary(recording: str, frame: pd.DataFrame) -> dict:
    anomaly_mask = frame["anomaly_prediction"] == "anomaly"
    scores = frame["anomaly_score"]
    return {
        "recording": recording,
        "true_condition": true_condition(recording),
        "windows": int(len(frame)),
        "anomalies": int(anomaly_mask.sum()),
        "normal_predictions": int((~anomaly_mask).sum()),
        "anomaly_rate": float(anomaly_mask.mean()),
        "mean_anomaly_score": float(scores.mean()),
        "median_anomaly_score": float(scores.median()),
        "minimum_anomaly_score": float(scores.min()),
        "maximum_anomaly_score": float(scores.max()),
        "mean_score_predicted_anomaly": float(scores[anomaly_mask].mean())
        if anomaly_mask.any()
        else float("nan"),
        "mean_score_predicted_normal": float(scores[~anomaly_mask].mean())
        if (~anomaly_mask).any()
        else float("nan"),
    }


def run() -> None:
    root = project_root()
    raw_dir = root / "data" / "raw"
    results_path = root / "data" / "processed" / "anomaly_results.csv"
    summary_path = root / "data" / "processed" / "anomaly_summary.csv"
    model_path = root / "models" / "anomaly_detector.pkl"
    classifier_path = root / "models" / "fault_classifier.pkl"

    if not classifier_path.exists():
        raise AnalysisValidationError(f"Existing Phase 7 model does not exist: {classifier_path}")
    raw_digests = validate_raw_recordings(raw_dir)
    classifier_digest = file_digest(classifier_path)

    baseline, baseline_counts = load_recordings(raw_dir, BASELINE_RECORDINGS)
    targets, target_counts = load_recordings(raw_dir, TARGET_RECORDINGS)

    scaler = StandardScaler()
    healthy_scaled = scaler.fit_transform(baseline[MODEL_FEATURE_COLUMNS])
    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(healthy_scaled)

    target_scaled = scaler.transform(targets[MODEL_FEATURE_COLUMNS])
    scores = detector.decision_function(target_scaled)
    predictions = detector.predict(target_scaled)
    results = targets[["timestamp_start_ms", "timestamp_end_ms", "recording"]].copy()
    results["recording"] = results["recording"].map(lambda recording: f"{recording}.csv")
    results["time_s"] = results["timestamp_start_ms"] / SAMPLE_RATE_HZ
    results["true_condition"] = results["recording"].map(true_condition)
    results["anomaly_score"] = scores
    results["anomaly_prediction"] = pd.Series(predictions, index=results.index).map(
        {1: "normal", -1: "anomaly"}
    )

    summary = pd.DataFrame(
        [score_summary(recording, results.loc[results["recording"] == recording]) for recording in TARGET_RECORDINGS]
    )
    anomaly_rows = results.loc[results["anomaly_prediction"] == "anomaly"]
    clusters = build_clusters(anomaly_rows)

    normal_mask = results["true_condition"] == "normal"
    fault_mask = ~normal_mask
    predicted_anomaly = results["anomaly_prediction"] == "anomaly"
    true_values = fault_mask.astype(int)
    predicted_values = predicted_anomaly.astype(int)
    tn, fp, fn, tp = confusion_matrix(true_values, predicted_values, labels=[0, 1]).ravel()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    joblib.dump(
        {
            "model": detector,
            "scaler": scaler,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "baseline_recordings": BASELINE_RECORDINGS,
        },
        model_path,
    )

    for recording, digest in raw_digests.items():
        if file_digest(raw_dir / recording) != digest:
            raise AnalysisValidationError(f"Raw file was modified: {recording}")
    if file_digest(classifier_path) != classifier_digest:
        raise AnalysisValidationError("fault_classifier.pkl was modified.")

    print("============================================================")
    print("PHASE 9 - UNSUPERVISED ANOMALY DETECTION")
    print("============================================================")
    print()
    print("HEALTHY BASELINE")
    print("----------------")
    print()
    for recording in BASELINE_RECORDINGS:
        print(f"{recording}:")
        print(f"{baseline_counts[recording]} windows")
        print()
    print("Total baseline windows:")
    print(len(baseline))
    print()
    print("TARGET DATA")
    print("-----------")
    print()
    for recording in TARGET_RECORDINGS:
        print(f"{recording}:")
        print(f"{target_counts[recording]} windows")
        print()
    print("Total target windows:")
    print(len(targets))
    print()
    print("ANOMALY MODEL")
    print("-------------")
    print()
    print("Model:")
    print("IsolationForest")
    print("Estimators:")
    print("300")
    print("Baseline recordings:")
    print(len(BASELINE_RECORDINGS))
    print("Baseline windows:")
    print(len(baseline))
    print()
    print("RECORDING-LEVEL RESULTS")
    print("-----------------------")
    print()
    print(summary.to_string(index=False))
    print()
    print("NORMAL BASELINE CHECK")
    print("---------------------")
    print()
    normal_summary = summary.loc[summary["recording"] == "normal_03.csv"].iloc[0]
    print("normal_03.csv")
    print()
    print(f"Anomaly rate: {normal_summary['anomaly_rate']}")
    print()
    print("FAULT RECORDING RESULTS")
    print("-----------------------")
    print()
    print(summary.loc[summary["true_condition"] != "normal"].to_string(index=False))
    print()
    print("ANOMALY SCORE ANALYSIS")
    print("----------------------")
    print()
    print(
        summary[
            [
                "recording",
                "mean_anomaly_score",
                "median_anomaly_score",
                "minimum_anomaly_score",
                "maximum_anomaly_score",
                "mean_score_predicted_anomaly",
                "mean_score_predicted_normal",
            ]
        ].to_string(index=False)
    )
    print()
    print("TEMPORAL ANOMALY CLUSTERS")
    print("-------------------------")
    print()
    for recording in TARGET_RECORDINGS:
        recording_anomalies = anomaly_rows.loc[anomaly_rows["recording"] == recording]
        print(f"{recording} anomalous timestamps (s):")
        print(", ".join(f"{value:.1f}" for value in recording_anomalies["time_s"]))
        print()
    if clusters.empty:
        print("No temporal anomaly clusters.")
    else:
        print(clusters.to_string(index=False))
    print()
    print("WINDOW-LEVEL EVALUATION")
    print("-----------------------")
    print()
    print(f"True normal windows: {int(normal_mask.sum())}")
    print(f"True fault windows: {int(fault_mask.sum())}")
    print(f"True positives: {int(tp)}")
    print(f"True negatives: {int(tn)}")
    print(f"False positives: {int(fp)}")
    print(f"False negatives: {int(fn)}")
    print(f"Accuracy: {accuracy_score(true_values, predicted_values)}")
    print(f"Precision: {precision_score(true_values, predicted_values, zero_division=0)}")
    print(f"Recall: {recall_score(true_values, predicted_values, zero_division=0)}")
    print(f"F1-score: {f1_score(true_values, predicted_values, zero_division=0)}")
    print()
    print("CONFUSION MATRIX")
    print("----------------")
    print("rows = true [normal, fault], columns = predicted [normal, anomaly]")
    print(confusion_matrix(true_values, predicted_values, labels=[0, 1]))
    print()
    print("OUTPUT")
    print("------")
    print(f"Saved anomaly results: {results_path.relative_to(root)}")
    print(f"Saved anomaly summary: {summary_path.relative_to(root)}")
    print(f"Saved model: {model_path.relative_to(root)}")
    print()
    print("============================================================")
    print("PHASE 9 COMPLETE")
    print("============================================================")
    print()
    print("SCIENTIFIC INTERPRETATION")
    print("=========================")
    print("- The anomaly detector was trained only on healthy physical recordings.")
    print("- The baseline consisted of normal_01.csv and normal_02.csv.")
    print("- normal_03.csv was treated as unseen healthy operation.")
    print("- Fault recordings were not used for fitting.")
    print("- Fault recordings were used only for evaluation.")
    print("- The anomaly score is not a probability.")
    print("- The evaluation covers only the available physical conditions.")
    print("- The detector does not establish detection of arbitrary unknown failures.")
    print("- The anomaly detector is complementary to the supervised fault classifier.")


if __name__ == "__main__":
    try:
        run()
    except AnalysisValidationError as error:
        print("PHASE 9 STOPPED")
        print("---------------")
        print(error)
        raise SystemExit(1)

"""Unsupervised anomaly detection for CWRU 12 kHz Drive-End bearing vibration."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ml.fault_classification.cwru_xgboost import (
    CLASS_NAMES,
    SAMPLE_RATE_HZ,
    WINDOW_SIZE,
    build_window_table,
    discover_signals,
)

RANDOM_STATE = 42

# Source-signal assignment for Normal baseline
# Fit only on normal operating loads 0, 1, 2; hold out load 3 as unseen healthy operation
TRAIN_NORMAL_SOURCE_FILES = [
    "data/external/CWRU-dataset/Normal/97_Normal_0.mat",
    "data/external/CWRU-dataset/Normal/98_Normal_1.mat",
    "data/external/CWRU-dataset/Normal/99_Normal_2.mat",
]
HOLDOUT_NORMAL_SOURCE_FILES = [
    "data/external/CWRU-dataset/Normal/100_Normal_3.mat",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_anomaly_detection(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Evaluate binary anomaly detection metrics.
    
    y_true: 0 for Normal, 1 for Anomaly/Fault
    y_pred: 0 for Normal, 1 for Anomaly/Fault
    scores: continuous anomaly score (higher = more anomalous)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main() -> None:
    root = project_root()
    cwru_root = root / "data" / "external" / "CWRU-dataset"
    model_path = root / "models" / "cwru_anomaly_detector.pkl"
    results_path = root / "data" / "processed" / "cwru_anomaly_results.csv"
    eval_path = root / "data" / "processed" / "cwru_anomaly_evaluation.csv"

    # Pre-run SHA256 snapshot
    source_paths = sorted(
        list((cwru_root / "Normal").rglob("*.mat"))
        + list((cwru_root / "12k_Drive_End_Bearing_Fault_Data").rglob("*.mat"))
    )
    before_hashes = {path: digest(path) for path in source_paths}

    # Discover all DE signals
    signals = discover_signals(cwru_root)
    if len(signals) != 65:
        raise ValueError(f"Expected 65 DE source signals, found {len(signals)}.")

    table = build_window_table(signals)
    metadata_columns = [
        "source_file",
        "source_signal",
        "class",
        "severity",
        "load",
        "outer_race_position",
        "window_index",
    ]
    feature_columns = [column for column in table.columns if column not in metadata_columns]

    # Source-signal partitioning
    normal_signals = [s for s in signals if s[2]["class"] == "NORMAL"]
    fault_signals = [s for s in signals if s[2]["class"] != "NORMAL"]

    train_normal_signals = [
        s for s in normal_signals if s[2]["source_file"] in TRAIN_NORMAL_SOURCE_FILES
    ]
    holdout_normal_signals = [
        s for s in normal_signals if s[2]["source_file"] in HOLDOUT_NORMAL_SOURCE_FILES
    ]

    train_source_keys = {f"{s[0]}::{s[1]}" for s in train_normal_signals}
    holdout_normal_source_keys = {f"{s[0]}::{s[1]}" for s in holdout_normal_signals}
    eval_fault_source_keys = {f"{s[0]}::{s[1]}" for s in fault_signals}
    eval_source_keys = holdout_normal_source_keys | eval_fault_source_keys

    # Zero leakage check
    if train_source_keys & eval_source_keys:
        raise ValueError("Source signal leakage detected between training and evaluation.")

    # Assign splits in table
    def get_split(row) -> str:
        key = f"{root / row.source_file}::{row.source_signal}"
        if key in train_source_keys:
            return "fit_normal"
        elif key in holdout_normal_source_keys:
            return "eval_holdout_normal"
        elif key in eval_fault_source_keys:
            return "eval_fault"
        else:
            raise ValueError(f"Unassigned source signal: {key}")

    table["split"] = [get_split(row) for row in table.itertuples()]

    train_df = table[table["split"] == "fit_normal"].copy().reset_index(drop=True)
    eval_df = table[table["split"].isin(["eval_holdout_normal", "eval_fault"])].copy().reset_index(drop=True)

    # Standardize features using ONLY healthy training baseline
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_eval = scaler.transform(eval_df[feature_columns])

    # Fit Isolation Forest strictly on healthy normal vibration
    detector = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    detector.fit(x_train)

    # Prediction: decision_function returns negative for anomalies, positive for normal
    raw_decision = detector.decision_function(x_eval)
    # Convert score so that higher = more anomalous
    anomaly_scores = -raw_decision
    raw_preds = detector.predict(x_eval)  # +1 = inlier/normal, -1 = outlier/anomaly
    predicted_anomaly = (raw_preds == -1).astype(int)  # 1 = anomaly, 0 = normal
    true_anomaly = (eval_df["class"] != "NORMAL").astype(int)  # 1 = anomaly, 0 = normal

    # Window-level metrics
    window_metrics = evaluate_anomaly_detection(true_anomaly.to_numpy(), predicted_anomaly, anomaly_scores)

    # Save detailed prediction results
    results = eval_df[metadata_columns].copy()
    results["anomaly_score"] = anomaly_scores
    results["raw_decision_score"] = raw_decision
    results["predicted_label"] = np.where(predicted_anomaly == 1, "ANOMALY", "NORMAL")
    results["true_anomaly_label"] = np.where(true_anomaly == 1, "ANOMALY", "NORMAL")
    results["is_correct"] = results["predicted_label"] == results["true_anomaly_label"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    # Source-level majority voting evaluation
    source_results = results.groupby(["source_file", "source_signal", "class", "severity", "load"], as_index=False).agg(
        true_anomaly_label=("true_anomaly_label", "first"),
        predicted_anomaly_fraction=("predicted_label", lambda v: float((v == "ANOMALY").mean())),
        mean_anomaly_score=("anomaly_score", "mean"),
    )
    source_results["predicted_label"] = np.where(source_results["predicted_anomaly_fraction"] >= 0.5, "ANOMALY", "NORMAL")
    source_results["is_correct"] = source_results["predicted_label"] == source_results["true_anomaly_label"]

    source_true = (source_results["true_anomaly_label"] == "ANOMALY").astype(int).to_numpy()
    source_pred = (source_results["predicted_label"] == "ANOMALY").astype(int).to_numpy()
    source_scores = source_results["mean_anomaly_score"].to_numpy()
    source_metrics = evaluate_anomaly_detection(source_true, source_pred, source_scores)

    # Per-class detection rates
    per_class_summary = {}
    for class_name in CLASS_NAMES:
        class_mask = eval_df["class"] == class_name
        if not class_mask.any():
            continue
        total_w = int(class_mask.sum())
        if class_name == "NORMAL":
            correct_w = int((predicted_anomaly[class_mask] == 0).sum())
            rate = correct_w / total_w
            per_class_summary[class_name] = {
                "windows": total_w,
                "correct": correct_w,
                "rate": rate,
                "rate_name": "Specificity (Normal Inlier Rate)",
            }
        else:
            correct_w = int((predicted_anomaly[class_mask] == 1).sum())
            rate = correct_w / total_w
            per_class_summary[class_name] = {
                "windows": total_w,
                "correct": correct_w,
                "rate": rate,
                "rate_name": "Fault Recall (Anomaly Detection Rate)",
            }

    # Save evaluation summary
    eval_rows = []
    for k, v in window_metrics.items():
        eval_rows.append({"level": "window", "metric": k, "value": v})
    for k, v in source_metrics.items():
        eval_rows.append({"level": "source", "metric": k, "value": v})
    for c_name, c_data in per_class_summary.items():
        eval_rows.append({"level": f"class_{c_name.lower()}", "metric": c_data["rate_name"], "value": c_data["rate"]})
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False)

    # Save model bundle
    bundle = {
        "model": detector,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "dataset": "CWRU 12 kHz Drive-End Bearing Dataset",
        "sampling_rate_hz": SAMPLE_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "training_source_signals": sorted(list(train_source_keys)),
        "holdout_normal_source_signals": sorted(list(holdout_normal_source_keys)),
        "evaluation_fault_source_signals": sorted(list(eval_fault_source_keys)),
        "contamination": 0.01,
        "random_state": RANDOM_STATE,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    # Verify CWRU raw data integrity
    unchanged = all(digest(path) == value for path, value in before_hashes.items())

    # Formatted Report
    print("============================================================")
    print("PHASE 20A — CWRU ANOMALY DETECTION (ISOLATION FOREST)")
    print("============================================================")
    print(f"Dataset: CWRU 12 kHz Drive-End Bearing Dataset")
    print(f"Total source signals: {len(signals)}")
    print(f"Healthy baseline fit signals: {len(train_normal_signals)}")
    print(f"Healthy holdout eval signals: {len(holdout_normal_signals)}")
    print(f"Evaluation fault signals: {len(fault_signals)}")
    print()
    print("SOURCE SIGNAL ASSIGNMENTS:")
    print("  Fitting Signals (Healthy Normal Only):")
    for s in train_normal_signals:
        print(f"    {s[0].name} :: {s[1]} (Load {s[2]['load']} HP)")
    print("  Held-out Healthy Evaluation Signal:")
    for s in holdout_normal_signals:
        print(f"    {s[0].name} :: {s[1]} (Load {s[2]['load']} HP)")
    print()
    print(f"Training windows (Healthy fit): {len(train_df)}")
    print(f"Evaluation windows: {len(eval_df)} ({len(eval_df[eval_df['class'] == 'NORMAL'])} Normal + {len(eval_df[eval_df['class'] != 'NORMAL'])} Fault)")
    print()
    print("WINDOW-LEVEL METRICS:")
    print(f"  Accuracy:             {window_metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy:    {window_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Precision (Anomaly):  {window_metrics['precision'] * 100:.2f}%")
    print(f"  Recall (Fault Det.):  {window_metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:             {window_metrics['f1'] * 100:.2f}%")
    print(f"  Specificity (Normal): {window_metrics['specificity'] * 100:.2f}%")
    print(f"  False Positive Rate:  {window_metrics['false_positive_rate'] * 100:.2f}%")
    print(f"  ROC-AUC:              {window_metrics['roc_auc']:.6f}")
    print(f"  PR-AUC:               {window_metrics['pr_auc']:.6f}")
    print()
    print("WINDOW CONFUSION MATRIX (rows = True [Normal, Anomaly], cols = Predicted [Normal, Anomaly]):")
    print(f"  [TN: {window_metrics['true_negatives']:<4}  FP: {window_metrics['false_positives']:<4}]")
    print(f"  [FN: {window_metrics['false_negatives']:<4}  TP: {window_metrics['true_positives']:<4}]")
    print()
    print("PER-CLASS EVALUATION:")
    for c_name, c_data in per_class_summary.items():
        print(f"  {c_name:<12} : {c_data['rate_name']} = {c_data['rate'] * 100:.2f}% ({c_data['correct']}/{c_data['windows']} windows)")
    print()
    print("SOURCE-LEVEL EVALUATION (Majority Voting):")
    print(f"  Source Accuracy:          {source_metrics['accuracy'] * 100:.2f}% ({int(source_results['is_correct'].sum())}/{len(source_results)} sources)")
    print(f"  Source Specificity:       {source_metrics['specificity'] * 100:.2f}% ({int(source_metrics['true_negatives'])}/{int(source_metrics['true_negatives'] + source_metrics['false_positives'])} sources)")
    print(f"  Source Fault Recall:      {source_metrics['recall'] * 100:.2f}% ({int(source_metrics['true_positives'])}/{int(source_metrics['true_positives'] + source_metrics['false_negatives'])} sources)")
    print()
    print("LEAKAGE & SCIENTIFIC AUDIT:")
    print(f"  Source Signal Leakage Check: {'PASS (0 overlap)' if not (train_source_keys & eval_source_keys) else 'FAIL'}")
    print(f"  Fault Signal Contamination in Fit: None (0 fault signals used in fit)")
    print(f"  CWRU Source Raw Data Integrity: {'PASS' if unchanged else 'FAIL'}")
    print(f"  Model Saved: {model_path.relative_to(root)}")
    print(f"  Results Saved: {results_path.relative_to(root)}")
    print(f"  Evaluation Saved: {eval_path.relative_to(root)}")
    print("============================================================")
    print("PHASE 20A COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()

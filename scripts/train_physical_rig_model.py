"""Physical-rig feature extraction and grouped 3-fold cross-validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

SAMPLE_RATE_HZ = 100
WINDOW_SIZE = 100  # 1-second window at 100 Hz

CLASS_MAP = {
    "normal": "NORMAL",
    "imbalance": "IMBALANCE",
    "misalignment": "MISALIGNMENT",
}
LABEL_ENCODING = {"NORMAL": 0, "IMBALANCE": 1, "MISALIGNMENT": 2}
LABEL_DECODING = {0: "NORMAL", 1: "IMBALANCE", 2: "MISALIGNMENT"}
CLASS_ORDER = ["NORMAL", "IMBALANCE", "MISALIGNMENT"]


def get_condition_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "normal" in lower:
        return "NORMAL"
    elif "imbalance" in lower:
        return "IMBALANCE"
    elif "misalignment" in lower:
        return "MISALIGNMENT"
    raise ValueError(f"Unknown condition for file: {filename}")


def compute_rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr))))


def compute_crest_factor(arr: np.ndarray) -> float:
    r = compute_rms(arr)
    if r < 1e-9:
        return 0.0
    return float(np.max(np.abs(arr)) / r)


def compute_spectral_features(arr: np.ndarray, fs: int = 100) -> tuple[float, float, float, float]:
    freqs, psd = welch(arr, fs=fs, nperseg=min(len(arr), 64))
    total_energy = float(np.sum(psd))
    if total_energy < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    spectral_rms = float(np.sqrt(np.mean(psd)))
    dominant_freq = float(freqs[np.argmax(psd)])
    spectral_centroid = float(np.sum(freqs * psd) / total_energy)
    return spectral_rms, dominant_freq, spectral_centroid, total_energy


def extract_features_from_recording(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    condition = get_condition_from_filename(csv_path.name)
    recording_name = csv_path.stem

    # Raw summary stats
    duration_s = (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0 if len(df) > 1 else 0
    raw_stats = {
        "recording": recording_name,
        "filename": csv_path.name,
        "condition": condition,
        "samples": len(df),
        "duration_seconds": duration_s,
        "sample_rate_hz": float(len(df) / duration_s) if duration_s > 0 else SAMPLE_RATE_HZ,
        "temp_mean": float(df["temperature_C"].mean()),
        "rpm_mean": float(df["rpm"].mean()),
        "vibration_mean_rms": float(np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2).mean()),
    }

    n_samples = len(df)
    usable_samples = (n_samples // WINDOW_SIZE) * WINDOW_SIZE
    df_usable = df.iloc[:usable_samples].reset_index(drop=True)

    rows = []
    for i in range(0, usable_samples, WINDOW_SIZE):
        chunk = df_usable.iloc[i : i + WINDOW_SIZE]
        
        ax = chunk["accel_x"].to_numpy(dtype=np.float64)
        ay = chunk["accel_y"].to_numpy(dtype=np.float64)
        az = chunk["accel_z"].to_numpy(dtype=np.float64)
        vm = np.sqrt(ax**2 + ay**2 + az**2)
        temp = chunk["temperature_C"].to_numpy(dtype=np.float64)
        rpm = chunk["rpm"].to_numpy(dtype=np.float64)

        # Spectral on vibration magnitude
        spec_rms, dom_freq, spec_centroid, spec_energy = compute_spectral_features(vm, fs=SAMPLE_RATE_HZ)

        row = {
            "recording": recording_name,
            "condition": condition,
            "window_index": i // WINDOW_SIZE,
            "timestamp_start_ms": float(chunk["timestamp_ms"].iloc[0]),
            "timestamp_end_ms": float(chunk["timestamp_ms"].iloc[-1]),
            
            # Accelerometer X
            "accel_x_mean": float(np.mean(ax)),
            "accel_x_std": float(np.std(ax, ddof=0)),
            "accel_x_rms": compute_rms(ax),
            "accel_x_peak": float(np.max(np.abs(ax))),
            "accel_x_p2p": float(np.ptp(ax)),
            "accel_x_crest": compute_crest_factor(ax),

            # Accelerometer Y
            "accel_y_mean": float(np.mean(ay)),
            "accel_y_std": float(np.std(ay, ddof=0)),
            "accel_y_rms": compute_rms(ay),
            "accel_y_peak": float(np.max(np.abs(ay))),
            "accel_y_p2p": float(np.ptp(ay)),
            "accel_y_crest": compute_crest_factor(ay),

            # Accelerometer Z
            "accel_z_mean": float(np.mean(az)),
            "accel_z_std": float(np.std(az, ddof=0)),
            "accel_z_rms": compute_rms(az),
            "accel_z_peak": float(np.max(np.abs(az))),
            "accel_z_p2p": float(np.ptp(az)),
            "accel_z_crest": compute_crest_factor(az),

            # Combined Vibration Magnitude
            "vibration_mean": float(np.mean(vm)),
            "vibration_std": float(np.std(vm, ddof=0)),
            "vibration_rms": compute_rms(vm),
            "vibration_peak": float(np.max(vm)),
            "vibration_p2p": float(np.ptp(vm)),
            "vibration_crest": compute_crest_factor(vm),

            # Spectral features
            "spectral_rms": spec_rms,
            "dominant_freq": dom_freq,
            "spectral_centroid": spec_centroid,
            "spectral_energy": spec_energy,

            # Temperature & RPM
            "temperature_mean": float(np.mean(temp)),
            "temperature_std": float(np.std(temp, ddof=0)),
            "rpm_mean": float(np.mean(rpm)),
            "rpm_std": float(np.std(rpm, ddof=0)),
        }
        rows.append(row)

    return pd.DataFrame(rows), raw_stats


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if len(csv_files) != 9:
        print(f"Warning: Found {len(csv_files)} files in {RAW_DIR} (expected 9).")

    all_windows = []
    raw_summaries = []

    print(f"Processing {len(csv_files)} raw recordings...")
    for f in csv_files:
        w_df, r_stats = extract_features_from_recording(f)
        all_windows.append(w_df)
        raw_summaries.append(r_stats)
        print(f"  {f.name:<22}: {r_stats['samples']:,} samples -> {len(w_df)} windows ({r_stats['condition']})")

    df_dataset = pd.concat(all_windows, ignore_index=True)
    df_raw = pd.DataFrame(raw_summaries)
    df_raw.to_csv(PROCESSED_DIR / "physical_rig_raw_summary.csv", index=False)
    print(f"\nTotal Dataset: {len(df_dataset)} windows across {len(df_raw)} recordings.")

    # Define feature columns
    meta_cols = ["recording", "condition", "window_index", "timestamp_start_ms", "timestamp_end_ms"]
    feature_cols = [c for c in df_dataset.columns if c not in meta_cols]

    # Save feature names
    with open(PROCESSED_DIR / "physical_rig_feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 3-Fold Grouped Cross-Validation
    # Fold 1: Test = *_01 (normal_01, imbalance_01, misalignment_01)
    # Fold 2: Test = *_02 (normal_02, imbalance_02, misalignment_02)
    # Fold 3: Test = *_03 (normal_03, imbalance_03, misalignment_03)
    folds = [
        {"fold": 1, "test_recordings": ["normal_01", "imbalance_01", "misalignment_01"]},
        {"fold": 2, "test_recordings": ["normal_02", "imbalance_02", "misalignment_02"]},
        {"fold": 3, "test_recordings": ["normal_03", "imbalance_03", "misalignment_03"]},
    ]

    out_of_fold_predictions = []
    fold_metrics = []

    print("\n--- RUNNING 3-FOLD GROUPED VALIDATION BY PHYSICAL RUN ---")
    for f_info in folds:
        fold_id = f_info["fold"]
        test_recs = f_info["test_recordings"]
        
        train_mask = ~df_dataset["recording"].isin(test_recs)
        test_mask = df_dataset["recording"].isin(test_recs)

        df_train = df_dataset[train_mask]
        df_test = df_dataset[test_mask]

        X_train = df_train[feature_cols].values
        y_train = df_train["condition"].map(LABEL_ENCODING).values
        X_test = df_test[feature_cols].values
        y_test = df_test["condition"].map(LABEL_ENCODING).values

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="mlogloss",
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        prec_mac = precision_score(y_test, y_pred, average="macro")
        rec_mac = recall_score(y_test, y_pred, average="macro")

        fold_metrics.append({
            "fold": fold_id,
            "test_recordings": ", ".join(test_recs),
            "train_windows": len(df_train),
            "test_windows": len(df_test),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_precision": prec_mac,
            "macro_recall": rec_mac,
            "macro_f1": f1_mac,
        })
        print(f"  Fold {fold_id} (Test: {test_recs}): Acc={acc*100:.2f}%, BalAcc={bal_acc*100:.2f}%, F1={f1_mac*100:.2f}%")

        for idx, (_, row) in enumerate(df_test.iterrows()):
            pred_lbl = LABEL_DECODING[y_pred[idx]]
            prob_dict = {f"prob_{LABEL_DECODING[k].lower()}": float(y_proba[idx][k]) for k in range(3)}
            pred_record = {
                "recording": row["recording"],
                "window_index": row["window_index"],
                "fold": fold_id,
                "true_condition": row["condition"],
                "predicted_condition": pred_lbl,
                "confidence": float(np.max(y_proba[idx])),
                **prob_dict,
            }
            out_of_fold_predictions.append(pred_record)

    df_oof = pd.DataFrame(out_of_fold_predictions)
    df_oof.to_csv(PROCESSED_DIR / "physical_rig_predictions.csv", index=False)

    # Calculate overall pooled metrics across all 8,100 out-of-fold predictions
    y_true_all = df_oof["true_condition"]
    y_pred_all = df_oof["predicted_condition"]

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    overall_prec_mac = precision_score(y_true_all, y_pred_all, average="macro")
    overall_rec_mac = recall_score(y_true_all, y_pred_all, average="macro")
    overall_f1_mac = f1_score(y_true_all, y_pred_all, average="macro")

    # Per-class metrics
    per_class_rows = []
    for cls in CLASS_ORDER:
        cls_mask_true = (y_true_all == cls)
        cls_mask_pred = (y_pred_all == cls)
        
        cls_prec = precision_score(y_true_all == cls, cls_mask_pred, zero_division=0)
        cls_rec = recall_score(y_true_all == cls, cls_mask_pred, zero_division=0)
        cls_f1 = f1_score(y_true_all == cls, cls_mask_pred, zero_division=0)
        
        per_class_rows.append({
            "condition": cls,
            "total_windows": int(cls_mask_true.sum()),
            "correct_windows": int((cls_mask_true & cls_mask_pred).sum()),
            "precision": cls_prec,
            "recall": cls_rec,
            "f1_score": cls_f1,
        })

    df_eval = pd.DataFrame([
        {"level": "overall", "metric": "accuracy", "value": overall_acc},
        {"level": "overall", "metric": "balanced_accuracy", "value": overall_bal_acc},
        {"level": "overall", "metric": "precision_macro", "value": overall_prec_mac},
        {"level": "overall", "metric": "recall_macro", "value": overall_rec_mac},
        {"level": "overall", "metric": "f1_macro", "value": overall_f1_mac},
    ] + [
        {"level": f"class_{r['condition'].lower()}", "metric": m, "value": r[m]}
        for r in per_class_rows for m in ["precision", "recall", "f1_score"]
    ])
    df_eval.to_csv(PROCESSED_DIR / "physical_rig_evaluation.csv", index=False)

    # Per-recording summary
    rec_summary = []
    for rec_name, group in df_oof.groupby("recording"):
        true_cond = group["true_condition"].iloc[0]
        pred_mode = group["predicted_condition"].mode().iloc[0]
        acc_rec = float((group["true_condition"] == group["predicted_condition"]).mean())
        mean_conf = float(group["confidence"].mean())
        rec_summary.append({
            "recording": rec_name,
            "true_condition": true_cond,
            "predicted_condition": pred_mode,
            "accuracy": acc_rec,
            "mean_confidence": mean_conf,
            "total_windows": len(group),
            "match": "PASS" if true_cond == pred_mode else "FAIL",
        })
    df_rec_summary = pd.DataFrame(rec_summary)
    df_rec_summary.to_csv(PROCESSED_DIR / "physical_rig_recording_summary.csv", index=False)

    # Train final full model on all 9 recordings & save feature importance
    print("\n--- TRAINING FINAL FULL DATASET MODEL ---")
    final_clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    final_clf.fit(df_dataset[feature_cols].values, df_dataset["condition"].map(LABEL_ENCODING).values)

    joblib.dump(final_clf, MODELS_DIR / "physical_rig_classifier.pkl")

    df_feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    df_feat_imp.to_csv(PROCESSED_DIR / "physical_rig_feature_importance.csv", index=False)

    print("\n========================================================")
    print("PHYSICAL-RIG HISTORICAL ML PIPELINE COMPLETE")
    print("========================================================")
    print(f"Overall Grouped Accuracy:    {overall_acc * 100:.2f}%")
    print(f"Balanced Accuracy:           {overall_bal_acc * 100:.2f}%")
    print(f"Macro F1-Score:              {overall_f1_mac * 100:.2f}%")
    print(f"Top 5 Features:\n{df_feat_imp.head(5).to_string(index=False)}")
    print(f"Recording Summary:\n{df_rec_summary.to_string(index=False)}")


if __name__ == "__main__":
    main()

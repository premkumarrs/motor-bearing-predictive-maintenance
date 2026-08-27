"""CWRU 12 kHz Drive-End bearing fault classification with XGBoost."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
SAMPLE_RATE_HZ = 12_000
WINDOW_SIZE = 4096
CLASS_NAMES = ["NORMAL", "BALL", "INNER_RACE", "OUTER_RACE"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def metadata_for(path: Path, signal_name: str) -> dict:
    parts = path.parts
    if "Normal" in parts:
        class_name = "NORMAL"
        severity = "N/A"
        load = path.stem.rsplit("_", 1)[-1]
        position = "N/A"
    else:
        root_index = parts.index("12k_Drive_End_Bearing_Fault_Data")
        family, severity = parts[root_index + 1 : root_index + 3]
        class_name = {"B": "BALL", "IR": "INNER_RACE", "OR": "OUTER_RACE"}[family]
        load = path.stem.rsplit("_", 1)[-1]
        position = next((part.lstrip("@").upper() for part in parts if part.startswith("@")), "N/A")
    return {
        "source_file": str(path.relative_to(project_root())).replace("\\", "/"),
        "source_signal": signal_name,
        "class": class_name,
        "severity": severity,
        "load": load,
        "outer_race_position": position,
    }


def discover_signals(cwru_root: Path) -> list[tuple[Path, str, dict]]:
    discovered = []
    paths = sorted(list((cwru_root / "Normal").rglob("*.mat")) + list((cwru_root / "12k_Drive_End_Bearing_Fault_Data").rglob("*.mat")))
    for path in paths:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        for signal_name in sorted(name for name in data if re.search(r"_DE_time$", name)):
            signal = np.asarray(data[signal_name]).reshape(-1).astype(np.float64)
            if signal.size < WINDOW_SIZE:
                raise ValueError(f"Signal is shorter than one window: {path}::{signal_name}")
            discovered.append((path, signal_name, metadata_for(path, signal_name)))
    return discovered


def spectral_features(window: np.ndarray) -> dict[str, float]:
    frequencies, power = welch(window, fs=SAMPLE_RATE_HZ, nperseg=min(1024, len(window)))
    power = np.maximum(power, np.finfo(float).eps)
    total_power = float(np.sum(power))
    probability = power / total_power
    centroid = float(np.sum(frequencies * probability))
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * probability)))
    dominant = float(frequencies[int(np.argmax(power))])
    normalized = probability / probability.sum()
    entropy = float(-np.sum(normalized * np.log2(normalized)))
    spectral_kurtosis = float(kurtosis(power, fisher=False, bias=False))
    band_edges = [0, 1000, 2000, 4000, 6000]
    result = {
        "dominant_frequency_hz": dominant,
        "spectral_rms": float(np.sqrt(np.mean(power))),
        "spectral_energy": total_power,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_kurtosis": spectral_kurtosis,
        "spectral_entropy": entropy,
    }
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        result[f"band_energy_{low}_{high}_hz"] = float(np.sum(power[(frequencies >= low) & (frequencies < high)]))
    return result


def extract_features(window: np.ndarray) -> dict[str, float]:
    absolute = np.abs(window)
    mean = float(np.mean(window))
    rms = float(np.sqrt(np.mean(window**2)))
    std = float(np.std(window, ddof=0))
    peak = float(np.max(absolute))
    mean_absolute = float(np.mean(absolute))
    root_amplitude = float(np.mean(np.sqrt(absolute)) ** 2)
    result = {
        "mean": mean,
        "std": std,
        "variance": float(np.var(window, ddof=0)),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(window)),
        "crest_factor": peak / rms if rms else 0.0,
        "kurtosis": float(kurtosis(window, fisher=False, bias=False)),
        "skewness": float(skew(window, bias=False)),
        "impulse_factor": peak / mean_absolute if mean_absolute else 0.0,
        "shape_factor": rms / mean_absolute if mean_absolute else 0.0,
        "clearance_factor": peak / root_amplitude if root_amplitude else 0.0,
    }
    result.update(spectral_features(window))
    return result


def build_window_table(signals: list[tuple[Path, str, dict]]) -> pd.DataFrame:
    rows = []
    for path, signal_name, metadata in signals:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        signal = np.asarray(data[signal_name]).reshape(-1).astype(np.float64)
        for window_index, start in enumerate(range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE)):
            row = extract_features(signal[start : start + WINDOW_SIZE])
            row.update(metadata)
            row["window_index"] = window_index
            rows.append(row)
    table = pd.DataFrame(rows)
    feature_columns = [column for column in table.columns if column not in {"source_file", "source_signal", "class", "severity", "load", "outer_race_position", "window_index"}]
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns).reset_index(drop=True)
    if table[feature_columns].isna().any().any():
        raise ValueError("Invalid feature values remain after extraction.")
    return table


def grouped_split(signals: list[tuple[Path, str, dict]]) -> dict[str, list[tuple[Path, str, dict]]]:
    rng = np.random.default_rng(RANDOM_STATE)
    splits = {"train": [], "validation": [], "test": []}
    for class_name in CLASS_NAMES:
        group = [item for item in signals if item[2]["class"] == class_name]
        rng.shuffle(group)
        test_count = max(1, round(len(group) * 0.15))
        validation_count = max(1, round(len(group) * 0.15))
        splits["test"].extend(group[:test_count])
        splits["validation"].extend(group[test_count : test_count + validation_count])
        splits["train"].extend(group[test_count + validation_count :])
    return splits


def evaluate(actual: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray | None = None) -> dict:
    result = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision_macro": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    if probabilities is not None and len(np.unique(actual)) == len(CLASS_NAMES):
        result["roc_auc_ovr_macro"] = float(roc_auc_score(actual, probabilities, multi_class="ovr", average="macro"))
    else:
        result["roc_auc_ovr_macro"] = float("nan")
    return result


def main() -> None:
    root = project_root()
    cwru_root = root / "data" / "external" / "CWRU-dataset"
    model_path = root / "models" / "cwru_xgboost.pkl"
    prediction_path = root / "data" / "processed" / "cwru_predictions.csv"
    evaluation_path = root / "data" / "processed" / "cwru_evaluation.csv"
    importance_path = root / "data" / "processed" / "cwru_feature_importance.csv"
    source_paths = sorted(list((cwru_root / "Normal").rglob("*.mat")) + list((cwru_root / "12k_Drive_End_Bearing_Fault_Data").rglob("*.mat")))
    before = {path: digest(path) for path in source_paths}

    signals = discover_signals(cwru_root)
    if len(signals) != 65:
        raise ValueError(f"Expected 65 DE source signals, found {len(signals)}.")
    table = build_window_table(signals)
    metadata_columns = ["source_file", "source_signal", "class", "severity", "load", "outer_race_position", "window_index"]
    feature_columns = [column for column in table.columns if column not in metadata_columns]
    splits = grouped_split(signals)
    split_keys = {name: {f"{item[0]}::{item[1]}" for item in items} for name, items in splits.items()}
    if split_keys["train"] & split_keys["validation"] or split_keys["train"] & split_keys["test"] or split_keys["validation"] & split_keys["test"]:
        raise ValueError("Source signal leakage detected between splits.")

    signal_to_split = {f"{path}::{signal}": split for split, items in splits.items() for path, signal, _ in items}
    table["split"] = [signal_to_split[f"{root / row.source_file}::{row.source_signal}"] for row in table.itertuples()]
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]
    test = table[table["split"] == "test"]
    x_train, y_train = train[feature_columns], train["class"].map(CLASS_TO_ID)
    x_validation, y_validation = validation[feature_columns], validation["class"].map(CLASS_TO_ID)
    x_test, y_test = test[feature_columns], test["class"].map(CLASS_TO_ID)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    predicted = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)
    window_metrics = evaluate(y_test.to_numpy(), predicted, probabilities)
    majority = int(y_train.mode().iloc[0])
    baseline_accuracy = float(accuracy_score(y_test, np.full(len(y_test), majority)))

    prediction_output = test[metadata_columns].copy()
    prediction_output["true_class"] = test["class"].to_numpy()
    prediction_output["predicted_class"] = [CLASS_NAMES[index] for index in predicted]
    prediction_output["confidence"] = probabilities.max(axis=1)
    prediction_output.to_csv(prediction_path, index=False)

    source_predictions = prediction_output.groupby(["source_file", "source_signal"], as_index=False).agg(true_class=("true_class", "first"), predicted_class=("predicted_class", lambda values: values.mode().iloc[0]), confidence=("confidence", "mean"))
    source_actual = source_predictions["true_class"].map(CLASS_TO_ID).to_numpy()
    source_predicted = source_predictions["predicted_class"].map(CLASS_TO_ID).to_numpy()
    source_metrics = evaluate(source_actual, source_predicted)
    matrix = confusion_matrix(y_test, predicted, labels=list(range(len(CLASS_NAMES))))
    source_matrix = confusion_matrix(source_actual, source_predicted, labels=list(range(len(CLASS_NAMES))))

    importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    importance.to_csv(importance_path, index=False)
    evaluation_rows = []
    for name, value in window_metrics.items(): evaluation_rows.append({"level": "window", "metric": name, "value": value})
    for name, value in source_metrics.items(): evaluation_rows.append({"level": "source", "metric": name, "value": value})
    evaluation_rows.append({"level": "baseline", "metric": "majority_accuracy", "value": baseline_accuracy})
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation_rows).to_csv(evaluation_path, index=False)

    joblib.dump({"model": model, "feature_columns": feature_columns, "class_names": CLASS_NAMES, "window_size": WINDOW_SIZE, "sampling_frequency_hz": SAMPLE_RATE_HZ, "grouping": "source_file + source_signal", "split_source_signals": {name: sorted(split_keys[name]) for name in split_keys}}, model_path)
    unchanged = all(digest(path) == value for path, value in before.items())

    print("============================================================")
    print("PHASE 15 - CWRU FAULT CLASSIFICATION")
    print("============================================================")
    print(f"Dataset: CWRU 12 kHz Drive-End Bearing Dataset")
    print(f"Source signals: {len(signals)}")
    print(f"Normal: {sum(item[2]['class'] == 'NORMAL' for item in signals)}")
    print(f"Ball: {sum(item[2]['class'] == 'BALL' for item in signals)}")
    print(f"Inner race: {sum(item[2]['class'] == 'INNER_RACE' for item in signals)}")
    print(f"Outer race: {sum(item[2]['class'] == 'OUTER_RACE' for item in signals)}")
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE / SAMPLE_RATE_HZ:.3f} s)")
    print(f"Total windows: {len(table)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Training source signals: {len(splits['train'])}")
    print(f"Validation source signals: {len(splits['validation'])}")
    print(f"Test source signals: {len(splits['test'])}")
    for split_name in ("train", "validation", "test"):
        print(f"{split_name.title()} source assignments:")
        for source_key in sorted(split_keys[split_name]):
            print(f"  {source_key}")
    print(f"Training windows: {len(train)}; validation windows: {len(validation)}; test windows: {len(test)}")
    print(f"Baseline accuracy: {baseline_accuracy:.6f}")
    print(f"XGBoost accuracy: {window_metrics['accuracy']:.6f}")
    print(f"Balanced accuracy: {window_metrics['balanced_accuracy']:.6f}")
    print(f"Macro precision: {window_metrics['precision_macro']:.6f}")
    print(f"Macro recall: {window_metrics['recall_macro']:.6f}")
    print(f"Macro F1: {window_metrics['f1_macro']:.6f}")
    print(f"Weighted F1: {window_metrics['f1_weighted']:.6f}")
    print(f"ROC-AUC OVR macro: {window_metrics['roc_auc_ovr_macro']:.6f}")
    print("Window confusion matrix:")
    print(matrix)
    print("Source confusion matrix:")
    print(source_matrix)
    print("Top 15 features:")
    print(importance.head(15).to_string(index=False))
    print("Source-signal grouping: source_file + source_signal")
    print(f"Leakage check: {'PASS' if not (split_keys['train'] & split_keys['validation'] or split_keys['train'] & split_keys['test'] or split_keys['validation'] & split_keys['test']) else 'FAIL'}")
    print(f"CWRU source integrity: {'PASS' if unchanged else 'FAIL'}")
    print(f"Model: {model_path.relative_to(root)}")
    print(f"Predictions: {prediction_path.relative_to(root)}")
    print(f"Evaluation: {evaluation_path.relative_to(root)}")
    print(f"Feature importance: {importance_path.relative_to(root)}")
    print("============================================================")
    print("PHASE 15 COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()

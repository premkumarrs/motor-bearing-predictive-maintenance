"""Error analysis for physical-rig fault classifier."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

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
]

CLASS_ORDER = ["normal", "imbalance", "misalignment"]
EXPECTED_TEST_ROWS = 2700
EXPECTED_WINDOWS_PER_RECORDING = 900
EXPECTED_TEST_RECORDINGS = ["normal_03", "imbalance_03", "misalignment_03"]
CLUSTER_GAP_S = 10.0
HIGH_CONFIDENCE = 0.90

CONFUSION_PAIRS = [
    ("normal", "imbalance"),
    ("normal", "misalignment"),
    ("imbalance", "normal"),
    ("imbalance", "misalignment"),
    ("misalignment", "normal"),
    ("misalignment", "imbalance"),
]


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


def recording_filename(recording: str) -> str:
    name = str(recording)
    if name.endswith(".csv"):
        return name
    return f"{name}.csv"


def recording_stem(recording: str) -> str:
    return Path(str(recording)).stem


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    model_path: Path,
) -> None:
    required_prediction_columns = [
        "timestamp_start_ms",
        "timestamp_end_ms",
        "recording",
        "true_label",
        "predicted_label",
        "confidence",
    ]
    missing = [column for column in required_prediction_columns if column not in predictions.columns]
    if missing:
        raise AnalysisValidationError(f"predictions.csv is missing columns: {missing}")

    if len(test) != EXPECTED_TEST_ROWS:
        raise AnalysisValidationError(f"test.csv has {len(test)} rows, expected {EXPECTED_TEST_ROWS}.")
    if len(predictions) != EXPECTED_TEST_ROWS:
        raise AnalysisValidationError(
            f"predictions.csv has {len(predictions)} rows, expected {EXPECTED_TEST_ROWS}."
        )

    if not predictions["recording"].equals(test["recording"]):
        raise AnalysisValidationError("Recording names in predictions.csv do not match test.csv.")
    if not predictions["true_label"].equals(test["label"]):
        raise AnalysisValidationError("True labels in predictions.csv do not match test.csv.")

    missing_predictions = int(predictions["predicted_label"].isna().sum())
    if missing_predictions:
        raise AnalysisValidationError(f"Missing predicted labels: {missing_predictions}")
    missing_confidence = int(predictions["confidence"].isna().sum())
    if missing_confidence:
        raise AnalysisValidationError(f"Missing confidence values: {missing_confidence}")

    test_recordings = set(test["recording"].map(recording_stem))
    missing_recordings = [name for name in EXPECTED_TEST_RECORDINGS if name not in test_recordings]
    if missing_recordings:
        raise AnalysisValidationError(f"Expected test recordings missing: {missing_recordings}")

    counts = test["recording"].map(recording_stem).value_counts()
    for name in EXPECTED_TEST_RECORDINGS:
        actual = int(counts.get(name, 0))
        if actual != EXPECTED_WINDOWS_PER_RECORDING:
            raise AnalysisValidationError(
                f"{name} has {actual} windows, expected {EXPECTED_WINDOWS_PER_RECORDING}."
            )

    shared = set(train["recording"].map(recording_filename)) & set(
        test["recording"].map(recording_filename)
    )
    if shared:
        raise AnalysisValidationError(f"Shared recordings between train and test: {sorted(shared)}")

    if not model_path.exists():
        raise AnalysisValidationError(f"Model file does not exist: {model_path}")


def recording_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recording in EXPECTED_TEST_RECORDINGS:
        subset = predictions.loc[predictions["recording"].map(recording_stem) == recording]
        correct = subset["true_label"] == subset["predicted_label"]
        true_classes = subset["true_label"].unique()
        if len(true_classes) != 1:
            true_class = ",".join(sorted(true_classes.astype(str)))
        else:
            true_class = str(true_classes[0])
        rows.append(
            {
                "recording": recording_filename(recording),
                "true_class": true_class,
                "windows": int(len(subset)),
                "correct": int(correct.sum()),
                "incorrect": int((~correct).sum()),
                "accuracy": float(correct.mean()) if len(subset) else float("nan"),
                "mean_confidence": float(subset["confidence"].mean()) if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_clusters(misclassified: pd.DataFrame) -> pd.DataFrame:
    clusters = []
    for recording, group in misclassified.groupby(misclassified["recording"].map(recording_stem), sort=False):
        ordered = group.sort_values("timestamp_start_ms")
        current = None
        for _, row in ordered.iterrows():
            time_s = float(row["time_s"])
            if current is None or time_s - current["cluster_end_s"] > CLUSTER_GAP_S:
                if current is not None:
                    clusters.append(current)
                current = {
                    "recording": recording_filename(recording),
                    "cluster_start_s": time_s,
                    "cluster_end_s": time_s,
                    "number_of_errors": 1,
                }
            else:
                current["cluster_end_s"] = time_s
                current["number_of_errors"] += 1
        if current is not None:
            clusters.append(current)
    return pd.DataFrame(
        clusters,
        columns=["recording", "cluster_start_s", "cluster_end_s", "number_of_errors"],
    )


def feature_distribution(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        train_mean = float(train[feature].mean())
        test_mean = float(test[feature].mean())
        train_std = float(train[feature].std(ddof=0))
        test_std = float(test[feature].std(ddof=0))
        smd = float("nan") if train_std == 0 else (test_mean - train_mean) / train_std
        rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "smd": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_smd", ascending=False, kind="mergesort")
    return table.reset_index(drop=True)


def prediction_distribution(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    distributions = {}
    for recording in EXPECTED_TEST_RECORDINGS:
        subset = predictions.loc[predictions["recording"].map(recording_stem) == recording]
        percents = (
            subset["predicted_label"].value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0) * 100.0
        )
        distributions[recording_filename(recording)] = percents
    return distributions


def plot_error_timeline(misclassified: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    colors = {
        "normal_03": "#2ca02c",
        "imbalance_03": "#d62728",
        "misalignment_03": "#1f77b4",
    }
    for recording in EXPECTED_TEST_RECORDINGS:
        subset = misclassified.loc[misclassified["recording"].map(recording_stem) == recording]
        if subset.empty:
            continue
        axis.scatter(
            subset["time_s"],
            [recording_filename(recording)] * len(subset),
            color=colors[recording],
            s=40,
            label=recording_filename(recording),
            zorder=3,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Recording")
    axis.set_title("Misclassified Test Windows Over Time")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run() -> None:
    root = project_root()
    train_path = root / "data" / "processed" / "train.csv"
    test_path = root / "data" / "processed" / "test.csv"
    predictions_path = root / "data" / "processed" / "evaluation" / "predictions.csv"
    model_path = root / "models" / "fault_classifier.pkl"
    distribution_path = root / "data" / "processed" / "evaluation" / "feature_distribution.csv"
    timeline_path = root / "data" / "processed" / "evaluation" / "error_timeline.png"

    protected = [
        root / "data" / "processed" / "evaluation" / "confusion_matrix.png",
        root / "data" / "processed" / "evaluation" / "feature_importance.png",
        root / "data" / "processed" / "evaluation" / "feature_importance.csv",
        predictions_path,
        model_path,
        train_path,
        test_path,
    ]

    for path in [test_path, model_path, predictions_path, train_path]:
        if not path.exists():
            raise AnalysisValidationError(f"Required file does not exist: {path}")

    protected_hashes = {str(path): file_digest(path) for path in protected if path.exists()}

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    predictions = pd.read_csv(predictions_path)
    validate_inputs(train, test, predictions, model_path)

    predictions = predictions.copy()
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions["time_s"] = predictions["timestamp_start_ms"] / 1000.0
    misclassified = predictions.loc[~predictions["correct"]].copy()
    correct = predictions.loc[predictions["correct"]].copy()

    recording_table = recording_performance(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_ORDER,
        zero_division=0,
    )
    report = classification_report(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_ORDER,
        digits=4,
        zero_division=0,
    )

    pair_counts = []
    for true_label, predicted_label in CONFUSION_PAIRS:
        count = int(
            (
                (misclassified["true_label"] == true_label)
                & (misclassified["predicted_label"] == predicted_label)
            ).sum()
        )
        if count:
            pair_counts.append((true_label, predicted_label, count))

    total = len(predictions)
    n_correct = int(predictions["correct"].sum())
    n_incorrect = int((~predictions["correct"]).sum())

    mean_conf_correct = float(correct["confidence"].mean()) if not correct.empty else float("nan")
    mean_conf_incorrect = float(misclassified["confidence"].mean()) if not misclassified.empty else float("nan")
    min_confidence = float(predictions["confidence"].min())
    max_confidence = float(predictions["confidence"].max())

    incorrect_conf = misclassified["confidence"]
    conf_lt_50 = int((incorrect_conf < 0.50).sum())
    conf_50_75 = int(((incorrect_conf >= 0.50) & (incorrect_conf < 0.75)).sum())
    conf_75_90 = int(((incorrect_conf >= 0.75) & (incorrect_conf < 0.90)).sum())
    conf_ge_90 = int((incorrect_conf >= HIGH_CONFIDENCE).sum())

    clusters = build_clusters(misclassified)
    distribution = feature_distribution(train, test)
    pred_dist = prediction_distribution(predictions)

    distribution[
        ["feature", "train_mean", "test_mean", "train_std", "test_std", "smd", "abs_smd"]
    ].to_csv(
        distribution_path,
        index=False,
    )
    plot_error_timeline(misclassified, timeline_path)

    best_row = recording_table.loc[recording_table["accuracy"].idxmax()]
    worst_row = recording_table.loc[recording_table["accuracy"].idxmin()]
    most_common_pair = max(pair_counts, key=lambda item: item[2]) if pair_counts else None

    print("============================================================")
    print("PHASE 8 - MODEL VALIDATION & ERROR ANALYSIS")
    print("============================================================")
    print()
    print("DATASET")
    print("-------")
    print()
    print("Test samples:")
    print(total)
    print()
    print("Test recordings:")
    print(len(EXPECTED_TEST_RECORDINGS))
    print()
    print("------------------------------------------------------------")
    print("RECORDING PERFORMANCE")
    print("------------------------------------------------------------")
    print()
    for _, row in recording_table.iterrows():
        print(row["recording"])
        print(f"  true class:          {row['true_class']}")
        print(f"  windows:             {row['windows']}")
        print(f"  correct:             {row['correct']}")
        print(f"  incorrect:           {row['incorrect']}")
        print(f"  accuracy:            {row['accuracy']}")
        print(f"  mean confidence:     {row['mean_confidence']}")
        print()
    print("------------------------------------------------------------")
    print("CLASS PERFORMANCE")
    print("------------------------------------------------------------")
    print()
    print(report)
    for label, prec, rec, f1_score, n_support in zip(CLASS_ORDER, precision, recall, f1, support):
        print(f"{label}")
        print(f"  precision: {prec}")
        print(f"  recall:    {rec}")
        print(f"  f1-score:  {f1_score}")
        print(f"  support:   {int(n_support)}")
        print()
    print("------------------------------------------------------------")
    print("CONFUSION ANALYSIS")
    print("------------------------------------------------------------")
    print()
    print(f"Total correct:    {n_correct}")
    print(f"Total incorrect:  {n_incorrect}")
    print(f"Total samples:    {total}")
    print()
    if not pair_counts:
        print("No confusion pairs.")
    else:
        for true_label, predicted_label, count in pair_counts:
            print(f"true {true_label} -> predicted {predicted_label}: {count}")
    print()
    print("------------------------------------------------------------")
    print("CONFIDENCE ANALYSIS")
    print("------------------------------------------------------------")
    print()
    print("Mean confidence of correct predictions:")
    print(mean_conf_correct)
    print()
    print("Mean confidence of incorrect predictions:")
    print(mean_conf_incorrect)
    print()
    print("Minimum confidence:")
    print(min_confidence)
    print()
    print("Maximum confidence:")
    print(max_confidence)
    print()
    print("Incorrect predictions by model confidence:")
    print(f"  confidence < 0.50:       {conf_lt_50}")
    print(f"  confidence 0.50-0.75:    {conf_50_75}")
    print(f"  confidence 0.75-0.90:    {conf_75_90}")
    print(f"  confidence >= 0.90:      {conf_ge_90}")
    print()
    print("These values are model confidence scores, not calibrated probabilities.")
    print()
    print("------------------------------------------------------------")
    print("TEMPORAL ERROR ANALYSIS")
    print("------------------------------------------------------------")
    print()
    print("Misclassified windows:")
    print()
    if misclassified.empty:
        print("None")
    else:
        display = misclassified[
            ["recording", "time_s", "true_label", "predicted_label", "confidence"]
        ].copy()
        display["recording"] = display["recording"].map(recording_filename)
        print(display.to_string(index=False))
    print()
    print("Error clusters (gap <= 10 s):")
    print()
    if clusters.empty:
        print("None")
    else:
        print(clusters.to_string(index=False))
    print()
    print("------------------------------------------------------------")
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("------------------------------------------------------------")
    print()
    print("Top features by absolute standardized mean difference (TEST vs TRAIN):")
    print()
    print(
        distribution.head(15)[
            ["feature", "train_mean", "test_mean", "train_std", "test_std", "smd"]
        ].to_string(index=False)
    )
    print()
    print("This comparison is descriptive only. Features were not removed or transformed.")
    print()
    print("------------------------------------------------------------")
    print("PREDICTION DISTRIBUTION")
    print("------------------------------------------------------------")
    print()
    for recording, percents in pred_dist.items():
        print(recording)
        print()
        for label in CLASS_ORDER:
            print(f"{label}: {float(percents[label]):.4f}%")
        print()
    print("------------------------------------------------------------")
    print("RECORDING ERROR ANALYSIS")
    print("------------------------------------------------------------")
    print()
    for _, row in recording_table.iterrows():
        error_rate = row["incorrect"] / row["windows"] if row["windows"] else float("nan")
        print(row["recording"])
        print(f"  error_count: {int(row['incorrect'])}")
        print(f"  error_rate:  {error_rate}")
        print()
    print("------------------------------------------------------------")
    print("FINAL ERROR SUMMARY")
    print("------------------------------------------------------------")
    print()
    print(f"Total test windows:           {total}")
    print(f"Correct predictions:          {n_correct}")
    print(f"Incorrect predictions:        {n_incorrect}")
    print(f"Overall accuracy:             {n_correct / total}")
    print()
    print(f"Best-performing recording:    {best_row['recording']} ({best_row['accuracy']})")
    print(f"Worst-performing recording:   {worst_row['recording']} ({worst_row['accuracy']})")
    print()
    if most_common_pair:
        print(
            "Most common confusion pair:   "
            f"true {most_common_pair[0]} -> predicted {most_common_pair[1]} "
            f"({most_common_pair[2]})"
        )
    print()
    print(f"Mean confidence (correct):    {mean_conf_correct}")
    print(f"Mean confidence (incorrect):  {mean_conf_incorrect}")
    print(f"High-confidence errors (>=0.90): {conf_ge_90}")
    print(f"Number of error clusters:     {len(clusters)}")
    print()
    print("Interpretation:")
    print("- Evaluation uses three held-out physical recordings.")
    print("- Those recordings come from the same experimental hardware/setup.")
    print("- This analysis does not establish performance across different motors,")
    print("  loads, sensor placements, or operating environments.")
    print("- Model confidence is not a calibrated probability.")
    print("- This analysis does not modify or improve the model.")
    print()
    print("============================================================")
    print("PHASE 8 COMPLETE")
    print("============================================================")

    for path, digest in protected_hashes.items():
        if file_digest(Path(path)) != digest:
            raise AnalysisValidationError(f"Protected file was modified: {path}")


if __name__ == "__main__":
    try:
        run()
    except AnalysisValidationError as error:
        print("PHASE 8 STOPPED")
        print("---------------")
        print(error)
        sys.exit(1)

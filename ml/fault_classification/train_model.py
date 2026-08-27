"""Train and evaluate baseline fault classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
CLASS_DISPLAY = ["Normal", "Imbalance", "Misalignment"]

EXPECTED_TRAIN_ROWS = 5400
EXPECTED_TEST_ROWS = 2700

METADATA_COLUMNS = [
    "timestamp_start_ms",
    "timestamp_end_ms",
    "recording",
]


class TrainingValidationError(Exception):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def recording_filename(recording: str) -> str:
    name = str(recording)
    if name.endswith(".csv"):
        return name
    return f"{name}.csv"


def validate_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing_features = [column for column in FEATURE_COLUMNS if column not in train.columns]
    if missing_features:
        raise TrainingValidationError(
            f"train.csv is missing feature columns: {missing_features}"
        )
    missing_features = [column for column in FEATURE_COLUMNS if column not in test.columns]
    if missing_features:
        raise TrainingValidationError(
            f"test.csv is missing feature columns: {missing_features}"
        )

    for name, frame in (("train.csv", train), ("test.csv", test)):
        if "label" not in frame.columns:
            raise TrainingValidationError(f"{name} is missing the label column.")
        if "recording" not in frame.columns:
            raise TrainingValidationError(f"{name} is missing the recording column.")

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise TrainingValidationError(
            f"Training rows = {len(train)}, expected {EXPECTED_TRAIN_ROWS}."
        )
    if len(test) != EXPECTED_TEST_ROWS:
        raise TrainingValidationError(
            f"Test rows = {len(test)}, expected {EXPECTED_TEST_ROWS}."
        )

    for name, frame in (("train.csv", train), ("test.csv", test)):
        missing_features = int(frame[FEATURE_COLUMNS].isna().sum().sum())
        if missing_features:
            raise TrainingValidationError(
                f"Missing feature values in {name}: {missing_features}"
            )
        feature_values = frame[FEATURE_COLUMNS]
        infinite_features = int(
            (feature_values == float("inf")).sum().sum()
            + (feature_values == float("-inf")).sum().sum()
        )
        if infinite_features:
            raise TrainingValidationError(
                f"Infinite feature values in {name}: {infinite_features}"
            )

        present = set(frame["label"].dropna().astype(str).unique())
        missing_classes = [label for label in CLASS_ORDER if label not in present]
        if missing_classes:
            raise TrainingValidationError(
                f"Missing classes in {name}: {missing_classes}"
            )


def leakage_check(train: pd.DataFrame, test: pd.DataFrame) -> set[str]:
    train_recordings = set(train["recording"].map(recording_filename))
    test_recordings = set(test["recording"].map(recording_filename))
    shared = train_recordings & test_recordings
    print("DATA LEAKAGE CHECK")
    print("------------------")
    print()
    print("Shared recordings:", len(shared))
    if shared:
        print("Shared recording names:", ", ".join(sorted(shared)))
        raise TrainingValidationError(
            "Shared recordings detected. Training stopped to prevent data leakage."
        )
    print()
    return shared


def plot_confusion_matrix(matrix, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_DISPLAY,
    )
    display.plot(ax=axis, cmap="Blues", colorbar=True, values_format="d")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Fault Classification Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    top = importance.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.barh(top["feature"], top["importance"], color="#1f4e79")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title("Top 15 Feature Importances")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def print_confusion_matrix(matrix) -> None:
    header = f"{'':<16}{'Normal':>12}{'Imbalance':>12}{'Misalignment':>14}"
    print(f"{'':<16}Predicted")
    print(header)
    print()
    for row_label, row in zip(CLASS_DISPLAY, matrix):
        print(f"{row_label:<16}{row[0]:>12}{row[1]:>12}{row[2]:>14}")


def run() -> None:
    root = project_root()
    train_path = root / "data" / "processed" / "train.csv"
    test_path = root / "data" / "processed" / "test.csv"
    eval_dir = root / "data" / "processed" / "evaluation"
    model_dir = root / "models"
    model_path = model_dir / "fault_classifier.pkl"
    confusion_path = eval_dir / "confusion_matrix.png"
    importance_csv_path = eval_dir / "feature_importance.csv"
    importance_png_path = eval_dir / "feature_importance.png"
    predictions_path = eval_dir / "predictions.csv"

    if not train_path.exists():
        raise TrainingValidationError(f"train.csv does not exist: {train_path}")
    if not test_path.exists():
        raise TrainingValidationError(f"test.csv does not exist: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_split(train, test)
    leakage_check(train, test)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"]
    x_test = test[FEATURE_COLUMNS]
    y_test = test["label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    class_index = {label: index for index, label in enumerate(model.classes_)}
    predicted_indices = [class_index[label] for label in y_pred]
    confidence = probabilities[range(len(y_pred)), predicted_indices]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    importance = importance.reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "timestamp_start_ms": test["timestamp_start_ms"].to_numpy(),
            "timestamp_end_ms": test["timestamp_end_ms"].to_numpy(),
            "recording": test["recording"].to_numpy(),
            "true_label": y_test.to_numpy(),
            "predicted_label": y_pred,
            "confidence": confidence,
        }
    )
    misclassified = predictions.loc[
        predictions["true_label"] != predictions["predicted_label"]
    ].copy()

    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_csv_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    plot_confusion_matrix(matrix, confusion_path)
    plot_feature_importance(importance, importance_png_path)
    joblib.dump(model, model_path)

    print("MODEL PERFORMANCE")
    print("-----------------")
    print()
    print("Accuracy:")
    print(accuracy)
    print()
    print("Accuracy percentage:")
    print(f"{accuracy * 100:.4f}%")
    print()
    print("Precision (macro):")
    print(precision)
    print()
    print("Recall (macro):")
    print(recall)
    print()
    print("F1-score (macro):")
    print(f1)
    print()
    print("CLASSIFICATION REPORT")
    print("---------------------")
    print()
    print(report)
    print("CONFUSION MATRIX")
    print("----------------")
    print()
    print_confusion_matrix(matrix)
    print()
    print("TOP FEATURE IMPORTANCE")
    print("----------------------")
    print()
    print(importance.head(15).to_string(index=False))
    print()
    print("MISCLASSIFIED SAMPLES")
    print("---------------------")
    print()
    if misclassified.empty:
        print("None")
    else:
        print(f"Count: {len(misclassified)}")
        print()
        print(
            misclassified[
                [
                    "timestamp_start_ms",
                    "recording",
                    "true_label",
                    "predicted_label",
                    "confidence",
                ]
            ].to_string(index=False)
        )
    print()
    print("CONFIDENCE")
    print("----------")
    print()
    print("Mean confidence:")
    print(float(confidence.mean()))
    print()
    print("Minimum confidence:")
    print(float(confidence.min()))
    print()
    print("Maximum confidence:")
    print(float(confidence.max()))
    print()
    print("MODEL SAVED")
    print("-----------")
    print()
    print("Saved to:")
    print("models/fault_classifier.pkl")
    print()
    print("EVALUATION FILES")
    print("----------------")
    print()
    print("data/processed/evaluation/confusion_matrix.png")
    print("data/processed/evaluation/feature_importance.csv")
    print("data/processed/evaluation/feature_importance.png")
    print("data/processed/evaluation/predictions.csv")


if __name__ == "__main__":
    try:
        run()
    except TrainingValidationError as error:
        print("TRAINING STOPPED")
        print("----------------")
        print(error)
        sys.exit(1)

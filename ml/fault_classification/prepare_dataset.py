"""Assign recording-level condition labels to extracted features."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

RECORDING_LABELS = {
    "normal_01.csv": "normal",
    "normal_02.csv": "normal",
    "normal_03.csv": "normal",
    "imbalance_01.csv": "imbalance",
    "imbalance_02.csv": "imbalance",
    "imbalance_03.csv": "imbalance",
    "misalignment_01.csv": "misalignment",
    "misalignment_02.csv": "misalignment",
    "misalignment_03.csv": "misalignment",
}

CONDITION_ORDER = ["normal", "imbalance", "misalignment"]

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

METADATA_COLUMNS = [
    "timestamp_start_ms",
    "timestamp_end_ms",
    "recording",
]

OUTPUT_COLUMNS = FEATURE_COLUMNS + METADATA_COLUMNS + ["label"]


class DatasetValidationError(Exception):
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


def label_for_recording(recording: str) -> str:
    filename = recording_filename(recording)
    if filename not in RECORDING_LABELS:
        raise DatasetValidationError(
            f"Unknown recording name: {filename!r}. "
            "Every recording must map to exactly one known condition."
        )
    return RECORDING_LABELS[filename]


def snapshot_raw_files(raw_dir: Path) -> dict[str, str]:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise DatasetValidationError(f"No raw CSV files found in {raw_dir}")
    return {str(path): file_digest(path) for path in csv_files}


def validate_input(features: pd.DataFrame, input_path: Path) -> None:
    if not input_path.exists():
        raise DatasetValidationError(f"Input file does not exist: {input_path}")

    missing_features = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing_features:
        raise DatasetValidationError(
            f"Input is missing expected feature columns: {missing_features}"
        )

    missing_metadata = [column for column in METADATA_COLUMNS if column not in features.columns]
    if missing_metadata:
        raise DatasetValidationError(
            f"Input is missing expected metadata columns: {missing_metadata}"
        )


def validate_dataset(
    features: pd.DataFrame,
    dataset: pd.DataFrame,
    raw_hashes_before: dict[str, str],
    raw_dir: Path,
    features_digest_before: str,
    features_path: Path,
) -> tuple[int, int, int, int]:
    unknown = sorted(
        {
            recording_filename(name)
            for name in dataset["recording"].unique()
            if recording_filename(name) not in RECORDING_LABELS
        }
    )
    if unknown:
        raise DatasetValidationError(f"Unknown recording names exist: {unknown}")

    labels_per_recording = dataset.groupby("recording")["label"].nunique()
    mixed = labels_per_recording[labels_per_recording != 1]
    if not mixed.empty:
        raise DatasetValidationError(
            "Recording belongs to more than one condition: "
            f"{mixed.index.tolist()}"
        )

    missing_labels = int(dataset["label"].isna().sum() + (dataset["label"].astype(str) == "").sum())
    if missing_labels:
        raise DatasetValidationError(f"Missing labels: {missing_labels}")

    feature_values = dataset[FEATURE_COLUMNS]
    missing_features = int(feature_values.isna().sum().sum())
    if missing_features:
        raise DatasetValidationError(f"Missing feature values: {missing_features}")

    infinite_features = int(
        (feature_values == float("inf")).sum().sum()
        + (feature_values == float("-inf")).sum().sum()
    )
    if infinite_features:
        raise DatasetValidationError(f"Infinite feature values: {infinite_features}")

    duplicate_rows = int(dataset.duplicated().sum())
    if duplicate_rows:
        raise DatasetValidationError(f"Duplicate rows: {duplicate_rows}")

    if len(dataset) != len(features):
        raise DatasetValidationError(
            f"Output row count ({len(dataset)}) does not equal "
            f"input row count ({len(features)})."
        )

    raw_hashes_after = snapshot_raw_files(raw_dir)
    if raw_hashes_after != raw_hashes_before:
        raise DatasetValidationError("Raw files were modified.")

    if file_digest(features_path) != features_digest_before:
        raise DatasetValidationError("Phase 4 features.csv was modified.")

    preserved = FEATURE_COLUMNS + METADATA_COLUMNS
    if not features[preserved].equals(dataset[preserved]):
        raise DatasetValidationError("Phase 4 feature values were modified.")

    input_counts = features["recording"].value_counts()
    output_counts = dataset["recording"].value_counts()
    if not input_counts.equals(output_counts):
        raise DatasetValidationError(
            "Recording window counts changed between input and output."
        )

    derived = dataset["recording"].map(label_for_recording)
    mismatched = dataset.loc[dataset["label"] != derived, "recording"].unique().tolist()
    if mismatched:
        raise DatasetValidationError(
            f"Labels do not correspond exactly to recording names: {mismatched}"
        )

    return missing_labels, missing_features, infinite_features, duplicate_rows


def print_report(
    input_path: Path,
    output_path: Path,
    input_rows: int,
    output_rows: int,
    missing_labels: int,
    missing_features: int,
    infinite_features: int,
    duplicate_rows: int,
    dataset: pd.DataFrame,
) -> None:
    rows_added = max(0, output_rows - input_rows)
    rows_removed = max(0, input_rows - output_rows)

    print("DATASET CONSTRUCTION COMPLETE")
    print("-----------------------------")
    print()
    print("Input:")
    print(input_path.as_posix() if input_path.as_posix().startswith("data/") else "data/processed/features.csv")
    print()
    print("Output:")
    print(
        output_path.as_posix()
        if output_path.as_posix().startswith("data/")
        else "data/processed/classification_dataset.csv"
    )
    print()
    print("Total input rows:")
    print(input_rows)
    print()
    print("Total output rows:")
    print(output_rows)
    print()
    print("Rows added:")
    print(rows_added)
    print()
    print("Rows removed:")
    print(rows_removed)
    print()
    print("Missing labels:")
    print(missing_labels)
    print()
    print("Missing feature values:")
    print(missing_features)
    print()
    print("Infinite feature values:")
    print(infinite_features)
    print()
    print("Duplicate rows:")
    print(duplicate_rows)
    print()
    print("CLASS DISTRIBUTION")
    print()
    class_counts = dataset["label"].value_counts()
    for condition in CONDITION_ORDER:
        if condition in class_counts.index:
            print(f"{condition:<16}{int(class_counts[condition])}")
    extra_labels = [label for label in class_counts.index if label not in CONDITION_ORDER]
    for label in extra_labels:
        print(f"{label:<16}{int(class_counts[label])}")
    print()
    print("RECORDING DISTRIBUTION")
    print()
    recording_counts = dataset.groupby(["label", "recording"]).size()
    for condition in CONDITION_ORDER:
        print(condition)
        condition_recordings = [
            filename
            for filename, mapped in RECORDING_LABELS.items()
            if mapped == condition
        ]
        for filename in condition_recordings:
            stem = Path(filename).stem
            if (condition, stem) in recording_counts.index:
                count = int(recording_counts[(condition, stem)])
            elif (condition, filename) in recording_counts.index:
                count = int(recording_counts[(condition, filename)])
            else:
                continue
            print(f"  {filename:<22}{count}")
        print()
    print("FINAL COLUMNS")
    print()
    for column in dataset.columns:
        print(column)
    print()
    print("Dataset saved successfully.")


def run() -> pd.DataFrame:
    root = project_root()
    input_path = root / "data" / "processed" / "features.csv"
    output_path = root / "data" / "processed" / "classification_dataset.csv"
    raw_dir = root / "data" / "raw"

    if not input_path.exists():
        raise DatasetValidationError(f"Input file does not exist: {input_path}")

    raw_hashes_before = snapshot_raw_files(raw_dir)
    features_digest_before = file_digest(input_path)

    features = pd.read_csv(input_path)
    validate_input(features, input_path)

    dataset = features.copy()
    dataset["label"] = dataset["recording"].map(label_for_recording)
    dataset = dataset[OUTPUT_COLUMNS]

    missing_labels, missing_features, infinite_features, duplicate_rows = validate_dataset(
        features=features,
        dataset=dataset,
        raw_hashes_before=raw_hashes_before,
        raw_dir=raw_dir,
        features_digest_before=features_digest_before,
        features_path=input_path,
    )

    dataset.to_csv(output_path, index=False)

    if file_digest(input_path) != features_digest_before:
        raise DatasetValidationError("Phase 4 features.csv was modified.")
    if snapshot_raw_files(raw_dir) != raw_hashes_before:
        raise DatasetValidationError("Raw files were modified.")

    print_report(
        input_path=Path("data/processed/features.csv"),
        output_path=Path("data/processed/classification_dataset.csv"),
        input_rows=len(features),
        output_rows=len(dataset),
        missing_labels=missing_labels,
        missing_features=missing_features,
        infinite_features=infinite_features,
        duplicate_rows=duplicate_rows,
        dataset=dataset,
    )
    return dataset


if __name__ == "__main__":
    try:
        run()
    except DatasetValidationError as error:
        print("DATASET CONSTRUCTION STOPPED")
        print("----------------------------")
        print(error)
        sys.exit(1)

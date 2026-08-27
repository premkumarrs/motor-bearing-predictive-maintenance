"""Recording-level train/test partitioning."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

TRAIN_RECORDINGS = [
    "normal_01.csv",
    "normal_02.csv",
    "imbalance_01.csv",
    "imbalance_02.csv",
    "misalignment_01.csv",
    "misalignment_02.csv",
]

TEST_RECORDINGS = [
    "normal_03.csv",
    "imbalance_03.csv",
    "misalignment_03.csv",
]

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

EXPECTED_INPUT_ROWS = 8100
EXPECTED_TRAIN_ROWS = 5400
EXPECTED_TEST_ROWS = 2700
EXPECTED_WINDOWS_PER_RECORDING = 900


class SplitValidationError(Exception):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_raw_files(raw_dir: Path) -> dict[str, str]:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise SplitValidationError(f"No raw CSV files found in {raw_dir}")
    return {str(path): file_digest(path) for path in csv_files}


def recording_filename(recording: str) -> str:
    name = str(recording)
    if name.endswith(".csv"):
        return name
    return f"{name}.csv"


def format_recording_counts(frame: pd.DataFrame, expected_order: list[str]) -> list[tuple[str, int]]:
    counts = frame["recording"].map(recording_filename).value_counts()
    rows = []
    for filename in expected_order:
        rows.append((filename, int(counts.get(filename, 0))))
    extra = [name for name in counts.index if name not in expected_order]
    for name in extra:
        rows.append((name, int(counts[name])))
    return rows


def validate_split(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_hashes_before: dict[str, str],
    raw_dir: Path,
    dataset_digest_before: str,
    dataset_path: Path,
) -> tuple[int, int]:
    if len(dataset) != EXPECTED_INPUT_ROWS:
        raise SplitValidationError(
            f"Input rows = {len(dataset)}, expected {EXPECTED_INPUT_ROWS}."
        )
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise SplitValidationError(
            f"Train rows = {len(train)}, expected {EXPECTED_TRAIN_ROWS}."
        )
    if len(test) != EXPECTED_TEST_ROWS:
        raise SplitValidationError(
            f"Test rows = {len(test)}, expected {EXPECTED_TEST_ROWS}."
        )
    if len(train) + len(test) != len(dataset):
        raise SplitValidationError(
            "Train + test does not equal input exactly."
        )

    missing_rows = len(dataset) - (len(train) + len(test))
    if missing_rows != 0:
        raise SplitValidationError(f"Rows were lost: {missing_rows}.")

    train_dupes = int(train.duplicated().sum())
    test_dupes = int(test.duplicated().sum())
    combined_dupes = int(
        pd.concat([train, test], ignore_index=True).duplicated().sum()
    )
    duplicate_rows = train_dupes + test_dupes + combined_dupes
    if duplicate_rows:
        raise SplitValidationError("Duplicate rows exist in the split.")

    train_recordings = set(train["recording"].map(recording_filename))
    test_recordings = set(test["recording"].map(recording_filename))
    shared = sorted(train_recordings & test_recordings)
    if shared:
        raise SplitValidationError(
            f"Recordings exist in both TRAIN and TEST: {shared}"
        )

    missing_train = [name for name in TRAIN_RECORDINGS if name not in train_recordings]
    if missing_train:
        raise SplitValidationError(
            f"Expected training recordings missing from TRAIN: {missing_train}"
        )
    extra_train = sorted(train_recordings - set(TRAIN_RECORDINGS))
    if extra_train:
        raise SplitValidationError(
            f"Unexpected recordings in TRAIN: {extra_train}"
        )

    missing_test = [name for name in TEST_RECORDINGS if name not in test_recordings]
    if missing_test:
        raise SplitValidationError(
            f"Expected test recordings missing from TEST: {missing_test}"
        )
    extra_test = sorted(test_recordings - set(TEST_RECORDINGS))
    if extra_test:
        raise SplitValidationError(
            f"Unexpected recordings in TEST: {extra_test}"
        )

    for split_name, frame in (("TRAIN", train), ("TEST", test)):
        present = set(frame["label"].dropna().unique())
        missing_classes = [name for name in CONDITION_ORDER if name not in present]
        if missing_classes:
            raise SplitValidationError(
                f"Missing classes in {split_name}: {missing_classes}"
            )

    for filename, count in format_recording_counts(train, TRAIN_RECORDINGS):
        if count != EXPECTED_WINDOWS_PER_RECORDING:
            raise SplitValidationError(
                f"Training recording {filename} has {count} rows, "
                f"expected {EXPECTED_WINDOWS_PER_RECORDING}."
            )
    for filename, count in format_recording_counts(test, TEST_RECORDINGS):
        if count != EXPECTED_WINDOWS_PER_RECORDING:
            raise SplitValidationError(
                f"Test recording {filename} has {count} rows, "
                f"expected {EXPECTED_WINDOWS_PER_RECORDING}."
            )

    for split_name, frame in (("TRAIN", train), ("TEST", test)):
        missing_labels = int(frame["label"].isna().sum() + (frame["label"].astype(str) == "").sum())
        if missing_labels:
            raise SplitValidationError(f"Missing labels in {split_name}: {missing_labels}")

        missing_features = int(frame[FEATURE_COLUMNS].isna().sum().sum())
        if missing_features:
            raise SplitValidationError(
                f"Missing feature values in {split_name}: {missing_features}"
            )

        feature_values = frame[FEATURE_COLUMNS]
        infinite_features = int(
            (feature_values == float("inf")).sum().sum()
            + (feature_values == float("-inf")).sum().sum()
        )
        if infinite_features:
            raise SplitValidationError(
                f"Infinite feature values in {split_name}: {infinite_features}"
            )

    if list(train.columns) != list(test.columns):
        raise SplitValidationError("Column order differs between train.csv and test.csv.")
    if list(train.columns) != list(dataset.columns):
        raise SplitValidationError(
            "Column order differs from classification_dataset.csv."
        )

    if snapshot_raw_files(raw_dir) != raw_hashes_before:
        raise SplitValidationError("Raw files were modified.")
    if file_digest(dataset_path) != dataset_digest_before:
        raise SplitValidationError("classification_dataset.csv was modified.")

    return missing_rows, 0


def print_report(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    missing_rows: int,
    duplicate_rows: int,
) -> None:
    print("DATASET SPLIT COMPLETE")
    print("-----------------------")
    print()
    print("Total samples:")
    print(len(dataset))
    print()
    print("Training samples:")
    print(len(train))
    print()
    print("Test samples:")
    print(len(test))
    print()
    print("TRAIN RECORDINGS")
    print()
    for filename, count in format_recording_counts(train, TRAIN_RECORDINGS):
        print(f"{filename:<22}{count}")
    print()
    print("TEST RECORDINGS")
    print()
    for filename, count in format_recording_counts(test, TEST_RECORDINGS):
        print(f"{filename:<22}{count}")
    print()
    print("TRAIN CLASS DISTRIBUTION")
    print()
    train_classes = train["label"].value_counts()
    for condition in CONDITION_ORDER:
        print(condition)
        print(int(train_classes.get(condition, 0)))
        print()
    print("TEST CLASS DISTRIBUTION")
    print()
    test_classes = test["label"].value_counts()
    for condition in CONDITION_ORDER:
        print(condition)
        print(int(test_classes.get(condition, 0)))
        print()
    print("LEAKAGE CHECK")
    print()
    print("Training recordings:")
    print(", ".join(TRAIN_RECORDINGS))
    print()
    print("Testing recordings:")
    print(", ".join(TEST_RECORDINGS))
    print()
    print("Shared recordings:")
    train_set = set(train["recording"].map(recording_filename))
    test_set = set(test["recording"].map(recording_filename))
    print(len(train_set & test_set))
    print()
    print("ROW INTEGRITY")
    print()
    print("Input rows:")
    print(len(dataset))
    print()
    print("Train rows:")
    print(len(train))
    print()
    print("Test rows:")
    print(len(test))
    print()
    print("Combined:")
    print(len(train) + len(test))
    print()
    print("Missing rows:")
    print(missing_rows)
    print()
    print("Duplicate rows:")
    print(duplicate_rows)
    print()
    print("Saved training data:")
    print("data/processed/train.csv")
    print()
    print("Saved test data:")
    print("data/processed/test.csv")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = project_root()
    dataset_path = root / "data" / "processed" / "classification_dataset.csv"
    train_path = root / "data" / "processed" / "train.csv"
    test_path = root / "data" / "processed" / "test.csv"
    raw_dir = root / "data" / "raw"

    if not dataset_path.exists():
        raise SplitValidationError(f"Input file does not exist: {dataset_path}")

    raw_hashes_before = snapshot_raw_files(raw_dir)
    dataset_digest_before = file_digest(dataset_path)

    dataset = pd.read_csv(dataset_path)
    recording_names = dataset["recording"].map(recording_filename)

    unassigned = sorted(set(recording_names) - set(TRAIN_RECORDINGS) - set(TEST_RECORDINGS))
    if unassigned:
        raise SplitValidationError(
            f"Recordings are not assigned to TRAIN or TEST: {unassigned}"
        )

    train = dataset.loc[recording_names.isin(TRAIN_RECORDINGS)].copy()
    test = dataset.loc[recording_names.isin(TEST_RECORDINGS)].copy()

    missing_rows, duplicate_rows = validate_split(
        dataset=dataset,
        train=train,
        test=test,
        raw_hashes_before=raw_hashes_before,
        raw_dir=raw_dir,
        dataset_digest_before=dataset_digest_before,
        dataset_path=dataset_path,
    )

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    if file_digest(dataset_path) != dataset_digest_before:
        raise SplitValidationError("classification_dataset.csv was modified.")
    if snapshot_raw_files(raw_dir) != raw_hashes_before:
        raise SplitValidationError("Raw files were modified.")

    print_report(
        dataset=dataset,
        train=train,
        test=test,
        missing_rows=missing_rows,
        duplicate_rows=duplicate_rows,
    )
    return train, test


if __name__ == "__main__":
    try:
        run()
    except SplitValidationError as error:
        print("DATASET SPLIT STOPPED")
        print("---------------------")
        print(error)
        sys.exit(1)

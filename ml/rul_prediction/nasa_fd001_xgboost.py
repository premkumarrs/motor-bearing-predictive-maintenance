"""Official NASA C-MAPSS FD001 RUL model using XGBoost."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

COLUMN_NAMES = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{index}" for index in range(1, 22)]
DATASET_DIR = "data/external/nasa_cmapps"
RUL_CAP = 125.0
VALIDATION_UNITS = list(range(81, 101))
FEATURE_COLUMNS = ["cycle", "setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_fd001(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path}")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for sensor in SENSOR_COLUMNS:
        result[f"{sensor}_diff"] = result.groupby("unit")[sensor].diff().fillna(0.0)
        result[f"{sensor}_rolling_mean"] = result.groupby("unit")[sensor].transform(lambda values: values.rolling(5, min_periods=1).mean())
        result[f"{sensor}_rolling_std"] = result.groupby("unit")[sensor].transform(lambda values: values.rolling(5, min_periods=1).std(ddof=0).fillna(0.0))
    return result


def build_training_targets(train: pd.DataFrame) -> pd.DataFrame:
    result = train.copy()
    max_cycle = result.groupby("unit")["cycle"].transform("max")
    result["rul"] = (max_cycle - result["cycle"]).clip(upper=RUL_CAP)
    return result


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "spearman": float(spearmanr(actual, predicted).statistic),
    }


def main() -> None:
    root = project_root()
    dataset_dir = root / DATASET_DIR
    train_path = dataset_dir / "train_FD001.txt"
    test_path = dataset_dir / "test_FD001.txt"
    rul_path = dataset_dir / "RUL_FD001.txt"
    model_path = root / "models" / "nasa_fd001_xgboost.pkl"
    output_path = root / "data" / "processed" / "nasa_fd001_rul_predictions.csv"
    source_paths = [train_path, test_path, rul_path, dataset_dir / "readme.txt"]
    before_hashes = {path: digest(path) for path in source_paths}

    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required FD001 file does not exist: {path}")

    train = build_training_targets(add_features(read_fd001(train_path)))
    test = add_features(read_fd001(test_path))
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])["rul"].to_numpy(dtype=float)
    if len(test_rul) != test["unit"].nunique():
        raise ValueError("RUL_FD001.txt count does not match test engine count.")

    feature_columns = [column for column in train.columns if column not in {"unit", "rul"}]
    train_units = sorted(train["unit"].unique())
    validation_units = [unit for unit in VALIDATION_UNITS if unit in train_units]
    fitting_units = [unit for unit in train_units if unit not in validation_units]
    fitting = train[train["unit"].isin(fitting_units)]
    validation = train[train["unit"].isin(validation_units)]

    validation_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )
    validation_model.fit(fitting[feature_columns], fitting["rul"], eval_set=[(validation[feature_columns], validation["rul"])], verbose=False)
    validation_predictions = np.maximum(validation_model.predict(validation[feature_columns]), 0.0)
    validation_metrics = metric_values(validation["rul"].to_numpy(), validation_predictions)

    final_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(train[feature_columns], train["rul"], verbose=False)
    test_last = test.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).sort_values("unit")
    test_predictions = np.maximum(final_model.predict(test_last[feature_columns]), 0.0)
    test_output = pd.DataFrame({
        "unit": test_last["unit"].astype(int).to_numpy(),
        "last_cycle": test_last["cycle"].astype(int).to_numpy(),
        "actual_rul_cycles": test_rul,
        "predicted_rul_cycles": test_predictions,
    })
    test_output["absolute_error_cycles"] = (test_output["actual_rul_cycles"] - test_output["predicted_rul_cycles"]).abs()
    test_metrics = metric_values(test_output["actual_rul_cycles"].to_numpy(), test_predictions)

    model_bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "dataset": "NASA C-MAPSS FD001",
        "target": "RUL in cycles",
        "rul_cap_cycles": RUL_CAP,
        "training_engines": train_units,
        "validation_engines": validation_units,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    test_output.to_csv(output_path, index=False)

    unchanged = all(digest(path) == value for path, value in before_hashes.items())
    print("============================================================")
    print("PHASE 14 - NASA FD001 XGBOOST RUL")
    print("============================================================")
    print()
    print("DATASET")
    print("-------")
    print("NASA C-MAPSS FD001 local files")
    print(f"Training engines: {train['unit'].nunique()}")
    print(f"Training rows: {len(train)}")
    print(f"Test engines: {test['unit'].nunique()}")
    print(f"Test rows: {len(test)}")
    print(f"Test RUL labels: {len(test_rul)}")
    print("Source files were not modified.")
    print()
    print("TARGET GENERATION")
    print("-----------------")
    print(f"Training RUL = max_cycle_per_engine - current_cycle, capped at {RUL_CAP:.0f} cycles.")
    print("Test targets use the supplied RUL_FD001.txt labels only for evaluation.")
    print()
    print("FEATURE ENGINEERING")
    print("-------------------")
    print(f"Features: {len(feature_columns)}")
    print("Chronological features: sensor differences and trailing 5-cycle mean/std; raw cycle is retained.")
    print("Unit identifiers are not model features.")
    print()
    print("ENGINE-LEVEL SPLIT")
    print("------------------")
    print(f"Training engines: {fitting_units}")
    print(f"Validation engines: {validation_units}")
    print("No engine appears in both training and validation: True")
    print()
    print("VALIDATION RESULTS")
    print("------------------")
    print(validation_metrics)
    print()
    print("OFFICIAL TEST RESULTS")
    print("---------------------")
    print("Model: XGBRegressor")
    print(f"MAE cycles: {test_metrics['mae']:.6f}")
    print(f"RMSE cycles: {test_metrics['rmse']:.6f}")
    print(f"R2: {test_metrics['r2']:.6f}")
    print(f"Spearman correlation: {test_metrics['spearman']:.6f}")
    print()
    print("PREDICTION RANGE")
    print("----------------")
    print(f"Minimum predicted RUL cycles: {test_predictions.min():.6f}")
    print(f"Maximum predicted RUL cycles: {test_predictions.max():.6f}")
    print(f"Finite and nonnegative predictions: {bool(np.isfinite(test_predictions).all() and (test_predictions >= 0).all())}")
    print()
    print("OUTPUT")
    print("------")
    print(f"Saved model: {model_path.relative_to(root)}")
    print(f"Saved predictions: {output_path.relative_to(root)}")
    print(f"Protected FD001 source files unchanged: {unchanged}")
    print()
    print("SCIENTIFIC INTERPRETATION")
    print("=========================")
    print("This is the official NASA C-MAPSS FD001 RUL model for this project.")
    print("Predictions are remaining operational cycles for the NASA FD001 test engines.")
    print("They are not predictions for the project's ESP32 motor recordings.")
    print("The existing LSTM remains a separate experimental proxy-RUL implementation.")
    print()
    print("PHASE 14 COMPLETE")
    print("=================")


if __name__ == "__main__":
    main()

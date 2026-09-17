"""
train_final.py
--------------
Trains the final XGBoost model on the full training split (80% of data)
with the exact same pipeline used during cross-validation, then saves
all artifacts needed for inference to models/.

Artifacts saved
---------------
models/final_pipeline.joblib   — fitted sklearn Pipeline (preprocessor + XGBoost)
models/feature_metadata.joblib — feature-group lists, ordinal orders, target info

Usage
-----
    python src/train_final.py
"""

import sys, warnings, json
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.preprocessing import load_and_clean
from src.features import engineer_features

# ── Constants ──────────────────────────────────────────────────────────────

SEED       = 42
DATA_PATH  = "laptop data/laptop_data.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Best hyperparameters from Phase 2 RandomizedSearchCV
BEST_PARAMS = {
    "n_estimators":     600,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.7,
    "colsample_bytree": 1.0,
    "random_state":     SEED,
    "verbosity":        0,
    "n_jobs":           -1,
}

# Feature groups (identical to 02_modeling.ipynb)
NUM_FEATURES = [
    "Inches", "ram_gb", "weight_kg", "total_pixels",
    "is_touchscreen", "is_ips", "cpu_speed_ghz",
    "ssd_gb", "hdd_gb", "flash_storage_gb", "total_storage_gb",
]
OHE_FEATURES = [
    "Company", "TypeName", "OpSys", "cpu_brand",
    "gpu_brand", "primary_storage_type",
]
CPU_ORDER = [["Atom", "AMD E-Series", "Celeron", "Pentium", "AMD A-Series",
              "ARM Cortex", "Core i3", "Core M", "Ryzen",
              "Core i5", "Core i7", "Xeon", "Other"]]
GPU_ORDER = [["Intel HD", "Intel UHD", "Intel Iris", "Intel Iris Plus",
              "Intel Iris Pro", "AMD Radeon R", "AMD Radeon", "AMD Radeon Pro",
              "AMD FirePro", "ARM Mali", "GeForce", "GeForce GTX",
              "GeForce RTX", "Quadro", "Other GPU"]]
ORD_CPU = ["cpu_family"]
ORD_GPU = ["gpu_family"]

ALL_FEATURES = NUM_FEATURES + OHE_FEATURES + ORD_CPU + ORD_GPU
TARGET       = "Price"


# ── Build pipeline ─────────────────────────────────────────────────────────

def build_final_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_FEATURES),
            ("ord_cpu", OrdinalEncoder(
                categories=CPU_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ORD_CPU),
            ("ord_gpu", OrdinalEncoder(
                categories=GPU_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ORD_GPU),
        ],
        remainder="drop",
    )
    model = XGBRegressor(**BEST_PARAMS)
    return Pipeline([("prep", preprocessor), ("model", model)])


# ── Train & save ───────────────────────────────────────────────────────────

def main():
    print("Loading and engineering features...")
    df_raw = load_and_clean(DATA_PATH)
    df     = engineer_features(df_raw)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # Same split as Phase 2 — training set only is used for fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    print(f"Train size : {X_train.shape[0]}")
    print(f"Test  size : {X_test.shape[0]}  (used ONLY for final metric verification)")

    # Fit on training set
    print("\nFitting final pipeline on training set...")
    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train_log)

    # Verify on test set (same as Phase 2 — should reproduce results)
    raw_preds      = pipeline.predict(X_test)
    preds_inr      = np.expm1(raw_preds)
    test_r2        = r2_score(y_test, preds_inr)
    test_mae       = mean_absolute_error(y_test, preds_inr)
    test_rmse      = np.sqrt(mean_squared_error(y_test, preds_inr))

    print(f"\nTest-set verification (should match Phase 2):")
    print(f"  R2   = {test_r2:.4f}")
    print(f"  MAE  = INR {test_mae:,.0f}")
    print(f"  RMSE = INR {test_rmse:,.0f}")

    # Save pipeline
    pipeline_path = MODELS_DIR / "final_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"\nSaved: {pipeline_path}")

    # Save feature metadata (all the lists needed for predict.py and notebooks)
    metadata = {
        "num_features":  NUM_FEATURES,
        "ohe_features":  OHE_FEATURES,
        "ord_cpu":       ORD_CPU,
        "ord_gpu":       ORD_GPU,
        "cpu_order":     CPU_ORDER,
        "gpu_order":     GPU_ORDER,
        "all_features":  ALL_FEATURES,
        "target":        TARGET,
        "target_transform": "log1p",
        "seed":          SEED,
        "model_type":    "XGBRegressor",
        "best_params":   BEST_PARAMS,
        "test_metrics": {
            "r2":   round(test_r2, 4),
            "mae":  round(test_mae, 0),
            "rmse": round(test_rmse, 0),
        },
        # CV metrics from Phase 2
        "cv_metrics": {
            "r2":   0.8364,
            "mae":  9273.0,
            "rmse": 14916.0,
        },
        "train_size": int(X_train.shape[0]),
        "test_size":  int(X_test.shape[0]),
    }
    meta_path = MODELS_DIR / "feature_metadata.joblib"
    joblib.dump(metadata, meta_path)
    print(f"Saved: {meta_path}")

    # Also store the train/test split indices + X_test for the interpretation notebook
    test_data = {
        "X_test":  X_test.reset_index(drop=True),
        "y_test":  y_test.reset_index(drop=True),
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        # Store the original df rows that correspond to test, for error analysis
        "df_test": df.iloc[X_test.index].reset_index(drop=True),
    }
    split_path = MODELS_DIR / "train_test_split.joblib"
    joblib.dump(test_data, split_path)
    print(f"Saved: {split_path}")

    print("\nAll artifacts saved to models/")
    return pipeline, metadata, test_data


if __name__ == "__main__":
    main()

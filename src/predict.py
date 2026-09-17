"""
predict.py
----------
Reusable prediction pipeline for laptop price estimation.

The pipeline applies the exact same feature engineering and preprocessing
used during model training, predicts log1p(Price) with the saved XGBoost
model, and inverse-transforms the result to INR using expm1.

Usage (single laptop)
---------------------
    from src.predict import predict_price

    result = predict_price({
        "Company":          "Dell",
        "TypeName":         "Notebook",
        "Inches":           15.6,
        "ScreenResolution": "Full HD 1920x1080",
        "Cpu":              "Intel Core i5 7200U 2.5GHz",
        "Ram":              "8GB",
        "Memory":           "256GB SSD",
        "Gpu":              "Intel HD Graphics 620",
        "OpSys":            "Windows 10",
        "Weight":           "1.86kg",
    })
    print(result)
    # {"predicted_price_inr": 32450.0, "predicted_log_price": 10.387, ...}

Usage (batch / DataFrame)
--------------------------
    from src.predict import predict_price_batch
    import pandas as pd

    df_new = pd.read_csv("new_laptops.csv")
    prices = predict_price_batch(df_new)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union

# ── Locate artifacts ───────────────────────────────────────────────────────

_MODELS_DIR      = Path(__file__).parent.parent / "models"
_PIPELINE_PATH   = _MODELS_DIR / "final_pipeline.joblib"
_METADATA_PATH   = _MODELS_DIR / "feature_metadata.joblib"

# Lazy-loaded singletons — loaded once on first call
_pipeline = None
_metadata = None


def _load_artifacts():
    """Load pipeline and metadata from disk (cached after first call)."""
    global _pipeline, _metadata
    if _pipeline is None:
        if not _PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {_PIPELINE_PATH}\n"
                "Run `python src/train_final.py` to generate it."
            )
        _pipeline = joblib.load(_PIPELINE_PATH)
        _metadata = joblib.load(_METADATA_PATH)
    return _pipeline, _metadata


# ── Input validation ───────────────────────────────────────────────────────

# Raw columns expected in input (matching the original CSV schema)
_REQUIRED_RAW_COLUMNS = [
    "Company", "TypeName", "Inches",
    "ScreenResolution", "Cpu", "Ram",
    "Memory", "Gpu", "OpSys", "Weight",
]

_VALID_RAM_PATTERN     = r"^\d+GB$"
_VALID_WEIGHT_PATTERN  = r"^\d+\.?\d*kg$"
_VALID_RESOLUTION_PATTERN = r"\d{3,4}x\d{3,4}"


def _validate_input(df: pd.DataFrame) -> None:
    """
    Raise ValueError with a descriptive message if required columns are
    missing or clearly malformed.
    """
    import re
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check Ram format
    bad_ram = df[~df["Ram"].str.match(_VALID_RAM_PATTERN, na=False)]
    if not bad_ram.empty:
        raise ValueError(
            f"Ram must be like '8GB', '16GB'. Got: {bad_ram['Ram'].tolist()}"
        )

    # Check Weight format
    bad_wt = df[~df["Weight"].str.match(_VALID_WEIGHT_PATTERN, na=False)]
    if not bad_wt.empty:
        raise ValueError(
            f"Weight must be like '1.86kg', '2.1kg'. Got: {bad_wt['Weight'].tolist()}"
        )

    # Check ScreenResolution contains a resolution
    bad_res = df[~df["ScreenResolution"].str.contains(
        _VALID_RESOLUTION_PATTERN, na=False)]
    if not bad_res.empty:
        raise ValueError(
            f"ScreenResolution must contain a pattern like '1920x1080'. "
            f"Got: {bad_res['ScreenResolution'].tolist()}"
        )


# ── Feature engineering (mirrors src/features.py exactly) ─────────────────

def _apply_feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing + feature engineering used during training.
    Imports from src.preprocessing and src.features to guarantee consistency.
    """
    import sys
    from pathlib import Path
    src_root = str(Path(__file__).parent.parent)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    from src.preprocessing import load_and_clean, OPSYS_MERGE_MAP, RARE_OPSYS_THRESHOLD, RARE_OPSYS_LABEL
    from src.features import engineer_features

    # --- Manual cleaning (mirrors load_and_clean but without re-reading CSV) ---
    df = df_raw.copy()

    # Strip whitespace
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # OS consolidation
    df["OpSys"] = df["OpSys"].replace(OPSYS_MERGE_MAP)

    # Group rare OS (apply same threshold — rare categories become "Other OS")
    # For a single prediction row, we cannot compute "rare" relative to training,
    # so we map known rare values explicitly using the training-set categories.
    _KNOWN_OS = {"Windows 10", "No OS", "Linux", "Windows 7",
                 "Chrome OS", "macOS", "Other OS"}
    df["OpSys"] = df["OpSys"].where(df["OpSys"].isin(_KNOWN_OS), other=RARE_OPSYS_LABEL)

    # Feature engineering
    df = engineer_features(df)
    return df


# ── Core prediction logic ──────────────────────────────────────────────────

def predict_price(
    input_data: Union[dict, pd.Series, pd.DataFrame],
    return_details: bool = False,
) -> Union[float, dict]:
    """
    Predict laptop price in INR for one or more laptops.

    Parameters
    ----------
    input_data : dict, pd.Series, or pd.DataFrame
        Raw laptop specification(s) using the original CSV column schema.
        Required columns: Company, TypeName, Inches, ScreenResolution,
        Cpu, Ram, Memory, Gpu, OpSys, Weight.

    return_details : bool, default False
        If True, return a dict with the predicted price plus intermediate
        values (log-price, confidence context).
        If False, return a float (single row) or list of floats (batch).

    Returns
    -------
    float or list[float] or dict
        Predicted price(s) in INR.

    Notes
    -----
    The model was trained on log1p(Price). This function:
    1. Builds raw input DataFrame
    2. Applies feature engineering (same as training)
    3. Selects the exact 19 model features
    4. Calls the fitted sklearn pipeline (preprocessor + XGBoost)
    5. Applies np.expm1() to convert log-price → INR
    """
    pipeline, metadata = _load_artifacts()
    all_features = metadata["all_features"]

    # Normalise input to DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
        single = True
    elif isinstance(input_data, pd.Series):
        df_input = pd.DataFrame([input_data.to_dict()])
        single = True
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        single = False
    else:
        raise TypeError(
            f"input_data must be dict, pd.Series, or pd.DataFrame, "
            f"got {type(input_data)}"
        )

    # Validate
    _validate_input(df_input)

    # Feature engineering
    df_engineered = _apply_feature_engineering(df_input)

    # Select model features (same order as training)
    X = df_engineered[all_features]

    # Predict (pipeline handles its own preprocessing internally)
    log_preds = pipeline.predict(X)
    price_preds = np.expm1(log_preds)

    if return_details:
        rows = []
        for i in range(len(df_input)):
            rows.append({
                "predicted_price_inr":  round(float(price_preds[i]), 2),
                "predicted_log_price":  round(float(log_preds[i]), 6),
                "model":                metadata["model_type"],
                "target_transform":     metadata["target_transform"],
                "note": (
                    "Price predicted via log1p transform. "
                    "Inverse-transformed with expm1 to return INR."
                ),
            })
        return rows[0] if single else rows

    if single:
        return round(float(price_preds[0]), 2)
    return [round(float(p), 2) for p in price_preds]


def predict_price_batch(df: pd.DataFrame) -> list:
    """
    Convenience wrapper for batch prediction from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the same raw columns as the training CSV.

    Returns
    -------
    list[float]
        Predicted prices in INR, one per row.
    """
    return predict_price(df, return_details=False)


# ── CLI quick-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sample inputs that span different price segments
    samples = [
        {
            "name": "Budget Acer Notebook",
            "input": {
                "Company": "Acer", "TypeName": "Notebook", "Inches": 15.6,
                "ScreenResolution": "1366x768",
                "Cpu": "Intel Core i3 6006U 2GHz", "Ram": "4GB",
                "Memory": "500GB HDD", "Gpu": "Intel HD Graphics 520",
                "OpSys": "Windows 10", "Weight": "2.1kg",
            },
        },
        {
            "name": "Mid-range Dell Ultrabook",
            "input": {
                "Company": "Dell", "TypeName": "Ultrabook", "Inches": 13.3,
                "ScreenResolution": "Full HD 1920x1080",
                "Cpu": "Intel Core i5 7200U 2.5GHz", "Ram": "8GB",
                "Memory": "256GB SSD", "Gpu": "Intel HD Graphics 620",
                "OpSys": "Windows 10", "Weight": "1.35kg",
            },
        },
        {
            "name": "Gaming MSI",
            "input": {
                "Company": "MSI", "TypeName": "Gaming", "Inches": 15.6,
                "ScreenResolution": "Full HD 1920x1080",
                "Cpu": "Intel Core i7 7700HQ 2.8GHz", "Ram": "16GB",
                "Memory": "256GB SSD +  1TB HDD",
                "Gpu": "Nvidia GeForce GTX 1060",
                "OpSys": "Windows 10", "Weight": "2.2kg",
            },
        },
    ]

    print("=" * 55)
    print("  SAMPLE PREDICTIONS")
    print("=" * 55)
    for s in samples:
        result = predict_price(s["input"], return_details=True)
        print(f"\n  {s['name']}")
        print(f"    Predicted price : INR {result['predicted_price_inr']:>10,.2f}")
        print(f"    log1p(Price)    : {result['predicted_log_price']:.4f}")
    print("=" * 55)

"""
features.py
-----------
Feature-engineering pipeline for the Laptop Price Prediction dataset.

Usage
-----
    from src.preprocessing import load_and_clean
    from src.features import engineer_features

    df_clean   = load_and_clean("laptop data/laptop_data.csv")
    df_feats   = engineer_features(df_clean)

Notes
-----
- Only reliable, rule-based extractions are performed.
- No target-based transformations (no leakage).
- Original raw columns are preserved alongside new engineered columns so
  the caller can decide which to use / drop for modelling.
- All functions are deterministic and safe to apply to train/test splits
  independently.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps and return an enriched DataFrame.

    New columns added
    -----------------
    RAM
        ram_gb                : int   — RAM in gigabytes

    Weight
        weight_kg             : float — Weight in kilograms

    Screen Resolution
        resolution_width      : int   — Horizontal pixel count
        resolution_height     : int   — Vertical pixel count
        total_pixels          : int   — width × height
        is_touchscreen        : int   — 1 if touchscreen, else 0
        is_ips                : int   — 1 if IPS panel, else 0

    CPU
        cpu_brand             : str   — Intel / AMD / Samsung / Other
        cpu_family            : str   — Core i7 / Core i5 / Celeron / …
        cpu_speed_ghz         : float — Clock speed in GHz (NaN if missing)

    GPU
        gpu_brand             : str   — Intel / Nvidia / AMD / ARM / Other
        gpu_family            : str   — HD Graphics / GeForce / Radeon / …

    Memory
        ssd_gb                : float — Total SSD storage in GB
        hdd_gb                : float — Total HDD storage in GB
        flash_storage_gb      : float — Total Flash/eMMC storage in GB
        hybrid_gb             : float — Total Hybrid storage in GB
        total_storage_gb      : float — Sum of all storage tiers in GB
        primary_storage_type  : str   — SSD / HDD / Flash / Hybrid / Mixed

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_and_clean()``.

    Returns
    -------
    pd.DataFrame
        Original columns retained; engineered columns appended.
    """
    df = df.copy()

    df = _extract_ram(df)
    df = _extract_weight(df)
    df = _extract_resolution(df)
    df = _extract_cpu_features(df)
    df = _extract_gpu_features(df)
    df = _extract_memory_features(df)

    return df


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _extract_ram(df: pd.DataFrame) -> pd.DataFrame:
    """Strip 'GB' suffix and convert Ram to integer gigabytes."""
    df["ram_gb"] = (
        df["Ram"]
        .str.replace("GB", "", regex=False)
        .str.strip()
        .astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# Weight
# ---------------------------------------------------------------------------

def _extract_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Strip 'kg' suffix and convert Weight to float kilograms."""
    df["weight_kg"] = (
        df["Weight"]
        .str.replace("kg", "", regex=False)
        .str.strip()
        .astype(float)
    )
    return df


# ---------------------------------------------------------------------------
# Screen Resolution
# ---------------------------------------------------------------------------

def _extract_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse ScreenResolution strings such as:
        'Full HD 1920x1080'
        'IPS Panel Full HD / Touchscreen 1920x1080'
        'IPS Panel Retina Display 2560x1600'

    Extracts width, height, total_pixels, is_touchscreen, is_ips.
    """
    sr = df["ScreenResolution"]

    # Resolution (width x height)
    res_extracted = sr.str.extract(r"(\d{3,4})x(\d{3,4})", expand=True)
    df["resolution_width"]  = res_extracted[0].astype(int)
    df["resolution_height"] = res_extracted[1].astype(int)
    df["total_pixels"]      = df["resolution_width"] * df["resolution_height"]

    # Binary flags
    df["is_touchscreen"] = sr.str.contains("Touchscreen", case=False, na=False).astype(int)
    df["is_ips"]         = sr.str.contains("IPS", case=False, na=False).astype(int)

    return df


# ---------------------------------------------------------------------------
# CPU Features
# ---------------------------------------------------------------------------

_CPU_FAMILY_PATTERNS = [
    (r"Core [im]7",  "Core i7"),
    (r"Core [im]5",  "Core i5"),
    (r"Core [im]3",  "Core i3"),
    (r"Core M",      "Core M"),
    (r"Xeon",        "Xeon"),
    (r"Celeron",     "Celeron"),
    (r"Pentium",     "Pentium"),
    (r"Atom",        "Atom"),
    (r"Ryzen",       "Ryzen"),
    (r"\bA\d\b",     "AMD A-Series"),  # AMD A4, A6, A9, A10 …
    (r"\bE\d\b",     "AMD E-Series"),  # AMD E2 …
    (r"Cortex",      "ARM Cortex"),
]


def _classify_cpu_family(cpu_str: str) -> str:
    for pattern, label in _CPU_FAMILY_PATTERNS:
        if re.search(pattern, cpu_str, re.IGNORECASE):
            return label
    return "Other"


def _extract_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cpu_brand, cpu_family, cpu_speed_ghz from the Cpu column.

    cpu_brand:
        First token of the Cpu string (Intel / AMD / Samsung / …).
    cpu_family:
        Processor tier mapped from regex patterns (Core i7 / Celeron / …).
    cpu_speed_ghz:
        Last numeric value followed by 'GHz' in the string.
        Set to NaN when not reliably extractable.
    """
    df["cpu_brand"] = df["Cpu"].str.split().str[0]

    df["cpu_family"] = df["Cpu"].apply(_classify_cpu_family)

    ghz_series = df["Cpu"].str.extract(r"(\d+\.?\d*)\s*GHz", expand=False)
    df["cpu_speed_ghz"] = pd.to_numeric(ghz_series, errors="coerce")

    return df


# ---------------------------------------------------------------------------
# GPU Features
# ---------------------------------------------------------------------------

_GPU_FAMILY_PATTERNS = [
    (r"GeForce GTX",    "GeForce GTX"),
    (r"GeForce RTX",    "GeForce RTX"),
    (r"GeForce\b",      "GeForce"),     # catch-all for other GeForce models
    (r"Quadro",         "Quadro"),
    (r"Iris\s+Plus",    "Intel Iris Plus"),
    (r"Iris\s+Pro",     "Intel Iris Pro"),
    (r"\bIris\b",       "Intel Iris"),
    (r"UHD Graphics",   "Intel UHD"),
    (r"HD Graphics",    "Intel HD"),
    (r"Radeon Pro",     "AMD Radeon Pro"),
    (r"Radeon R",       "AMD Radeon R"),
    (r"Radeon\b",       "AMD Radeon"),
    (r"FirePro",        "AMD FirePro"),
    (r"Mali",           "ARM Mali"),
]


def _classify_gpu_family(gpu_str: str) -> str:
    for pattern, label in _GPU_FAMILY_PATTERNS:
        if re.search(pattern, gpu_str, re.IGNORECASE):
            return label
    return "Other GPU"


def _extract_gpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract gpu_brand and gpu_family from the Gpu column.

    gpu_brand:
        First token of the Gpu string (Intel / Nvidia / AMD / ARM / …).
    gpu_family:
        Product-line category mapped from regex patterns.
    """
    df["gpu_brand"] = df["Gpu"].str.split().str[0]

    df["gpu_family"] = df["Gpu"].apply(_classify_gpu_family)

    return df


# ---------------------------------------------------------------------------
# Memory / Storage Features
# ---------------------------------------------------------------------------

def _parse_storage_string(mem_str: str) -> dict:
    """
    Parse a Memory value such as:
        '256GB SSD'
        '512GB SSD +  1TB HDD'
        '128GB Flash Storage'
        '1.0TB Hybrid'
        '508GB Hybrid'

    Returns a dict with keys: ssd_gb, hdd_gb, flash_gb, hybrid_gb.
    """
    def _to_gb(value: str, unit: str) -> float:
        v = float(value)
        return v * 1000 if "TB" in unit else v

    result = {"ssd_gb": 0.0, "hdd_gb": 0.0, "flash_gb": 0.0, "hybrid_gb": 0.0}

    parts = [p.strip() for p in mem_str.split("+")]
    for part in parts:
        m = re.match(r"(\d+\.?\d*)\s*(GB|TB)\s+(.*)", part, re.IGNORECASE)
        if not m:
            continue
        size = _to_gb(m.group(1), m.group(2).upper())
        desc = m.group(3).lower()

        if "ssd" in desc:
            result["ssd_gb"] += size
        elif "hdd" in desc:
            result["hdd_gb"] += size
        elif "flash" in desc:
            result["flash_gb"] += size
        elif "hybrid" in desc:
            result["hybrid_gb"] += size

    return result


def _primary_storage_type(row: pd.Series) -> str:
    """
    Determine the primary storage type from individual storage columns.
    'Primary' = largest contributor, with a tie-break priority SSD > HDD > Flash > Hybrid.
    If more than one type is present, label as 'Mixed'.
    """
    types_present = sum([
        row["ssd_gb"] > 0,
        row["hdd_gb"] > 0,
        row["flash_storage_gb"] > 0,
        row["hybrid_gb"] > 0,
    ])
    if types_present > 1:
        return "Mixed"
    if row["ssd_gb"] > 0:
        return "SSD"
    if row["hdd_gb"] > 0:
        return "HDD"
    if row["flash_storage_gb"] > 0:
        return "Flash"
    if row["hybrid_gb"] > 0:
        return "Hybrid"
    return "Unknown"


def _extract_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Memory column into numeric storage features.

    New columns: ssd_gb, hdd_gb, flash_storage_gb, hybrid_gb,
                 total_storage_gb, primary_storage_type.
    """
    parsed = df["Memory"].apply(_parse_storage_string).apply(pd.Series)
    parsed = parsed.rename(columns={"flash_gb": "flash_storage_gb"})

    df["ssd_gb"]           = parsed["ssd_gb"]
    df["hdd_gb"]           = parsed["hdd_gb"]
    df["flash_storage_gb"] = parsed["flash_storage_gb"]
    df["hybrid_gb"]        = parsed["hybrid_gb"]
    df["total_storage_gb"] = (
        df["ssd_gb"] + df["hdd_gb"] + df["flash_storage_gb"] + df["hybrid_gb"]
    )
    df["primary_storage_type"] = df[
        ["ssd_gb", "hdd_gb", "flash_storage_gb", "hybrid_gb"]
    ].apply(_primary_storage_type, axis=1)

    return df


# ---------------------------------------------------------------------------
# Convenience: list of all engineered feature names
# ---------------------------------------------------------------------------

ENGINEERED_FEATURES = [
    # RAM
    "ram_gb",
    # Weight
    "weight_kg",
    # Screen
    "resolution_width", "resolution_height", "total_pixels",
    "is_touchscreen", "is_ips",
    # CPU
    "cpu_brand", "cpu_family", "cpu_speed_ghz",
    # GPU
    "gpu_brand", "gpu_family",
    # Storage
    "ssd_gb", "hdd_gb", "flash_storage_gb", "hybrid_gb",
    "total_storage_gb", "primary_storage_type",
]

# Passthrough categorical features (already clean after preprocessing)
CATEGORICAL_PASSTHROUGH = [
    "Company", "TypeName", "OpSys",
]

# Already-numeric raw features
NUMERIC_PASSTHROUGH = [
    "Inches",
]

# Target
TARGET = "Price"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import load_and_clean

    df_raw   = load_and_clean("laptop data/laptop_data.csv")
    df_feats = engineer_features(df_raw)

    print(f"Shape after feature engineering : {df_feats.shape}")
    print(f"\nEngineered columns added:")
    for col in ENGINEERED_FEATURES:
        dtype = df_feats[col].dtype
        n_null = df_feats[col].isna().sum()
        print(f"  {col:<25} dtype={dtype}   nulls={n_null}")

    print("\nPrimary storage type distribution:")
    print(df_feats["primary_storage_type"].value_counts())

    print("\nCPU family distribution:")
    print(df_feats["cpu_family"].value_counts())

    print("\nGPU family distribution:")
    print(df_feats["gpu_family"].value_counts())

    print("\nSample rows (engineered cols only):")
    print(df_feats[ENGINEERED_FEATURES].head(5).to_string())

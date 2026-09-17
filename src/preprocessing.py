"""
preprocessing.py
----------------
Data cleaning pipeline for the Laptop Price Prediction dataset.

Usage
-----
    from src.preprocessing import load_and_clean

    df_clean = load_and_clean("laptop data/laptop_data.csv")

Notes
-----
- Does NOT modify the original CSV.
- Does NOT perform any target-based feature engineering.
- Safe to call on train/test splits independently.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = "laptop data/laptop_data.csv"

# Operating-system consolidation map
# Merge near-duplicate categories into a canonical label.
OPSYS_MERGE_MAP = {
    "Mac OS X": "macOS",          # older label for the same platform
    "Windows 10 S": "Windows 10", # S-mode is still Windows 10
}

# Rare OS categories (very few rows) → grouped to avoid sparsity in modelling.
# Kept separate from merge so the analyst can decide later whether to use
# the grouped or ungrouped version.
RARE_OPSYS_THRESHOLD = 10  # categories with fewer rows than this get grouped
RARE_OPSYS_LABEL = "Other OS"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and return a cleaned DataFrame.

    Steps
    -----
    1. Load CSV.
    2. Drop the redundant index column ``Unnamed: 0``.
    3. Strip leading/trailing whitespace from all string columns.
    4. Consolidate near-duplicate OS labels.
    5. Group very rare OS categories under ``Other OS``.
    6. Preserve all rows (no rows are dropped for being outliers or rare).

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    df = pd.read_csv(filepath)

    # 1. Drop redundant index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 3. Merge near-duplicate OS labels (macOS / Mac OS X, Windows 10 S)
    df["OpSys"] = df["OpSys"].replace(OPSYS_MERGE_MAP)

    # 4. Group very rare OS categories
    df = _group_rare_categories(df, col="OpSys",
                                threshold=RARE_OPSYS_THRESHOLD,
                                rare_label=RARE_OPSYS_LABEL)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_rare_categories(
    df: pd.DataFrame,
    col: str,
    threshold: int,
    rare_label: str,
) -> pd.DataFrame:
    """
    Replace categories with fewer than *threshold* occurrences with *rare_label*.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to group.
    threshold : int
        Minimum count to retain a category label.
    rare_label : str
        Replacement label for rare categories.

    Returns
    -------
    pd.DataFrame
    """
    counts = df[col].value_counts()
    rare_cats = counts[counts < threshold].index.tolist()
    if rare_cats:
        df = df.copy()
        df[col] = df[col].where(~df[col].isin(rare_cats), other=rare_label)
    return df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    print(f"Shape after cleaning : {df.shape}")
    print(f"Columns              : {df.columns.tolist()}")
    print(f"\nOpSys value counts:\n{df['OpSys'].value_counts()}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

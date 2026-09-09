import numpy as np
import pandas as pd

MISSING_VALUES = {
    "missing","Missing","MISSING",
    "none","None","NONE",
    "null","Null","NULL",
    "nan","NaN","NA","N/A","n/a",
    "", " ", "-", "--",
}

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def clean_text_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind not in "OUS":
        return s
    return s.astype(str).str.strip().replace(MISSING_VALUES, np.nan)

def clean_output_column(y: pd.Series, valid_labels) -> pd.Series:
    valid = set(valid_labels)
    y_clean = []
    for v in y.astype(str).str.strip():
        if v in valid:
            y_clean.append(v)
        elif v in MISSING_VALUES:
            y_clean.append(np.nan)
        else:
            y_clean.append(np.nan)
    return pd.Series(y_clean, index=y.index)

def clean_numeric_like_strings(df: pd.DataFrame, max_unique_ratio: float = 0.50) -> pd.DataFrame:
    """
    Convert numeric-like strings to floats for object columns that look numeric.
    Keeps true categorical text intact.

    Heuristic:
    - Only consider object columns.
    - Attempt numeric conversion; if enough values become numeric, convert column.
    - Avoid converting columns that are mostly unique strings (likely IDs/categories).
    """
    df2 = df.copy()
    nrows = max(len(df2), 1)

    for col in df2.columns:
        if df2[col].dtype.kind != "O":
            continue

        s = df2[col].astype(str).str.strip()

        # quick skip: high uniqueness often means categorical/IDs
        nunique = s.nunique(dropna=True)
        if (nunique / nrows) > max_unique_ratio:
            continue

        converted = pd.to_numeric(s, errors="coerce")

        # if conversion yields enough numeric values, accept conversion
        non_na_before = s.replace("nan", np.nan).notna().sum()
        non_na_after = converted.notna().sum()

        if non_na_before == 0:
            continue

        # require at least 70% of non-missing to be numeric
        if (non_na_after / non_na_before) >= 0.70:
            df2[col] = converted

    return df2

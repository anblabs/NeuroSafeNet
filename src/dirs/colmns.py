def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_text_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind not in "OUS":
        return s
    MISSING_VALUES = {
        "Null"
    }
    return s.astype(str).str.strip().replace(MISSING_VALUES, np.nan)

def strip_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")
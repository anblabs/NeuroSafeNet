import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ninflam.core.registry import get

# Ensure plugins are imported so registration happens
from ninflam.pre_train.plugins import scaler_robust  # noqa: F401

log = logging.getLogger(__name__)

def build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    scaler_cls = get("scaler", name)
    return scaler_cls().build()

def build_preprocessor(X_df: pd.DataFrame, scaler_name: str, onehot_dense: bool):
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    scaler = build_scaler(scaler_name)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    # sklearn compatibility: sparse_output introduced in newer versions
    if onehot_dense:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore")

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    log.info("Numeric cols: %d | Categorical cols: %d", len(numeric_cols), len(categorical_cols))
    return pre, numeric_cols, categorical_cols

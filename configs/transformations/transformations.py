from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, padas as pd
def build_preprocessor(X_df: pd.DataFrame):
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    full = ColumnTransformer(
        [
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return full, numeric_cols, categorical_cols



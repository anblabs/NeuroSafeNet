
def clean_output_column(y):
    import numpy as np
    import pandas as pd

    # Values that should be considered missing
    MISSING = {
        "Missing"
    }

    # Allowed valid categories
    # VALID = {"Good", "Bad"}  # <--- you must confirm these
    VALID = {"No", "Proliferative", "Moderate", "Severe"}

    y_clean = []
    for v in y.astype(str).str.strip():
        if v in VALID:
            y_clean.append(v)
        elif v in MISSING:
            y_clean.append(np.nan)
        else:
            # All other weird/free-text values = convert to NaN
            y_clean.append(np.nan)

    return pd.Series(y_clean)

    def _clean(self, df: pd.DataFrame):
        df2 = df.copy()

        # clean text series
        for col in df2.columns:
            df2[col] = clean_text_series(df2[col])

        # convert numeric-like strings safely
        df2 = clean_numeric_like_strings(df2)

        if self.data_cfg.output_col not in df2.columns:
            raise KeyError(f"Missing label column: {self.data_cfg.output_col}")

        # Keep "cleaned" snapshot (like your CLEAN_CSV)
        df_cleaned = df2.copy()

        X_df = df2[self.data_cfg.input_cols].copy()
        y_raw = df2[self.data_cfg.output_col]
        y = clean_output_column(y_raw, self.data_cfg.valid_labels)

        # impute y before splitting (same as your code)
        imp_y = SimpleImputer(strategy="most_frequent")
        y_imp = imp_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        save_joblib(self.art.path("imputer_y.joblib"), imp_y)

        y_series = pd.Series(y_imp, index=y.index)

        return X_df, y_series, df_cleaned

def _build_imputer_X(self, X_df, num_cols, cat_cols) -> ColumnTransformer:
    from sklearn.impute import SimpleImputer
    imputer = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
        ],
        remainder="drop"
    )
    imputer.fit(X_df)
    return imputer
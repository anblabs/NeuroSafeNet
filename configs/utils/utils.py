def _save(self, bundle, df_cleaned, X_df, y_series):
    # Core artifacts
    save_joblib(self.art.path("preprocessor.joblib"), bundle["preprocessor"])
    save_npz(self.art.path("splits.npz"), **bundle["splits"])
    save_json(self.art.path("classes.json"), {"classes": bundle["classes"]})

    # Feature names
    if self.prep_cfg.save_feature_names:
        np.save(self.art.path("feature_names.npy"), np.array(bundle["feature_names"], dtype=object))
        pd.DataFrame({"feature_name": bundle["feature_names"]}).to_csv(self.art.path("feature_names.csv"), index=False)

    # Cleaned CSV (like your CLEAN_CSV)
    if self.io_cfg.save_cleaned_csv:
        save_csv(self.art.path("cleaned.csv"), df_cleaned)

    # IN+OUT cleaned CSV
    if self.io_cfg.save_in_out_clean_csv:
        in_out_cols = list(self.data_cfg.input_cols) + [self.data_cfg.output_col]
        save_csv(self.art.path("in_out_clean.csv"), df_cleaned[in_out_cols].copy())

def _load(self) -> pd.DataFrame:
    loader = ExcelLoader(self.data_cfg.file_name, self.data_cfg.sheet)
    df = loader.load()
    return strip_columns(df)
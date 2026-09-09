import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from ninflam.config import DataCfg, IOCfg, PreprocessingCfg, SplitCfg
from ninflam.data.cleaning import (
    clean_numeric_like_strings,
    clean_output_column,
    clean_text_series,
    strip_columns,
)
from ninflam.data.excel_loader import ExcelLoader
from ninflam.io.artifacts import ArtifactManager
from ninflam.io.serializer import save_csv, save_json, save_joblib, save_npz
from ninflam.pre_train.builders import build_preprocessor
from ninflam.pre_train.feature_names import get_expanded_feature_names
from ninflam.splitting.stratified import split_train_internal_external


log = logging.getLogger(__name__)


class NInflamPipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_cfg = DataCfg(**OmegaConf.to_container(cfg.data, resolve=True))
        self.split_cfg = SplitCfg(**OmegaConf.to_container(cfg.split, resolve=True))
        self.prep_cfg = PreprocessingCfg(**OmegaConf.to_container(cfg.preprocessing, resolve=True))
        self.io_cfg = IOCfg(**OmegaConf.to_container(cfg.io, resolve=True))

        self.base = Path(self.data_cfg.file_name).stem
        self.outdir = Path(self.io_cfg.outputs_dirname) / f"preprocessed_{self.io_cfg.outdir_tag}" / self.base
        self.art = ArtifactManager(str(self.outdir))

    def run(self) -> None:
        df = self._load()
        X_df, y_series, df_cleaned = self._clean(df)
        X_tr_df, X_int_df, X_ext_df, y_tr, y_int, y_ext = self._split(X_df, y_series)
        bundle = self._preprocess(X_tr_df, X_int_df, X_ext_df, y_tr, y_int, y_ext)
        self._save(bundle, df_cleaned, X_df, y_series)
        log.info("Done. Outputs in: %s", self.outdir)

    def _load(self) -> pd.DataFrame:
        loader = ExcelLoader(self.data_cfg.file_name, self.data_cfg.sheet)
        return loader.load()

    def _clean(self, df: pd.DataFrame):
        df = strip_columns(df.copy())

        for col in df.columns:
            df[col] = clean_text_series(df[col])

        df = clean_numeric_like_strings(df)
        self._validate_columns(df)

        X_df = df[self.data_cfg.input_cols].copy()
        y_series = clean_output_column(df[self.data_cfg.output_col], self.data_cfg.valid_labels)

        missing_labels = y_series.isna()
        if missing_labels.any():
            if self.data_cfg.label_missing_strategy != "drop":
                raise ValueError(
                    "Only label_missing_strategy='drop' is supported in the package pipeline. "
                    "Missing labels should not be imputed for supervised model training."
                )
            dropped = int(missing_labels.sum())
            log.warning("Dropping %d rows with missing or invalid target labels.", dropped)
            X_df = X_df.loc[~missing_labels].copy()
            y_series = y_series.loc[~missing_labels].copy()
            df = df.loc[~missing_labels].copy()

        return X_df, y_series, df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = list(self.data_cfg.input_cols) + [self.data_cfg.output_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(
                "Configured columns are missing from the dataset: "
                f"{missing}. Available columns: {df.columns.tolist()}"
            )

    def _split(self, X_df: pd.DataFrame, y_series: pd.Series):
        return split_train_internal_external(
            X_df,
            y_series,
            external_size=self.split_cfg.external_test_size,
            internal_size=self.split_cfg.internal_test_size,
            random_state=self.split_cfg.random_state,
            min_class_count_for_stratify=self.split_cfg.min_class_count_for_stratify,
        )

    def _preprocess(self, X_tr_df, X_int_df, X_ext_df, y_tr, y_int, y_ext):
        pre, num_cols, cat_cols = build_preprocessor(
            X_tr_df,
            scaler_name=self.prep_cfg.scaler,
            onehot_dense=self.prep_cfg.onehot_dense,
        )
        pre.fit(X_tr_df)

        feature_names = get_expanded_feature_names(pre, num_cols, cat_cols)

        X_tr = pre.transform(X_tr_df)
        X_int = pre.transform(X_int_df)
        X_ext = pre.transform(X_ext_df)

        le = LabelEncoder()
        le.fit(pd.concat([pd.Series(y_tr), pd.Series(y_int), pd.Series(y_ext)], axis=0))

        return {
            "preprocessor": pre,
            "label_encoder": le,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "feature_names": feature_names,
            "classes": le.classes_.tolist(),
            "raw_splits": {
                "X_tr": X_tr_df,
                "X_int": X_int_df,
                "X_ext": X_ext_df,
                "y_tr": y_tr,
                "y_int": y_int,
                "y_ext": y_ext,
            },
            "splits": {
                "X_tr": X_tr,
                "y_tr": le.transform(y_tr),
                "X_int": X_int,
                "y_int": le.transform(y_int),
                "X_ext": X_ext,
                "y_ext": le.transform(y_ext),
            },
        }

    def _save(self, bundle, df_cleaned: pd.DataFrame, X_df: pd.DataFrame, y_series: pd.Series) -> None:
        save_json(
            self.art.path("config_input_output_cols.json"),
            {
                "output_col": self.data_cfg.output_col,
                "input_cols": list(self.data_cfg.input_cols),
                "valid_labels": list(self.data_cfg.valid_labels),
            },
        )

        if self.io_cfg.save_cleaned_csv:
            save_csv(self.art.path("cleaned.csv"), df_cleaned)

        in_out_clean = X_df.copy()
        in_out_clean[self.data_cfg.output_col] = y_series
        if self.io_cfg.save_in_out_clean_csv:
            save_csv(self.art.path("in_out_clean.csv"), in_out_clean)

        imputed_X = self._impute_for_review(bundle["preprocessor"], X_df)
        if self.io_cfg.save_imputed_full_csv:
            save_csv(self.art.path("imputed_inputs.csv"), imputed_X)

        in_out_imputed = imputed_X.copy()
        in_out_imputed[self.data_cfg.output_col] = y_series.to_numpy()
        if self.io_cfg.save_in_out_imputed_csv:
            save_csv(self.art.path("in_out_imputed.csv"), in_out_imputed)

        save_joblib(self.art.path("preprocessor.joblib"), bundle["preprocessor"])
        save_joblib(self.art.path("label_encoder.joblib"), bundle["label_encoder"])
        save_npz(self.art.path("splits.npz"), **bundle["splits"])
        np.save(self.art.path("feature_names.npy"), np.array(bundle["feature_names"], dtype=object))
        pd.DataFrame({"feature_name": bundle["feature_names"]}).to_csv(
            self.art.path("feature_names.csv"), index=False
        )
        np.savez_compressed(
            self.art.path("labels_encoded.npz"),
            classes_=np.array(bundle["classes"], dtype=object),
        )

        raw = bundle["raw_splits"]
        for split_name in ("tr", "int", "ext"):
            split_df = raw[f"X_{split_name}"].copy()
            split_df[self.data_cfg.output_col] = raw[f"y_{split_name}"]
            save_csv(self.art.path(f"raw_{split_name}.csv"), split_df)

        report = {
            "file_name": self.data_cfg.file_name,
            "sheet": self.data_cfg.sheet,
            "output_col": self.data_cfg.output_col,
            "input_cols": list(self.data_cfg.input_cols),
            "rows_after_label_cleaning": int(len(X_df)),
            "numeric_cols": list(bundle["num_cols"]),
            "categorical_cols": list(bundle["cat_cols"]),
            "classes": bundle["classes"],
            "split_shapes": {
                name: list(value.shape) if hasattr(value, "shape") else [len(value)]
                for name, value in bundle["splits"].items()
            },
            "outputs": {
                "preprocessor": "preprocessor.joblib",
                "label_encoder": "label_encoder.joblib",
                "splits": "splits.npz",
                "feature_names": "feature_names.csv",
                "labels": "labels_encoded.npz",
            },
        }
        save_json(self.art.path("preprocess_report.json"), report)

    def _impute_for_review(self, preprocessor, X_df: pd.DataFrame) -> pd.DataFrame:
        fitted = clone(preprocessor).fit(X_df.copy())
        rows = {}

        if "num" in fitted.named_transformers_:
            num_cols = list(fitted.transformers_[0][2])
            if num_cols:
                num_pipe = fitted.named_transformers_["num"]
                rows.update(
                    pd.DataFrame(
                        num_pipe.named_steps["imputer"].transform(X_df[num_cols]),
                        columns=num_cols,
                        index=X_df.index,
                    ).to_dict("series")
                )

        if "cat" in fitted.named_transformers_:
            cat_cols = list(fitted.transformers_[1][2])
            if cat_cols:
                cat_pipe = fitted.named_transformers_["cat"]
                rows.update(
                    pd.DataFrame(
                        cat_pipe.named_steps["imputer"].transform(X_df[cat_cols]),
                        columns=cat_cols,
                        index=X_df.index,
                    ).to_dict("series")
                )

        return pd.DataFrame(rows, index=X_df.index)[list(X_df.columns)]

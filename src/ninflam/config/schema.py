from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataCfg:
    file_name: str
    sheet: int
    output_col: str
    input_cols: List[str]
    valid_labels: List[str]
    label_missing_strategy: str = "drop"

@dataclass(frozen=True)
class SplitCfg:
    random_state: int
    external_test_size: float
    internal_test_size: float
    min_class_count_for_stratify: int = 5

@dataclass(frozen=True)
class PreprocessingCfg:
    scaler: str
    onehot_dense: bool
    save_feature_names: bool = True

@dataclass(frozen=True)
class IOCfg:
    outdir_tag: str
    outputs_dirname: str
    save_cleaned_csv: bool = True
    save_imputed_full_csv: bool = True
    save_in_out_clean_csv: bool = True
    save_in_out_imputed_csv: bool = True

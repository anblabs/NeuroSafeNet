import os
FILE_PATH = os.path.join("../Data", file_name)
SHEET     = 0
OUTDIR    = f"../Data/preprocessed_{preprocessed_folder_tag}"

base = file_name.replace(".xlsx", "").replace(".csv", "")

PREP_PATH   = os.path.join(OUTDIR, f"preprocessor_{base}.joblib")
SPLITS_NPZ  = os.path.join(OUTDIR, f"splits_preprocessed_{base}.npz")
LABELS_NPZ  = os.path.join(OUTDIR, f"labels_encoded_{base}.npz")
REPORT_JSON = os.path.join(OUTDIR, f"preprocess_report_{base}.json")
CLEAN_CSV   = os.path.join(OUTDIR, f"data_cleaned_{base}.csv")
IN_OUT_CLEAN_CSV = os.path.join(OUTDIR, f"data_cleaned_IN_OUT_{base}.csv")
IN_OUT_IMPUTED_CSV = os.path.join(OUTDIR, f"data_imputed_IN_OUT_{base}.csv")
IMPUTED_CSV = os.path.join(OUTDIR, f"data_imputed_{base}.csv")

# NEW — save expanded feature names
FEATURE_NAMES_NPY = os.path.join(OUTDIR, f"feature_names_{base}.npy")
FEATURE_NAMES_CSV = os.path.join(OUTDIR, f"feature_names_{base}.csv")


from Neuro_Inflammation.Data.filename import preprocessed_folder_tag, file_name
import os

BASE = file_name
PREPROC_DIR = f"../Data/preprocessed_{preprocessed_folder_tag}"
OUTDIR = f"outputs_{preprocessed_folder_tag}"

MODELDIR = os.path.join(OUTDIR, "models")
DATADIR = os.path.join(OUTDIR, "predictions")

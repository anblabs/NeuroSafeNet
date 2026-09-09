from pathlib import Path
from Neuro_Inflammation.Data.filename import preprocessed_folder_tag, file_name

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"

PREPROC_DIR = DATA_DIR / f"preprocessed_{preprocessed_folder_tag}"
OUTDIR = ROOT_DIR / f"outputs_{preprocessed_folder_tag}"

MODELDIR = OUTDIR / "models"
DATADIR = OUTDIR / "predictions"
METADIR = OUTDIR / "metadata"

SPLITS_NPZ = PREPROC_DIR / f"splits_preprocessed_{file_name}.npz"
FEATURE_NAMES_NPY = PREPROC_DIR / f"feature_names_{file_name}.npy"
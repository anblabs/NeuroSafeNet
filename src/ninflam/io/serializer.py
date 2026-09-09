import json
import numpy as np
from joblib import dump

def save_joblib(path, obj) -> None:
    dump(obj, path)

def save_npz(path, **arrays) -> None:
    np.savez_compressed(path, **arrays)

def save_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_csv(path, df) -> None:
    df.to_csv(path, index=False)

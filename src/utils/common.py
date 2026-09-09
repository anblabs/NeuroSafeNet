import numpy as np
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def filter_rare_classes(X, y, min_count=5):
    y = np.asarray(y)
    unique_labels, counts = np.unique(y, return_counts=True)

    keep_labels = unique_labels[counts >= min_count]
    mask = np.isin(y, keep_labels)

    X_filtered = X[mask]
    y_filtered = y[mask]

    return X_filtered, y_filtered, keep_labels.tolist()


def remap_labels_to_contiguous(y):
    y = np.asarray(y)
    unique_labels = sorted(np.unique(y).tolist())
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_remapped = np.array([label_map[v] for v in y], dtype=int)
    return y_remapped, unique_labels, label_map
import numpy as np


def load_preprocessed_data(splits_npz_path, feature_names_npy_path):
    data = np.load(splits_npz_path, allow_pickle=True)

    # Use training split from preprocessing output as the base dataset
    X = data["X_tr"]
    y = data["y_tr"]

    feature_names = np.load(feature_names_npy_path, allow_pickle=True).tolist()

    return X, y, feature_names
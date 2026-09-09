import numpy as np


def safe_predict_proba(model, X, n_classes: int):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.ndim == 2 and proba.shape[1] == 1 and n_classes == 2:
            p = proba[:, 0]
            proba = np.vstack([1 - p, p]).T

        if proba.ndim != 2 or proba.shape[1] != n_classes:
            proba = np.full((len(X), n_classes), 1.0 / n_classes)

        return proba

    return np.full((len(X), n_classes), 1.0 / n_classes)


def predict_model(model, X, n_classes: int):
    y_pred = model.predict(X)
    y_proba = safe_predict_proba(model, X, n_classes)
    return y_pred, y_proba
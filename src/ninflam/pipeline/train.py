import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ninflam.splitting.kfold import make_stratified_kfold


log = logging.getLogger(__name__)


def run_training_pipeline(cfg: DictConfig) -> None:
    paths = _resolve_paths(cfg)
    _ensure_training_dirs(paths)

    splits = np.load(paths["splits"], allow_pickle=True)
    feature_names = np.load(paths["feature_names"], allow_pickle=True).tolist()
    classes = np.load(paths["labels"], allow_pickle=True)["classes_"].tolist()

    X_tr = splits["X_tr"]
    y_tr = splits["y_tr"]
    X_int = splits["X_int"]
    y_int = splits["y_int"]
    X_ext = splits["X_ext"]
    y_ext = splits["y_ext"]

    models = get_models(
        n_classes=len(classes),
        random_state=int(cfg.split.random_state),
        selected_model_names=list(cfg.training.models),
    )

    run_summary = {
        "classes": [str(c) for c in classes],
        "feature_count": len(feature_names),
        "models_requested": list(cfg.training.models),
        "use_kfold": bool(cfg.training.get("use_kfold", False)),
        "n_splits": int(cfg.training.get("n_splits", 0)),
        "models_trained": [],
        "models_failed": {},
    }

    for model_name, model in models.items():
        log.info("Training %s", model_name)
        try:
            if bool(cfg.training.get("use_kfold", False)):
                cv_result = _cross_validate_model(
                    model,
                    X_tr,
                    y_tr,
                    classes,
                    n_splits=int(cfg.training.get("n_splits", 5)),
                    random_state=int(cfg.split.random_state),
                )
                _save_cv_outputs(paths, model_name, cv_result)

            model.fit(X_tr, y_tr)
            dump(model, paths["models_dir"] / f"{model_name}.joblib")

            int_result = _evaluate_split(model, X_int, y_int, classes, "internal")
            ext_result = _evaluate_split(model, X_ext, y_ext, classes, "external")

            _save_model_outputs(paths, model_name, int_result, ext_result, classes)
            run_summary["models_trained"].append(model_name)
        except Exception as exc:
            log.exception("Model failed: %s", model_name)
            run_summary["models_failed"][model_name] = str(exc)

    metadata = {
        "preprocessed_dir": str(paths["preprocessed_dir"]),
        "training_dir": str(paths["training_dir"]),
        "feature_names": feature_names,
        **run_summary,
    }
    _write_json(paths["training_dir"] / "run_metadata.json", metadata)
    log.info("Training complete. Outputs in: %s", paths["training_dir"])


def _resolve_paths(cfg: DictConfig) -> dict:
    base = Path(str(cfg.data.file_name)).stem
    preprocessed_dir = (
        Path(str(cfg.io.outputs_dirname))
        / f"preprocessed_{cfg.io.outdir_tag}"
        / base
    )
    training_dir = (
        Path(str(cfg.io.outputs_dirname))
        / f"training_{cfg.io.outdir_tag}"
        / base
    )
    return {
        "preprocessed_dir": preprocessed_dir,
        "training_dir": training_dir,
        "models_dir": training_dir / "models",
        "metrics_dir": training_dir / "metrics",
        "predictions_dir": training_dir / "predictions",
        "confusion_dir": training_dir / "confusion_matrices",
        "cv_dir": training_dir / "cross_validation",
        "splits": preprocessed_dir / "splits.npz",
        "feature_names": preprocessed_dir / "feature_names.npy",
        "labels": preprocessed_dir / "labels_encoded.npz",
    }


def _ensure_training_dirs(paths: dict) -> None:
    required = [paths["splits"], paths["feature_names"], paths["labels"]]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Training requires preprocessing outputs first. Missing files: "
            + ", ".join(missing)
        )

    for key in ("training_dir", "models_dir", "metrics_dir", "predictions_dir", "confusion_dir", "cv_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def get_models(n_classes: int, random_state: int, selected_model_names=None) -> dict:
    is_binary = n_classes == 2
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            random_state=random_state,
        ),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if is_binary else "multi:softprob",
            num_class=None if is_binary else n_classes,
            eval_metric="logloss" if is_binary else "mlogloss",
            tree_method="hist",
            random_state=random_state,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            loss_function="Logloss" if is_binary else "MultiClass",
            verbose=False,
            random_state=random_state,
        )
    except Exception:
        pass

    if selected_model_names is None:
        return models

    return {name: model for name, model in models.items() if name in selected_model_names}


def _evaluate_split(model, X, y_true, classes, split_name: str) -> dict:
    y_pred = model.predict(X)
    y_proba = _safe_predict_proba(model, X, len(classes))

    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=[str(c) for c in classes],
            zero_division=0,
            output_dict=True,
        ),
    }

    return {
        "split": split_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
        ),
    }


def _cross_validate_model(model, X, y, classes, n_splits: int, random_state: int) -> dict:
    y = np.asarray(y)
    min_class_count = int(np.min(np.bincount(y))) if len(y) else 0
    effective_splits = min(n_splits, min_class_count)

    if effective_splits < 2:
        return {
            "requested_splits": n_splits,
            "effective_splits": effective_splits,
            "skipped": True,
            "reason": "Not enough samples per class for stratified k-fold.",
            "folds": [],
            "summary": {},
        }

    cv = make_stratified_kfold(effective_splits, random_state)
    fold_rows = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        result = _evaluate_split(
            fold_model,
            X[valid_idx],
            y[valid_idx],
            classes,
            split_name=f"fold_{fold_idx}",
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "accuracy": result["metrics"]["accuracy"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "precision_macro": result["metrics"]["precision_macro"],
                "recall_macro": result["metrics"]["recall_macro"],
                "f1_macro": result["metrics"]["f1_macro"],
            }
        )

    summary = {}
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    for metric in metric_names:
        values = np.array([row[metric] for row in fold_rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return {
        "requested_splits": n_splits,
        "effective_splits": effective_splits,
        "skipped": False,
        "folds": fold_rows,
        "summary": summary,
    }


def _safe_predict_proba(model, X, n_classes: int) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.full((len(X), n_classes), 1.0 / n_classes)

    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == n_classes:
        return proba

    if proba.ndim == 2 and proba.shape[1] == 1 and n_classes == 2:
        p = proba[:, 0]
        return np.vstack([1 - p, p]).T

    return np.full((len(X), n_classes), 1.0 / n_classes)


def _save_model_outputs(paths, model_name, int_result, ext_result, classes) -> None:
    metrics = {
        "internal": int_result["metrics"],
        "external": ext_result["metrics"],
    }
    _write_json(paths["metrics_dir"] / f"{model_name}.json", metrics)

    for result in (int_result, ext_result):
        split = result["split"]
        pred_df = pd.DataFrame(
            {
                "y_true": result["y_true"],
                "y_pred": result["y_pred"],
                "y_true_label": [classes[int(i)] for i in result["y_true"]],
                "y_pred_label": [classes[int(i)] for i in result["y_pred"]],
            }
        )
        for idx, class_name in enumerate(classes):
            pred_df[f"proba_{class_name}"] = result["y_proba"][:, idx]
        pred_df.to_csv(paths["predictions_dir"] / f"{model_name}_{split}.csv", index=False)

        cm_df = pd.DataFrame(
            result["confusion_matrix"],
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        cm_df.to_csv(paths["confusion_dir"] / f"{model_name}_{split}.csv")


def _save_cv_outputs(paths, model_name, cv_result) -> None:
    _write_json(paths["cv_dir"] / f"{model_name}_cv.json", cv_result)

    if cv_result.get("folds"):
        pd.DataFrame(cv_result["folds"]).to_csv(
            paths["cv_dir"] / f"{model_name}_fold_metrics.csv",
            index=False,
        )

    if cv_result.get("summary"):
        pd.DataFrame([cv_result["summary"]]).to_csv(
            paths["cv_dir"] / f"{model_name}_cv_summary.csv",
            index=False,
        )


def _write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

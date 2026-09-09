from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def get_models(n_classes: int, random_state: int, selected_model_names=None):
    is_binary = n_classes == 2
    models = {}

    # Always available sklearn models
    models["RF"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=10,
        random_state=random_state,
    )

    models["NaiveBayes"] = GaussianNB()

    models["LogisticRegression"] = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )

    models["SVM"] = SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )

    models["KNN"] = KNeighborsClassifier(n_neighbors=5)

    models["LDA"] = LinearDiscriminantAnalysis()

    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )

    # Optional: XGBoost
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

    # Optional: CatBoost
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

    filtered = {k: v for k, v in models.items() if k in selected_model_names}
    return filtered
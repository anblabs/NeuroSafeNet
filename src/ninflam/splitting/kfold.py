from sklearn.model_selection import StratifiedKFold


def make_stratified_kfold(n_splits, random_state):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
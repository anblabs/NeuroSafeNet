from sklearn.model_selection import train_test_split


def make_holdout_split(X, y, test_ratio, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )
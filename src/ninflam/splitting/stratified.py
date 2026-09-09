import pandas as pd
from sklearn.model_selection import train_test_split

def safe_stratify_decision(y: pd.Series, min_class_count: int = 5) -> bool:
    y = pd.Series(y).dropna()
    if y.empty:
        return False

    counts = y.value_counts()
    if counts.empty:
        return False

    return counts.min() >= min_class_count

def split_train_internal_external(X, y, external_size, internal_size, random_state, min_class_count_for_stratify: int = 5):
    use_stratify = safe_stratify_decision(pd.Series(y), min_class_count=min_class_count_for_stratify)

    X_rest, X_ext, y_rest, y_ext = train_test_split(
        X, y,
        test_size=external_size,
        stratify=y if use_stratify else None,
        random_state=random_state
    )

    rel_internal = internal_size / (1 - external_size)

    # If stratify breaks due to tiny counts after first split, disable for second split
    if use_stratify:
        try_counts = pd.Series(y_rest).value_counts()
        if not try_counts.empty and try_counts.min() < 2:
            use_stratify = False

    X_tr, X_int, y_tr, y_int = train_test_split(
        X_rest, y_rest,
        test_size=rel_internal,
        stratify=y_rest if use_stratify else None,
        random_state=random_state
    )

    return X_tr, X_int, X_ext, y_tr, y_int, y_ext

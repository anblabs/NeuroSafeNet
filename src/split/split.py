def split_train_internal_external(X, y):
    class_counts = y.value_counts()
    use_stratify = class_counts.min() >= 5

    X_rest, X_ext, y_rest, y_ext = train_test_split(
        X, y, test_size=TEST_EXTERNAL_SIZE,
        stratify=y if use_stratify else None,
        random_state=RANDOM_STATE
    )

    rel_internal = TEST_INTERNAL_SIZE / (1 - TEST_EXTERNAL_SIZE)

    if use_stratify and y_rest.value_counts().min() < 2:
        use_stratify = False

    X_tr, X_int, y_tr, y_int = train_test_split(
        X_rest, y_rest, test_size=rel_internal,
        stratify=y_rest if use_stratify else None,
        random_state=RANDOM_STATE
    )
    return X_tr, X_int, X_ext, y_tr, y_int, y_ext


def _split(self, X_df, y_series):
    return split_train_internal_external(
        X_df, y_series,
        external_size=self.split_cfg.external_test_size,
        internal_size=self.split_cfg.internal_test_size,
        random_state=self.split_cfg.random_state,
        min_class_count_for_stratify=self.split_cfg.min_class_count_for_stratify,
    )
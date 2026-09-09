def fit_and_transform_splits(pre, X_tr_df, X_int_df, X_ext_df):
    pre.fit(X_tr_df)
    return pre.transform(X_tr_df), pre.transform(X_int_df), pre.transform(X_ext_df)
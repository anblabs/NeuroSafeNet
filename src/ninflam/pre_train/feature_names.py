def get_expanded_feature_names(pre, num_cols, cat_cols):
    if hasattr(pre, "get_feature_names_out"):
        try:
            return pre.get_feature_names_out().tolist()
        except Exception:
            pass

    names = list(num_cols)
    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
    for col, cats in zip(cat_cols, ohe.categories_):
        for v in cats:
            names.append(f"{col}={v}")
    return names

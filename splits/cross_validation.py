from sklearn.model_selection import StratifiedKFold

def stratified_kfold_split(X, y, n_splits=5, random_state=42):
    """Generates Stratified K-Fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(X, y):
        yield X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
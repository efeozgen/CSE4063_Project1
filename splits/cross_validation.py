from sklearn.model_selection import StratifiedKFold

class CrossValidation:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y):
        """Generates Stratified K-Fold splits."""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_index, test_index in skf.split(X, y):
            yield X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

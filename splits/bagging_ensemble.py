import numpy as np
from sklearn.model_selection import train_test_split

class BaggingEnsemble:
    def __init__(self, n_samples, test_size, random_state=None):
        self.n_samples = n_samples
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """Performs bootstrap aggregation split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        n_samples = self.n_samples or len(X_train)
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X_train), size=n_samples, replace=True)
        X_bootstrap = X_train.iloc[indices]
        y_bootstrap = y_train.iloc[indices]
        return [(X_bootstrap, X_test, y_bootstrap, y_test)]
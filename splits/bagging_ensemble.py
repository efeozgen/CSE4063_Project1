import numpy as np

class BaggingEnsemble:
    def __init__(self, n_samples=None, random_state=None):
        self.n_samples = n_samples
        self.random_state = random_state

    def split(self, X, y):
        """Performs bootstrap aggregation split."""
        n_samples = self.n_samples or len(X)
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=n_samples, replace=True)
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y.iloc[indices]
        # Return the entire dataset as test set for simplicity
        yield X_bootstrap, X, y_bootstrap, y

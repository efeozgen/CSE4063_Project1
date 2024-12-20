import numpy as np
from sklearn.model_selection import train_test_split

def bootstrap_aggregation_split(X, y, n_samples=None, random_state=None):
    """Performs bootstrap aggregation split."""
    n_samples = n_samples or len(X)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=n_samples, replace=True)
    X_bootstrap = X.iloc[indices]
    y_bootstrap = y.iloc[indices]
    return X_bootstrap, y_bootstrap
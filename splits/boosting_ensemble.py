from sklearn.model_selection import train_test_split

class BoostingEnsemble:
    def __init__(self, step, test_size, random_state=None):
        self.step = step
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """Splits data incrementally for boosting-like strategies."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        num_instances = len(X_train)
        current_index = 0
        while current_index < num_instances:
            next_index = min(num_instances, current_index + int(self.step * num_instances))
            yield X_train.iloc[:next_index], X_test, y_train.iloc[:next_index], y_test
            current_index = next_index
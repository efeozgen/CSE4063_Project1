class BoostingEnsemble:
    def __init__(self, step=0.1):
        self.step = step

    def split(self, X, y):
        """Splits data incrementally for boosting-like strategies."""
        num_instances = len(X)
        current_index = 0
        while current_index < num_instances:
            next_index = min(num_instances, current_index + int(self.step * num_instances))
            yield X.iloc[:next_index], y.iloc[:next_index]
            current_index = next_index

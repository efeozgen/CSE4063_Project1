class BoostingEnsemble:
    def __init__(self, step=0.1):
        self.step = step

    def split(self, X, y):
        """Splits data incrementally for boosting-like strategies."""
        num_instances = len(X)
        current_index = 0
        while current_index < num_instances:
            next_index = min(num_instances, current_index + int(self.step * num_instances))
            if next_index == num_instances:
                break
            print(f"Yielding train set size: {next_index}, test set size: {num_instances - next_index}")
            yield X.iloc[:next_index], X.iloc[next_index:], y.iloc[:next_index], y.iloc[next_index:]
            current_index = next_index
        # Yield the final split with the remaining data as the training set
        if current_index < num_instances:
            print(f"Yielding final train set size: {current_index}, test set size: {num_instances - current_index}")
            yield X.iloc[:current_index], X.iloc[current_index:], y.iloc[:current_index], y.iloc[current_index:]

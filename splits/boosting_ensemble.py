from sklearn.model_selection import train_test_split

def incremental_split(X, y, step=0.1):
    """Splits data incrementally for boosting-like strategies."""
    num_instances = len(X)
    current_index = 0
    while current_index < num_instances:
        next_index = min(num_instances, current_index + int(step * num_instances))
        yield X.iloc[:next_index], y.iloc[:next_index]
        current_index = next_index

from sklearn.tree import DecisionTreeClassifier

class GiniClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion='gini', random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy

from sklearn.neural_network import MLPClassifier

class ANNTwoHiddenLayers:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy

from sklearn.naive_bayes import GaussianNB

class NaiveBayesClassifier:
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy

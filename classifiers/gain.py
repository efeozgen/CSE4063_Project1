def decision_tree_gain_ratio(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin
    import numpy as np

    class DecisionTreeWithGainRatio(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.model = DecisionTreeClassifier(criterion='entropy', random_state=42)

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

    model = DecisionTreeWithGainRatio()
    model.fit(X_train, y_train)
    return model
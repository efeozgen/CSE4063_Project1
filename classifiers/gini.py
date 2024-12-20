def decision_tree_gini(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    return model
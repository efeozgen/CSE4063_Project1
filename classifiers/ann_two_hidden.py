def ann_two_hidden_layers(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model
def support_vector_machine_classifier(X_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    return model
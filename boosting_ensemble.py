import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class BoostingEnsemble:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.weights = []

    def fit(self, X, y):

        w = np.ones(len(y)) / len(y)  # Ağırlıklar

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=3)  # Zayıf öğrenici derinliğini artır
            model.fit(X, y, sample_weight=w)

            predictions = model.predict(X)
            incorrect = (predictions != y)  # Yanlış sınıflandırılan örnekler

            error = np.sum(w * incorrect) / np.sum(w)  # Hata oranı
            if error == 0:
                alpha = self.learning_rate * 1  # Eğer hata sıfırsa, katkıyı artır
            else:
                alpha = self.learning_rate * np.log((1 - error) / (error + 1e-10))  # Alpha değeri

            self.models.append(model)
            self.weights.append(alpha)

            w = w * np.exp(-alpha * (2 * incorrect - 1))  # Yanlış sınıflara ağırlık artışı
            w = w / np.sum(w)  # Ağırlıkları normalize et

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(np.unique(X))))  # Her sınıf için tahminler

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            for i, class_prediction in enumerate(pred):
                predictions[i, class_prediction] += weight  # Ağırlıklı tahmin

        return np.argmax(predictions, axis=1)  # En yüksek skorlu sınıf

    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        
        # Tahminleri ve gerçek etiketleri yazdırma
        for true_label, predicted_label in zip(y, predictions):
            print(f"Gerçek: {true_label}, Tahmin: {predicted_label}")
        
        # Doğruluk hesaplama
        accuracy = np.mean(predictions == y)
        return accuracy


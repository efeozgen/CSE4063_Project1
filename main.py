from splits.holdout import HoldoutSplit
from splits.cross_validation import CrossValidation
from splits.bagging_ensemble import BaggingEnsemble
from splits.boosting_ensemble import BoostingEnsemble

from classifiers.gain import GainClassifier
from classifiers.gini import GiniClassifier
from classifiers.naive_bayes import NaiveBayesClassifier
from classifiers.ann_one_hidden import ANNOneHiddenLayer
from classifiers.ann_two_hidden import ANNTwoHiddenLayers
from classifiers.svm import SVMClassifier
from preprocess import preprocess_data, write_clean_data, csv_to_json

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_all_models_with_splits(df):
    X = df.drop(columns=["popularity_label"])
    y = df["popularity_label"]

    splits = [
        ("Holdout", HoldoutSplit(test_size=0.2)),
        ("Cross Validation", CrossValidation(n_splits=5)),
        ("Bagging", BaggingEnsemble(n_samples=len(X), test_size=0.2)),
        ("Boosting", BoostingEnsemble(step=0.2, test_size=0.2)),
    ]

    models = [
        ("Gain Ratio", GainClassifier()),
        ("Gini Index", GiniClassifier()),
        ("Naive Bayes", NaiveBayesClassifier()),
        ("ANN 1 Hidden", ANNOneHiddenLayer()),
        ("ANN 2 Hidden", ANNTwoHiddenLayers()),
        ("SVM", SVMClassifier()),
    ]

    results = []

    for split_name, split_method in splits:
        for model_name, model in models:
            print(f"Running {model_name} with {split_name}...")
            accuracies = []
            for train_X, test_X, train_y, test_y in split_method.split(X, y):
                model.fit(train_X, train_y)
                predictions = model.predict(test_X)
                accuracy = accuracy_score(test_y, predictions)
                accuracies.append(accuracy)
            
            # Ortalama doğruluğu hesapla ve yazdır
            mean_accuracy = np.mean(accuracies)
            # print(f"Model: {model_name}, Split: {split_name}, Mean Accuracy: {mean_accuracy:.4f}")

            results.append((model_name, split_name, mean_accuracy))
            print(f"Done {model_name} with {split_name}...")

    # Accuracy'ye göre sıralama ve yazdırma
    results.sort(key=lambda x: x[2], reverse=True)
    print("\nEvaluation of all models and splits (sorted by accuracy):")
    for result in results:
        model_name, split_name, mean_accuracy = result
        print(f"Model: {model_name}, Split: {split_name}, Mean Accuracy: {mean_accuracy:.4f}")

    # DataFrame oluşturma
    df_results = pd.DataFrame(results, columns=["Model", "Split Method", "Mean Accuracy"])

    # results klasörünü oluşturma
    os.makedirs("results", exist_ok=True)

    # DataFrame'i JSON dosyasına aktarma
    df_results.to_json("results/results.json", orient="records", lines=True)

    # DataFrame'i CSV dosyasına aktarma
    df_results.to_csv("results/results.csv", index=False)

    # Grid tablo şeklinde görselleştirme
    pivot_table = df_results.pivot(index="Model", columns="Split Method", values="Mean Accuracy")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Model Accuracy by Split Method")
    plt.show()

if __name__ == "__main__":
    datapath = "data/spotify_songs.csv"
    processed_data_path = "data/clean_data_filtered.csv"
    json_output_path = "data/clean_data_filtered.json"

    # Preprocess the data
    processed_df = preprocess_data(datapath)

    # Save processed data
    write_clean_data(processed_df, processed_data_path)
    csv_to_json(processed_data_path, json_output_path)

    # Run all models with all split methods
    run_all_models_with_splits(processed_df)
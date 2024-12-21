# from splits.holdout import HoldoutSplit
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
from splits.holdout import HoldoutSplit


def run_all_models_with_splits(df):
    X = df.drop(columns=["popularity_label"])
    y = df["popularity_label"]

    splits = [
        ("Holdout", HoldoutSplit(test_size=0.3)),
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

    for split_name, split_method in splits:
        for model_name, model in models:
            print(f"Running {model_name} with {split_name}...")
            for train_X, test_X, train_y, test_y in split_method.split(X, y):
                model.fit(train_X, train_y)
                accuracy = model.score(test_X, test_y)
                print(f"Model: {model_name}, Split: {split_name}, Accuracy: {accuracy}")


if __name__ == "__main__":
    datapath = "data/spotify_songs.csv"
    processed_data_path = "data/clean_data.csv"
    json_output_path = "data/clean_data.json"

    # Preprocess the data
    processed_df = preprocess_data(datapath)

    # Save processed data
    write_clean_data(processed_df, processed_data_path)
    csv_to_json(processed_data_path, json_output_path)

    # Run all models with all split methods
    run_all_models_with_splits(processed_df)

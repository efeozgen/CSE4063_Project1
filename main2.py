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

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_heatmap(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Filter the correlation matrix to include only track_popularity and popularity_label
    corr_filtered = corr[['track_popularity', 'popularity_label']].drop(['track_popularity', 'popularity_label'])

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 9))

    # Draw the heatmap
    sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, annot_kws={"size": 12}, cbar_kws={"shrink": .8})

    # Set the title
    plt.title('Correlation Heatmap with Track Popularity and Popularity Label', fontsize=16)

    # Set the labels
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Features', fontsize=10)

    # Adjust the y-axis labels to align with the rows
    plt.gca().yaxis.set_tick_params(pad=10)

    # Adjust layout to fit the figure on the screen
    plt.tight_layout()

    # Show the plot
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.show()

def print_low_correlation_labels(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Filter the correlations with popularity_label between -0.05 and 0.05 (inclusive)
    low_corr = corr['popularity_label'][(corr['popularity_label'] >= -0.05) & (corr['popularity_label'] <= 0.05)]

    # Print the labels
    print("Labels with correlation between -0.05 and 0.05 with popularity_label:")
    print(low_corr)

def run_all_models_with_splits(df):
    X = df.drop(columns=["popularity_label", "track_popularity"])
    y = df["popularity_label"]

    splits = [
        #("Cross Validation", CrossValidation(n_splits=5)),
        #("Bagging", BaggingEnsemble(n_samples=len(X))),
        # ("Boosting", BoostingEnsemble(step=0.2)),
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

    # Plot the correlation heatmap
    plot_correlation_heatmap(processed_df)

    # Print labels with low correlation with popularity_label
    print_low_correlation_labels(processed_df)

    # Run all models with all split methods
    run_all_models_with_splits(processed_df)

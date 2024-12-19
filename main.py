# main.py
from preprocess import preprocess_data, write_clean_data, csv_to_json
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    datapath = "data/spotify_songs.csv"
    processed_data_path = "data/clean_data.csv"
    json_output_path = "data/clean_data.json"

    # Preprocess the data
    processed_df = preprocess_data(datapath)

    # Train-test split
    X = processed_df.drop(columns=["popularity_label_encoded"])
    y = processed_df["popularity_label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print("Train set dağılımı:")
    print(y_train.value_counts(normalize=True))

    print("Test set dağılımı:")
    print(y_test.value_counts(normalize=True))

    # Save processed data
    write_clean_data(processed_df, processed_data_path)
    csv_to_json(processed_data_path, json_output_path)


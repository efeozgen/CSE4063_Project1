import pandas as pd
import csv
import json
from feature_encoding import FeatureEncoder
from sklearn.model_selection import train_test_split

def read_data(datapath):
    return pd.read_csv(datapath, encoding="utf-8")

def write_clean_data(df):
    return df.to_csv("data/clean_data.csv")

def csv_to_json(csv_file, json_file):
    # CSV dosyasını JSON'a dönüştür
    data = []
    with open(csv_file, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    # JSON dosyasına yaz
    with open(json_file, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)

    print(
        f"CSV dosyasından JSON formatına dönüştürüldü ve {json_file} dosyasına kaydedildi."
    )

def detect_nulls(df):
    return pd.DataFrame(
        {
            "nulls": df.isnull().sum(),
            "Empty Strings": df.map(lambda x: x == None).sum(),
        }
    )

def handle_float(df, columns):
    for column in columns:
        if column in df.columns:
            try:
                df[column] = df[column].apply(lambda x: float(x))
            except ValueError as e:
                print(f"Hata! '{column}' sütunu dönüştürülemedi: {e}")
        else:
            print(f"Hata! '{column}' sütunu DataFrame'de bulunamadı.")
    return df

def drop_columns(df, columns):
    return df.drop(columns=columns)

def handle_duplicates(df):
    return df.drop_duplicates(subset="track_id", keep="first")

if __name__ == "__main__":
    datapath = "data/spotify_songs.csv"
    df_raw = read_data(datapath)

    nulls = detect_nulls(df_raw)

    columns_to_convert = [
        "track_popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    df = handle_float(df_raw, columns_to_convert)

    columns_to_drop = [
        "track_name",
        "track_album_id",
        "track_album_name",
        "playlist_name",
        "playlist_id",
    ]
    df = drop_columns(df, columns_to_drop)

    # Popularity labeling
    bins_popularity = [0, 40, 70, 100]
    labels_popularity = ["Low", "Medium", "High"]
    df["popularity_label"] = pd.cut(
        df["track_popularity"].astype(int),
        bins=bins_popularity,
        labels=labels_popularity,
    )

    # Release date labeling
    bins_date = [
        df["track_album_release_date"].min(),
        "1970-01-01",
        "1980-01-01",
        "1990-01-01",
        "2000-01-01",
        "2010-01-01",
        "2015-01-01",
        "2020-01-01",
        df["track_album_release_date"].max(),
    ]
    labels_release_date = [
        "<1970",
        "1970-1980",
        "1980-1990",
        "1990-2000",
        "2000-2010",
        "2010-2015",
        "2015-2020",
        ">2020",
    ]
    df["release_date_label"] = pd.cut(
        df["track_album_release_date"].astype(str),
        bins=bins_date,
        labels=labels_release_date,
    )

    print(
        "number of duplicate track_id's before handle duplicate =>",
        df["track_id"].duplicated().sum(),
    )
    df = handle_duplicates(df)
    print(
        "number of duplicate track_id's after handle duplicate =>",
        df["track_id"].duplicated().sum(),
    )

    df = drop_columns(df, ["track_id"])

    encoder = FeatureEncoder(df)

    df = encoder.frequency_encode("track_artist")
    df = encoder.label_encode("release_date_label")
    df = encoder.one_hot_encode("playlist_genre", "genre")
    df = encoder.one_hot_encode("playlist_subgenre", "subgenre")
    df = encoder.label_encode("popularity_label")

    columns_to_drop = [
        "track_artist",
        "track_album_release_date",
        "release_date_label",
        "popularity_label",
        "track_popularity",
        "playlist_genre",
        "playlist_subgenre",
    ]
    df = encoder.drop_original_columns(columns_to_drop)

    processed_df = encoder.get_dataframe()

    # Stratified Train-Test Split
    X = processed_df.drop(columns=["popularity_label_encoded"])
    y = processed_df["popularity_label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("Train set dağılımı:")
    print(y_train.value_counts(normalize=True))

    print("Test set dağılımı:")
    print(y_test.value_counts(normalize=True))

    write_clean_data(df)
    csv_to_json("data/clean_data.csv", "data/clean_data.json")

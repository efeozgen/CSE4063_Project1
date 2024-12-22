# preprocess.py
import pandas as pd
import csv
import json
from feature_encoding import FeatureEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# Reading the dataset
def read_data(datapath):
    return pd.read_csv(datapath, encoding="utf-8")


# Writing the cleaned dataset
def write_clean_data(df, output_path):
    df.to_csv(output_path, index=False)


# Converting CSV to JSON
def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    with open(json_file, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)
    print(
        f"CSV dosyasından JSON formatına dönüştürüldü ve {json_file} dosyasına kaydedildi."
    )


# Detecting null values
def detect_nulls(df):
    return pd.DataFrame(
        {
            "nulls": df.isnull().sum(),
            "Empty Strings": df.apply(lambda x: x.eq(None).sum()),
        }
    )


# Handling float conversions
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


# Dropping unnecessary columns
def drop_columns(df, columns):
    return df.drop(columns=columns, errors="ignore")


# Handling duplicate entries
def handle_duplicates(df):
    return df.drop_duplicates(subset="track_id", keep="first")


# Preprocessing pipeline
def preprocess_data(datapath):
    df_raw = read_data(datapath)

    # # Detect nulls
    # nulls = detect_nulls(df_raw)
    # print("Null değerlerin özeti:")
    # print(nulls)

    # Columns to convert to float
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

    # Columns to drop
    columns_to_drop = [
        "track_name",
        "track_album_id",
        "track_album_name",
        "playlist_name",
        "playlist_id",
    ]
    df = drop_columns(df, columns_to_drop)

    # Popularity labeling
    bins_popularity = [0, 33, 67, 100]
    labels_popularity = [1, 2, 3]
    df["popularity_label"] = pd.cut(
        df["track_popularity"].astype(int),
        bins=bins_popularity,
        labels=labels_popularity,
        right=False,  # Aralıkların [start, end) olması için
    )

    # Eğer track_popularity 0 ise popularity_label'ı 1 olarak ayarla
    df.loc[df["track_popularity"] == 0, "popularity_label"] = 1

    # Eğer track_popularity 100 ise popularity_label'ı 3 olarak ayarla
    df.loc[df["track_popularity"] == 100, "popularity_label"] = 3

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
        f"number of duplicate track_id's before handle duplicate => {df['track_id'].duplicated().sum()}"
    )
    df = handle_duplicates(df)
    print(
        f"number of duplicate track_id's after handle duplicate => {df['track_id'].duplicated().sum()}"
    )

    df = drop_columns(df, ["track_id"])

    encoder = FeatureEncoder(df)

    df = encoder.frequency_encode("track_artist")
    df = encoder.label_encode("release_date_label")
    df = encoder.one_hot_encode("playlist_genre", "genre")
    df = encoder.one_hot_encode("playlist_subgenre", "subgenre")
    # df = encoder.label_encode("popularity_label")
    

    columns_to_drop = [
        "track_artist",
        "track_album_release_date",
        "release_date_label",
        "playlist_genre",
        # "track_popularity",
        "playlist_subgenre",
    ]
    
    df = encoder.drop_original_columns(columns_to_drop)
    
    # Korelasyon matrisini hesaplayın
    correlation_matrix = df.corr()

    # track_popularity ve popularity_label ile olan korelasyonları seçin
    correlation_with_target = correlation_matrix[["track_popularity", "popularity_label"]]

    # Korelasyon matrisini görselleştirin
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", vmin=-0.15, vmax=0.15)
    plt.title("Correlation with track_popularity and popularity_label")
    # plt.show()
    
    # Korelasyonu düşük olan özellikleri belirleyin
    threshold = 0.05  # Korelasyon eşiği
    low_correlation_features = correlation_with_target[
        (correlation_with_target["track_popularity"].abs() < threshold) &
        (correlation_with_target["popularity_label"].abs() < threshold)
    ].index.tolist()
    
    low_correlation_features = [feature for feature in low_correlation_features if "subgenre" not in feature and "genre" not in feature and "encoded" not in feature]

    # Korelasyonu düşük olan özellikleri veri setinden düşürün
    df = df.drop(columns=low_correlation_features)
    df = df.drop(columns=["track_popularity"])
    
    drops = pd.DataFrame(
        {
            "Dropped Features": low_correlation_features,
        }
    )
    
    print("Dropped low correlation features:\n", drops)
    
    write_clean_data(df, "data/clean_data_filtered.csv")
    csv_to_json("data/clean_data_filtered.csv", "data/clean_data_filtered.json")
    
    # # Detect nulls
    # nulls = detect_nulls(df)
    # print("Null değerlerin özeti:")
    # print(nulls)

    return df
    # return encoder.get_dataframe()

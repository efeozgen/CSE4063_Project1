import pandas as pd
from detect_nulls import detect_nulls


def read_raw_data(row_data_path):
    return pd.read_json(row_data_path)


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


if __name__ == "__main__":
    row_data_path = "data/raw_data.json"
    df_raw = read_raw_data(row_data_path)

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
    # print(df[columns_to_convert])

    columns_to_drop = [
        "track_id",
        "track_name",
        "track_album_id",
        "track_album_name",
        "playlist_name",
        "playlist_id",
    ]
    df_clean = drop_columns(df, columns_to_drop)
    print(df_clean.head())

    # Popularity classification
    bins = [0, 40, 70, 100]
    labels = ["Low", "Medium", "High"]
    df["popularity_class"] = pd.cut(df["track_popularity"].astype(int), bins=bins, labels=labels)
    
    print(df["popularity_class"].value_counts())

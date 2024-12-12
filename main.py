import pandas as pd


def read_raw_data(row_data_path):
    return pd.read_json(row_data_path)


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

    # Release date
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
    labels_popularity = ["<1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "2010-2015", "2015-2020", ">2020"]
    df["release_date_label"] = pd.cut(
        df["track_album_release_date"].astype(str),
        bins=bins_date,
        labels=labels_popularity,
    )

    # print(df["popularity_class"].value_counts())
    print("number of duplicate track_id's before handle duplicate =>", df["track_id"].duplicated().sum())
    df = handle_duplicates(df)
    print("number of duplicate track_id's after handle duplicate =>", df["track_id"].duplicated().sum())
    print(df["track_id"].duplicated() == True)

#feature_encoding.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FeatureEncoder:
    def __init__(self, df):
        self.df = df
        self.label_encoders = {}

    def frequency_encode(self, column_name):
        """Apply Frequency Encoding to a column."""
        frequency = self.df[column_name].value_counts()
        self.df[f"{column_name}_encoded"] = self.df[column_name].map(frequency)
        return self.df

    def label_encode(self, column_name):
        """Apply Label Encoding to a column."""
        label_encoder = LabelEncoder()
        self.df[f"{column_name}_encoded"] = label_encoder.fit_transform(self.df[column_name])
        self.label_encoders[column_name] = label_encoder  # Save encoder for inverse_transform if needed
        return self.df

    def one_hot_encode(self, column_name, prefix):
        """Apply One-Hot Encoding to a column."""
        one_hot = pd.get_dummies(self.df[column_name], prefix=prefix)
        self.df = pd.concat([self.df, one_hot], axis=1)
        return self.df

    def drop_original_columns(self, columns):
        """Drop original categorical columns from the dataset."""
        self.df = self.df.drop(columns, axis=1)
        return self.df

    def get_dataframe(self):
        """Return the processed dataframe."""
        return self.df

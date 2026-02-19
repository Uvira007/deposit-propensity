"""
Preprocess bank marketing data: encode categoricals, standardize numericals, split and return transformer
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
        df: pd.DataFrame,
        target_column: str,
        positive_class: str,
        categorical_columns: list[str],
        drop_columns: list[str],
        test_size: float = 0.2,
        random_state: int = 42):
    """
    Encode categoricals, drop columns, split into train/test.
    Returns X_train, X_test, Y_train, Y_test, fitted ColumnTransformer and feature names
    """
    df = df.copy()
    # Encode target column as binary
    y = (df[target_column].astype(str).str.lower() == positive_class.lower()).astype(int)
    # Drop specified columns
    X = df.drop(columns = [target_column] + [c for c in drop_columns if c in df.columns])

    categorical_columns = [col for col in categorical_columns if col in X.columns]
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Define ColumnTransformer
    transformer = ColumnTransformer(
        [("num",
          StandardScaler(),
          numerical_columns),
          ("cat",
           OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
           categorical_columns),
           ],
        remainder = "drop",
    )
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size,
                                                      random_state=random_state, stratify = y
    )
    # transform the data
    X_train = pd.DataFrame(
        transformer.fit_transform(X_train),
        columns = _get_feature_names(transformer, numerical_columns, categorical_columns),
        index = X_train.index,
    )
    X_test = pd.DataFrame(
        transformer.fit_transform(X_test),
        columns = _get_feature_names(transformer, numerical_columns, categorical_columns),
        index = X_test.index,
    )
    return X_train, X_test, Y_train, Y_test, transformer, X_train.columns.tolist()

def _get_feature_names(transformer: ColumnTransformer, 
                       numerical_columns: list[str], 
                       categorical_columns: list[str])-> list[str]:
    """Build feature names after columnTransformer (numeric + one-hot encoded)"""
    names = list[str](numerical_columns)
    cat_enc = transformer.named_transformers_["cat"]
    cats = list[Any](cat_enc.get_feature_names_out(categorical_columns))
    names.extend(cats)
    return names

if __name__ == "__main__":
    dataset = pd.read_csv("data/raw/bank-full.csv", sep=";")
    dataset = dataset[:50]
    X_train, X_test, Y_train, Y_test, transformer, feature_names = preprocess_data(
        dataset,
        target_column="y",
        positive_class="yes",
        categorical_columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
        drop_columns=["duration"],
    )
    print("X_train:", X_train.head(5))
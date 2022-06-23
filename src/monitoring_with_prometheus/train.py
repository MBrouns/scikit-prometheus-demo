from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklego.preprocessing import ColumnSelector
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from skprometheus.metrics import MetricRegistry
from skprometheus.pipeline import Pipeline
from skprometheus.preprocessing import OneHotEncoder
from skprometheus.impute import SimpleImputer


categorical_features = [
    "region",
    "tenure",
    "top_pack",
]

numerical_features = [
    "montant",
    "frequence_rech",
    "revenue",
    "arpu_segment",
    "frequence",
    "data_volume",
    "on_net",
    "orange",
    "tigo",
    "zone1",
    "zone2",
    "regularity",
    "freq_top_pack",
]


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load training data"""
    logger.info(f"Loading data from: {data_path}")
    df_train = pd.read_csv(data_path)

    return (
        df_train
        .rename(dict(zip(df_train.columns, df_train.columns.str.lower())), axis=1)
        .drop(["mrg"], axis=1)
        .pipe(parse_pandas_dtypes)
    )


def parse_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse data types for pandas dataframe of model features"""

    return df.assign(
        region=lambda d: d["region"].astype("category"),
        tenure=lambda d: d["tenure"].astype("category"),
        montant=lambda d: d["montant"].astype("float"),
        frequence_rech=lambda d: d["frequence_rech"].astype("float"),
        revenue=lambda d: d["revenue"].astype("float"),
        arpu_segment=lambda d: d["arpu_segment"].astype("float"),
        frequence=lambda d: d["frequence"].astype("float"),
        data_volume=lambda d: d["data_volume"].astype("float"),
        on_net=lambda d: d["on_net"].astype("float"),
        orange=lambda d: d["orange"].astype("float"),
        tigo=lambda d: d["tigo"].astype("float"),
        zone1=lambda d: d["zone1"].astype("float"),
        zone2=lambda d: d["zone2"].astype("float"),
        regularity=lambda d: d["regularity"].astype("float"),
        top_pack=lambda d: d["top_pack"].astype("category"),
        freq_top_pack=lambda d: d["freq_top_pack"].astype("float"),
    )


def split_X_y(df):
    """Split train or test dataframe in X and y"""
    logger.info("Splitting dataframe in X and y")
    X = df.drop(["churn"], axis=1)
    y = df["churn"]

    return X, y


def save_model(model, timestamp):
    """Save model to a pickle file using joblib"""
    filename = f"models/DecisionTree.pkl"
    logger.info(f"Saving model to file: {filename}")
    joblib.dump(model, filename)


def train_model(data_path, n_rows=100_000):
    """Training job for a model"""

    timestamp = f"{datetime.now():%Y-%m-%dT%H-%M-%S}"
    logger.info(f"Start model training job at timestamp: {timestamp}")

    df_train = load_data(data_path).head(n_rows)
    X, y = split_X_y(df_train)
    MetricRegistry.set_labels(['model_version'])
    with MetricRegistry.label(model_version='DecisionTree'):
        categorical_pipeline = Pipeline([
            ("selector", ColumnSelector(categorical_features)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])

        numerical_pipeline = Pipeline([
            ("selector", ColumnSelector(numerical_features)),
            ("imputation", SimpleImputer(missing_values=pd.NA, strategy="mean"))
        ])

        model = Pipeline([
            ("preprocessing", FeatureUnion([
                ("categorical_pipeline", categorical_pipeline),
                ("numerical_pipeline", numerical_pipeline),
            ])),
            ("model", DecisionTreeClassifier()),
        ])

        logger.info("Start model fitting, this may take a while ...")
        model.fit(X, y)

    save_model(model, timestamp)

    return model


if __name__ == "__main__":
    train_model(Path("data/train.csv"))

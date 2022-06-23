import asyncio
from typing import Union

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import joblib
import numpy as np
import pandas as pd
from prometheus_async.aio import time
from prometheus_client import generate_latest
from pydantic import BaseModel

from monitoring_with_prometheus.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUEST_ERROR, RESPONSE_DIST
from monitoring_with_prometheus.train import parse_pandas_dtypes

from skprometheus.pipeline import Pipeline
from skprometheus.preprocessing import OneHotEncoder
from skprometheus.impute import SimpleImputer

Pipeline([])
OneHotEncoder()
SimpleImputer()

app = FastAPI()

# Load prediction model
MODEL_VERSION = "DecisionTree"
model = joblib.load(f"models/{MODEL_VERSION}.pkl")


@app.get("/")
def root():
    """Root of the prometheus monitoring app"""
    return {"message": "Prometheus monitoring app root"}


@app.get('/metrics', response_class=PlainTextResponse)
def metrics():
    return generate_latest()


class Features(BaseModel):
    user_id: str
    region: Union[str, None]
    tenure: Union[str, None]
    montant: Union[float, None]
    frequence_rech: Union[float, None]
    revenue: Union[float, None]
    arpu_segment: Union[float, None]
    frequence: Union[float, None]
    data_volume: Union[float, None]
    on_net: Union[float, None]
    orange: Union[float, None]
    tigo: Union[float, None]
    zone1: Union[float, None]
    zone2: Union[float, None]
    regularity: Union[float, None]
    top_pack: Union[str, None]
    freq_top_pack: Union[float, None]


@app.post("/predict/model")
@REQUEST_LATENCY.labels(model_version=MODEL_VERSION).time()
def post_model_prediction(features: Features) -> int:
    """Get a prediction from the model that is deployed based on input features"""
    REQUEST_COUNT.labels(model_version=MODEL_VERSION).inc()

    X = pd.DataFrame().from_dict(features.dict(), orient="index").T
    X = parse_pandas_dtypes(X)

    if np.random.uniform() < 0.01:
        REQUEST_ERROR.labels(model_version=MODEL_VERSION).inc()
        raise ValueError("Could not make a prediction")

    prediction = model.predict_proba(X)[:, 1].item()
    RESPONSE_DIST.labels(model_version=MODEL_VERSION).observe(prediction)

    return prediction




import time

from loguru import logger
import requests

from monitoring_with_prometheus.train import load_data, parse_pandas_dtypes, split_X_y


if __name__ == "__main__":
    X, _ = (
        load_data("data/train.csv")
        .loc[100_000:, :]
        .reset_index(drop=True)
        .pipe(parse_pandas_dtypes)
        .pipe(split_X_y)
    )

    logger.info(f"Simulating {len(X)} instances")
    time.sleep(3)

    for i in range(len(X)):
        features = X.iloc[i].to_json()
        r = requests.post("http://localhost:5050/predict/model", data=features)
        if r.status_code == 200:
            logger.info(f"instance: {i}, prediction: {r.json()}")
        else:
            logger.info(f"instance: {i}, failed")

"""
Prediction pipeline
"""
import time
import logging

import pandas as pd
import numpy as np

from src.config.directories import directories as dirs
from src.constants import (c_PREDICTIONS_DATA,
                           c_LEXIQUE,
                           c_PREDICTIONS
                          )
from src.predict.main import predict
from src.io import load_json_file, save_prediction
from src.train import LabelEncoder


logger = logging.getLogger(__name__)


def prediction_transform(prob, label_encoder):
    y_pred = np.argmax(prob, axis=1)
    predictions = [(k,v) for k,v in zip(label_encoder.decode(y_pred), 
                    [pb[y_] for pb,y_ in zip(prob, y_pred)])]
    return predictions


def main():
    start = time.time()
    logger.info("Starting prediction job...")

    label_encoder = LabelEncoder()
    # load data and lexique
    X = load_json_file(dirs.test_dir / c_PREDICTIONS_DATA)
    lexiques = load_json_file(dirs.config / c_LEXIQUE)
    # transform data in DataFrame
    X = pd.DataFrame(X)
    # predict
    prob = predict(X, lexiques, do_probabilities=True)
    predictions = prediction_transform(prob, label_encoder)
    # save predictions
    save_prediction(predictions, path=dirs.raw_store_dir /c_PREDICTIONS)

    run_duration = time.time() - start
    logger.info("Prediction job done.")
    logger.info(f"Prediction took {run_duration:.2f}s to execute")
    return None


if __name__ == '__main__':
    main()
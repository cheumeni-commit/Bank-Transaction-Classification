"""
Prediction pipeline
"""
import time
import logging

import pandas as pd

from src.predict.main import predict
from src.io import load_predictions_data
from src.config.directories import directories as dirs
from src.constants import c_PREDICTIONS_DATA

logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("Starting prediction job...")
    
    X = load_predictions_data(dirs.test_dir / c_PREDICTIONS_DATA)
    X = pd.DataFrame(X)

    predictions = predict(X)

    #save_predictions(predictions)

    run_duration = time.time() - start
    logger.info("Prediction job done.")
    logger.info(f"Prediction took {run_duration:.2f}s to execute")
    return predictions


if __name__ == '__main__':
    main()
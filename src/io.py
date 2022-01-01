import logging
import joblib  # or pickle.
import json

import pandas as pd

from src.cli import context
from src.constants import (c_DATASET,
                           c_SAVE_METRICS,
                           c_SAVE_MODEL
                          )

logger = logging.getLogger(__name__)


def get_data():
    logger.info("Loading data")

    text_data = load_dataset()

    logger.info("Data catalog loaded. ✅")
    return {'text_data': text_data}   
  
  
def save_dataset(dataset, *, path):
    dataset.to_csv(path, index=False)
    logger.info(f"Dataset saved at {path.relative_to(context.dirs.root_dir)}")


def load_dataset():
    return pd.read_csv(context.dirs.inputs / c_DATASET,\
                    encoding = "ISO-8859-1",
                    sep=';'
                    )


def save_model(model, *, path):
    logger.info("Model are saved")
    return joblib.dump(model, path)


def save_metrics(metrics, *, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
        

def save_lexique(lexique, *, path):
    with open(path, 'w') as f:
        json.dump(lexique, f, indent=2)


def save_metrics_per_class(metrics, *, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_training_output(output, *, directory):
    save_model(output['model'], path = str(directory) + '/' + c_SAVE_MODEL)
    save_metrics(output['metrics'], path = str(directory) + '/' + c_SAVE_METRICS)
    

def load_model(path, default=None):
    try:
        return joblib.load(path)
    except ValueError:
        if default is None:
             default = {}
        return default
    

def load_json_file(path, default=None):
    try:
        with open(path) as fp:
            return json.load(fp)
    except ValueError:
        if default is None:
             default = {}
        return default



def save_prediction(predictions,*, path):
    #https://docs.python.org/3/library/json.html
    with open(path, 'w') as fp:
        json.dump(predictions, fp, indent=2)


from collections import defaultdict
import logging
from typing import List

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.config.config import get_config

_MODELS_REGISTRY_ = {'RandomForestClassifier': RandomForestClassifier,
                     'Xgb_Classifier': xgb.XGBClassifier,
                     'SGDClassifier': SGDClassifier
                     }

logger = logging.getLogger(__name__)


def _loadModels()-> List:

    Model = []
    name_model = []
    for _, v in get_config().model.items():
        Model.append(_MODELS_REGISTRY_[v.get('name')](**v.get('params')))
        name_model.append(v.get('name'))
    return Model, name_model


def get_model():
    
    """ Load Model """
    try:
        Model, name_model = _loadModels()
        print(Model)
    except:
        logger.info("The model is not available ")

    return Model, name_model
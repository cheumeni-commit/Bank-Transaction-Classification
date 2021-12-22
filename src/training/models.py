import logging
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.config.config import get_config

_MODELS_REGISTRY_ = {'RandomForest': RandomForestClassifier,
                     'Xgb_Classifier': xgb.XGBClassifier,
                     'SGDClassifier': SGDClassifier
                     }

logger = logging.getLogger(__name__)


def get_model():
    
    """ Load Model """
    try:
        Model = _MODELS_REGISTRY_[get_config().model.get('model').get('name')]
    except:
        logger.info("The model is not available ")

    return Model(**get_config().model.get('model').get('params'))
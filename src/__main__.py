import logging

from src.config.directories import directories as dirs
#from src.run_dataset import main as main_datatset
from src.run_train import main as main_train
from src.run_predict import main as main_predict
from src.training.models import get_model

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.debug("I a testing the logging config")
    main_train()
    main_predict()
    #get_model()

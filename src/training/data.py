import copy
import logging

import pandas as pd

from src.constants import (c_DETAIL_ECRITURE,
						   c_TARGET,
						   c_TEST_DATA,
						   c_TEXT_TRANSFORMES
                          )

from src.io import get_data


logger = logging.getLogger(__name__)


def _copy_dataSet(data:pd.DataFrame)->pd.DataFrame:
	return data.copy()
	

def _get_text_transactions(data:pd.DataFrame)->pd.DataFrame:
	"extract column of text (detail de l'ecriture) and Class"
	return pd.DataFrame(data[[c_DETAIL_ECRITURE, c_TARGET]], columns=[c_DETAIL_ECRITURE, c_TARGET])

	
def build_dataset():

	"Build dataset with text transformation"

	logger.info("Copy of initial Dataset")
	catalog = get_data()
	# Copy of dataset
	logger.info("Extraction of Class and text Column")
	dataset_copy = _copy_dataSet(catalog.get(c_TEST_DATA))
	# dataset with text transaction and class
	dataset = _get_text_transactions(dataset_copy)
	
	return dataset
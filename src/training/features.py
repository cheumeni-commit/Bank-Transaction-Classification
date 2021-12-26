import re

from src.constants import c_TEXT_TRANSFORMES


def _preprocess_test(data, column_ecriture):
	"""Conditional preprocessing on our text"""
	
	transaction = []
	if column_ecriture != None:
		for text in data[column_ecriture]:
			text = _preprocess(text)
			transaction.append(text)
		data[c_TEXT_TRANSFORMES] = transaction
	else: data[c_TEXT_TRANSFORMES] = _preprocess(data)
	return data


def _preprocess(text):
	# Lower
	text = text.lower()
	# Spacing and filters
	text = re.sub('[^a-zA-Z]', ' ', text)
	text = text.split()
	text = [word.lower() for word in text if len(word) > 1]
	text = ' '.join(text)
	return text


def build_train_test_set(data, *, column_ecriture=None):
    # add column text transform
    dataset = _preprocess_test(data, column_ecriture)
    return dataset
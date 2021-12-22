import joblib

import pandas as pd

from src.config.directories import directories as dirs
from src.constants import (c_SAVE_MODEL,
                           c_DETAIL_ECRITURE
                          )
from src.train import vectorizer
from src.training.features import build_train_test_set


def _load_model():
    loaded_model = joblib.load(str(dirs.raw_store_dir) + '/' + c_SAVE_MODEL)
    return loaded_model


def _model_inference(model, X, do_probabilities=False):

    X = predict_dataset(X, c_DETAIL_ECRITURE)

    if do_probabilities:
        pred = model.predict_proba(X)
    else:
        pred = model.predict(X)
    return pred


def predict(X, *, do_probabilities=False):
    model = _load_model()
    result = _model_inference(model, X, do_probabilities)
    return result


def _encoding(corpus):
    #vectorizer = Count_Vectorizer()
    #feature_selector = Select_Percentile()

    X_bow = vectorizer.transform(corpus).toarray()
    X_features = feature_selector.transform(X_bow)
    return X_features
    

def predict_transaction(transaction):
    corpus = build_train_test_set(transaction)
    X_features = _encoding(corpus)
    return X_features


def predict_dataset(dataset, transactionColumn):
    dataset = build_train_test_set(dataset, column_ecriture=transactionColumn)
    X = dataset.loc[:,transactionColumn].values
    X_features = _encoding(X)
    return X_features
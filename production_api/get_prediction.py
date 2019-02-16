"""
Usage:
    from get_prediction import stemmed_words, load_model, get_predictions
    # stemmed words must be imported for loading the model

    model = load_model()

    pred_idx, pred_probs = get_predictions(descriptions, model, 3)

Versions:
    sklearn: 0.20.2
    numpy: 1.16.1
    python: 3.6.8
    nltk: 3.4

"""


import re
from typing import Callable, Generator, Any

import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# ---------
# Constants
# ---------
MIN_DESC_LEN: int = 40  # The minimum length of the description in characters
MIN_PREDICTING_PROBA = (
    0.5
)  # the minimum probability for a class, to not be classified as not classified
MODEL_PATH = "/app/model.joblib"

# --------------------------------
# for loading the model via joblib
# --------------------------------

stemmer: SnowballStemmer = SnowballStemmer("german", ignore_stopwords=True)

analyzer: Callable = CountVectorizer().build_analyzer()


def stemmed_words(doc: np.str_) -> Generator[Any, None, None]:
    """Stemming function needed for loading model
    """

    return (stemmer.stem(w) for w in analyzer(doc))


def load_model(path: str = None):
    """Loads the model

    Returns:
        The loaded model
    """

    model = joblib.load(path or MODEL_PATH)
    return model


# ------------------
# Data preprocessing
# ------------------


def _preprocess_data(desc: np.ndarray) -> np.ndarray:
    """function for preproccessing data, removes contact data via regex


    Arguments:
        desc {np.array} -- the descriptions of the cases

    Returns:
        np.array -- preprocessed desc with the same size as disc
    """
    # -------------------
    # remove contact data
    # -------------------
    remove_emails = np.vectorize(
        lambda x: re.sub(r"\S*@\S*\s?", "", re.sub("email:", "", re.sub("e-mail:", "", x.lower())))
    )
    remove_telephones = np.vectorize(
        lambda x: re.sub(
            r"(\(?([\d \-\)\–\+\/\(]+)\)?([ .-–\/]?)([\d]+))",
            "",
            re.sub("tel.:", "", re.sub("telefon:", "", x.lower())),
        )
    )
    remove_links = np.vectorize(
        lambda x: re.sub(
            "http://",
            "",
            re.sub(
                "https://",
                "",
                re.sub(r"www.[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", x.lower()),
            ),
        )
    )
    desc = remove_links(remove_telephones(remove_emails(desc)))

    return desc


# ------------------
# Predicting Classes
# ------------------
def _predict_classes(desc: np.ndarray, model: Pipeline, n_preds: int) -> np.ndarray:
    """predict the classes of the desc

    Arguments:
        desc {np.array} -- The *preprocessed* descriptions
        model {sklPipeline}

    Returns:
        np.array -- A np.array with shape (len(4, desc)) with ints for the classes
             [0] -- The classes (including not classified)
             [1] -- The probability for belonging to class 0
             [2] -- The probability for belonging to class 1
             [3] -- The probability for belonging to class 2

    Classes:
        0 -- Verkehrsunfall, Feuer
        1 -- Raub, Einbruch, Vandalismus - Generell: 'mittlere Kriminalität'
        2 -- Drogen, Mord - Auch Alkohol wird hierzu klassifiziert
        3 -- Unclassified
    """
    if len(desc[0]) < MIN_DESC_LEN:
        return [], []

    pred = model.predict_proba(desc)[0]
    pred_idx = np.argsort(-pred)[:n_preds]
    if pred[pred_idx[0]] < MIN_PREDICTING_PROBA:
        return [], []
    pred = pred[pred_idx]
    return pred_idx.tolist(), pred.tolist()


# ----------------------
# Puting it all together
# ----------------------


def get_predictions(desc: np.ndarray, model, n_classes: int) -> np.ndarray:
    """Get the predictions

    Arguments:
        desc {np.array} -- [description]

    Returns:
        np.array -- [description]
    """

    desc = _preprocess_data(desc)
    pred = _predict_classes(desc, model, n_classes)
    return pred

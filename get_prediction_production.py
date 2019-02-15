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
MIN_PREDICTING_PROBA = 0.6  # the minimum probability for a class, to not be classified as not classified
MODEL_PATH = "/home/peer/Code/AI/praktikum/production/model.joblib"

# --------------------------------
# for loading the model via pickle
# --------------------------------

stemmer: SnowballStemmer = SnowballStemmer("german", ignore_stopwords=True)

analyzer: Callable = CountVectorizer().build_analyzer()


def stemmed_words(doc: np.str_) -> Generator[Any, None, None]:
    """Stemming function needed for loading model
    """

    return (stemmer.stem(w) for w in analyzer(doc))


def _load_model(path: str):
    """Loads the model
    
    Arguments:
        path {str} -- The path to the model
    
    Returns:
        The loaded model
    """

    model = joblib.load(path)
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
            "http://", "", re.sub("https://", "", re.sub(r"www.[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", x.lower()))
        )
    )
    desc = remove_links(remove_telephones(remove_emails(desc)))

    # ----------------------------------------
    # Make too short descriptions unclassified
    # ----------------------------------------
    str_len = np.vectorize(len)
    desc[np.where(str_len(desc) < MIN_DESC_LEN)] = "Not classified"

    return desc


# ------------------
# Predicting Classes
# ------------------
def _predict_classes(desc: np.ndarray, model: Pipeline) -> np.ndarray:
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
    pred = model.predict_proba(desc)
    pred_idx = np.argmax(pred, 1)
    pred_idx[np.where(pred[np.arange(len(pred_idx)), pred_idx] < MIN_PREDICTING_PROBA)] = 3
    pred_idx[np.where(desc == "Not classified")] = 3
    pred = np.concatenate((pred_idx.reshape((1, -1)), pred.T))

    return pred


# ----------------------
# Puting it all together
# ----------------------


def get_predictions(desc: np.ndarray) -> np.ndarray:
    """Get the predictions
    
    Arguments:
        desc {np.array} -- [description]
    
    Returns:
        np.array -- [description]
    """

    model = _load_model(MODEL_PATH)
    desc = _preprocess_data(desc)
    pred = _predict_classes(desc, model)
    return pred

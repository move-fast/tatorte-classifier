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
import os
import logging

import dill
import numpy as np

from configuration import MODEL_DIR, MIN_DESC_LEN, MIN_PREDICTING_PROBA
from tatorte_classifier.machine_learning.model import Model

logger = logging.getLogger(__name__)


# --------------------------------
# loading the model
# --------------------------------
def load_model(model_id) -> Model:
    """Loads the model

    Returns:
        The loaded model
    """

    model = None
    if os.path.isfile(f"{MODEL_DIR}/{model_id}"):
        model = dill.load(open(f"{MODEL_DIR}/{model_id}", "rb"))
    else:
        logger.warning(f"File {MODEL_DIR}/{model_id} does not exist")
        model = Model("sgd", {}, {})  # random model if no model exists
    return model


# ------------------
# Predicting Classes
# ------------------
def _predict_classes(desc: np.ndarray, model: Model, n_preds: int) -> np.ndarray:
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

    pred = model.pipeline.predict_proba(desc)[0]
    pred_idx = np.argsort(-pred)[:n_preds]
    if pred[pred_idx[0]] < MIN_PREDICTING_PROBA:
        return [], []
    pred = pred[pred_idx]
    return pred_idx, pred


# ----------------------
# Puting it all together
# ----------------------


def get_predictions(desc: np.ndarray, model: Model, n_classes: int) -> np.ndarray:
    """Get the predictions

    Arguments:
        desc {np.array} -- [description]

    Returns:
        np.array -- [description]
    """

    pred = _predict_classes(desc, model, n_classes)
    return pred
